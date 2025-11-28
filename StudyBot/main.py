import os
import re
import json
import logging
import io
import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import pdfplumber
from google import genai
from google.api_core.exceptions import NotFound as GoogleNotFound
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from telegram.helpers import escape_markdown
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# ----------------------
# Environment & Logging
# ----------------------
load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "bot.db")
PAGES_JSON_TEMPLATE = os.path.join(DATA_DIR, "{chat_id}_pages.json")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "30"))
# Webhook config (for hosting on Cloud Run/Render/etc.)
USE_WEBHOOK = os.getenv("USE_WEBHOOK", "false").lower() in {"1", "true", "yes"}
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_URL_BASE = os.getenv("WEBHOOK_URL_BASE", "")  # e.g., https://your-domain.com
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/tg-webhook") # path part only

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("studybot")

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

# Safety check
if not GEMINI_API_KEY or not TELEGRAM_TOKEN:
    print("Error: Keys not found in .env file! Set GEMINI_API_KEY and TELEGRAM_TOKEN.")
    # Don't exit for local tests; but production should exit

# ----------------------
# Gemini Setup (google-genai)
# ----------------------
genai_client: Optional[genai.Client] = None
current_model_name: Optional[str] = None
MODEL_CANDIDATES: List[str] = []

try:
    if GEMINI_API_KEY:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        # Client will read GEMINI_API_KEY from env if set
        genai_client = genai.Client()
except Exception as e:
    logger.warning(f"Failed to init GenAI client: {e}")
    genai_client = None

# Prepare fallback list; prefer .env value, then popular defaults
candidates = [
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini-2.5-pro",
    "gemini-pro",
]
seen = set()
MODEL_CANDIDATES = [m for m in candidates if m and not (m in seen or seen.add(m))]


def choose_model(preferred: Optional[str] = None) -> bool:
    """Pick a supported model name; verify against list() if possible."""
    global current_model_name
    if genai_client is None:
        return False
    names = []
    if preferred:
        names.append(preferred)
    names.extend([m for m in MODEL_CANDIDATES if m != preferred])
    allowed_base: Optional[set] = None
    try:
        models = genai_client.models.list()
        raw = [getattr(m, "name", "") for m in models]
        allowed_base = {n.split("/")[-1] for n in raw if n}
    except Exception as e:
        logger.warning(f"Could not list models: {e}")
    for name in names:
        short = name.split("/")[-1]
        if allowed_base is not None and short not in allowed_base:
            continue
        current_model_name = short
        logger.info(f"Using Gemini model: {name}")
        return True
    # If no candidate matched, fall back to the first available model from the account
    if allowed_base:
        # Prefer 2.5-flash, then 2.5-pro, then any *flash*, then any *pro*, else first
        preference = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-pro",
        ]
        for p in preference:
            if p in allowed_base:
                current_model_name = p
                logger.info(f"Using Gemini model (fallback): {p}")
                return True
        # fuzzy contains
        for candidate in sorted(allowed_base):
            if "flash" in candidate or "pro" in candidate:
                current_model_name = candidate
                logger.info(f"Using Gemini model (fallback fuzzy): {candidate}")
                return True
        # as last resort, pick the first one
        current_model_name = sorted(allowed_base)[0]
        logger.info(f"Using Gemini model (fallback first): {current_model_name}")
        return True
    current_model_name = None
    return False

# ----------------------
# DB Helpers
# ----------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_docs (
              chat_id TEXT PRIMARY KEY,
              pdf_path TEXT NOT NULL,
              pages_json_path TEXT NOT NULL,
              page_count INTEGER NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chapters (
              chat_id TEXT NOT NULL,
              chapter_label TEXT NOT NULL,
              start_page INTEGER NOT NULL,
              end_page INTEGER NOT NULL,
              PRIMARY KEY (chat_id, chapter_label)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                chat_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (chat_id, key)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def upsert_user_doc(chat_id: str, pdf_path: str, pages_json_path: str, page_count: int):
    conn = get_db()
    try:
        conn.execute(
            """
            INSERT INTO user_docs(chat_id, pdf_path, pages_json_path, page_count, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(chat_id) DO UPDATE SET
              pdf_path=excluded.pdf_path,
              pages_json_path=excluded.pages_json_path,
              page_count=excluded.page_count,
              updated_at=excluded.updated_at
            """,
            (chat_id, pdf_path, pages_json_path, page_count, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def write_chapters(chat_id: str, chapters: List[Tuple[str, int, int]]):
    conn = get_db()
    try:
        # clear existing rows for chat
        conn.execute("DELETE FROM chapters WHERE chat_id=?", (chat_id,))
        conn.executemany(
            "INSERT INTO chapters(chat_id, chapter_label, start_page, end_page) VALUES(?,?,?,?)",
            [(chat_id, label.lower(), s, e) for (label, s, e) in chapters],
        )
        conn.commit()
    finally:
        conn.close()


def read_doc_meta(chat_id: str) -> Optional[sqlite3.Row]:
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM user_docs WHERE chat_id=?", (chat_id,)).fetchone()
        return row
    finally:
        conn.close()


def read_chapter_range(chat_id: str, chapter_term: str) -> Optional[Tuple[int, int]]:
    term = chapter_term.strip().lower()
    # normalize e.g., "chapter 1", "1", "i"
    if not term.startswith("chapter"):
        term = f"chapter {term}"
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT start_page, end_page FROM chapters WHERE chat_id=? AND chapter_label=?",
            (chat_id, term),
        ).fetchone()
        if row:
            return int(row[0]), int(row[1])
        return None
    finally:
        conn.close()


def get_setting(chat_id: str, key: str, default: Optional[str] = None) -> Optional[str]:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE chat_id=? AND key=?",
            (chat_id, key),
        ).fetchone()
        return row[0] if row else default
    finally:
        conn.close()


def set_setting(chat_id: str, key: str, value: str) -> None:
    conn = get_db()
    try:
        conn.execute(
            """
            INSERT INTO settings(chat_id, key, value) VALUES(?,?,?)
            ON CONFLICT(chat_id, key) DO UPDATE SET value=excluded.value
            """,
            (chat_id, key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_output_format(chat_id: str) -> str:
    return (get_setting(chat_id, "output_format", "markdown") or "markdown").lower()


def set_output_format(chat_id: str, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt not in {"markdown", "latex"}:
        fmt = "markdown"
    set_setting(chat_id, "output_format", fmt)

# ----------------------
# PDF Helpers
# ----------------------

def extract_pages_text(pdf_path: str, max_pages: int = 0) -> Tuple[List[str], int]:
    """
    Extract text per page. Returns (pages_text_list, total_pages_extracted)
    max_pages=0 means extract all pages.
    """
    texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages
        page_limit = len(pages) if max_pages <= 0 else min(len(pages), max_pages)
        for idx, page in enumerate(pages[:page_limit]):
            try:
                t = page.extract_text() or ""
                texts.append(t)
            except Exception as e:
                logger.error(f"PDF page {idx+1} extract error: {e}")
                texts.append("")
    return texts, len(texts)


CHAP_RE = re.compile(r"^\s*(chapter|CHAPTER)\s+([0-9IVXLC]+)\b")


def build_chapter_index(pages_text: List[str]) -> List[Tuple[str, int, int]]:
    """Return list of (chapter_label, start_page, end_page), 1-based pages."""
    markers: List[Tuple[str, int]] = []
    for i, content in enumerate(pages_text, start=1):
        if not content:
            continue
        # look only at first ~10 lines to catch headings fast
        head = "\n".join(content.splitlines()[:10])
        m = CHAP_RE.search(head)
        if m:
            raw_num = m.group(2)
            label = f"Chapter {raw_num}"
            markers.append((label, i))
    chapters: List[Tuple[str, int, int]] = []
    for j, (label, start) in enumerate(markers):
        end = (markers[j + 1][1] - 1) if j + 1 < len(markers) else len(pages_text)
        chapters.append((label, start, end))
    return chapters


def save_pages_json(chat_id: str, pages: List[str]) -> str:
    path = PAGES_JSON_TEMPLATE.format(chat_id=str(chat_id))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pages, f)
    return path


def load_pages_json(chat_id: str) -> Optional[List[str]]:
    path = PAGES_JSON_TEMPLATE.format(chat_id=str(chat_id))
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load pages json for {chat_id}: {e}")
        return None

# ----------------------
# Telegram Helpers
# ----------------------

def split_text(text: str, limit: int = 3800) -> List[str]:
    chunks = []
    t = text or ""
    while len(t) > limit:
        # try to split on paragraph boundary
        idx = t.rfind("\n\n", 0, limit)
        if idx == -1:
            idx = t.rfind("\n", 0, limit)
        if idx == -1:
            idx = limit
        chunks.append(t[:idx].strip())
        t = t[idx:].lstrip()
    if t:
        chunks.append(t)
    return chunks


async def reply_markdown(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    # Escape only user-provided dynamic parts when needed outside this helper
    try:
        if update.message:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        elif update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        else:
            # As a fallback, send via bot
            await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        # Fallback to plain text if markdown fails
        if update.message:
            await update.message.reply_text(text)
        elif update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text(text)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


async def reply_long(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    for part in split_text(text):
        if update.message:
            await update.message.reply_text(part)
        elif update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text(part)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=part)

# ----------------------
# Bot Commands
# ----------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 Welcome to StudyBot!\n\n"
        "1) Upload a PDF handout.\n"
        "2) I'll process and remember it for you.\n"
        "3) Use /quiz to generate an exam.\n\n"
        "Examples:\n"
        "• /quiz\n"
        "• /quiz chapter 1 Logic and Philosophy\n"
        "• /quiz pages 5-25 Probability"
    )
    await reply_markdown(update, context, msg)
    await send_main_keyboard(update, context)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "ℹ️ Commands:\n"
        "• /start — intro\n"
        "• /status — show what's loaded\n"
        "• /models — list available Gemini models\n"
        "• /quiz [topic] — generate an exam\n"
        "• /quiz chapter N [topic] — focus on chapter N\n"
        "• /quiz pages a-b [topic] — focus on pages a..b\n"
        "• /answers — reveal the answer key for the last quiz\n"
        "Then upload a PDF to update context."
    )
    await reply_markdown(update, context, msg)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    meta = read_doc_meta(chat_id)
    if not meta:
        await update.message.reply_text("No PDF uploaded yet.")
        return
    page_count = int(meta["page_count"]) if meta else 0
    # Count chapters
    conn = get_db()
    try:
        rows = conn.execute("SELECT chapter_label, start_page, end_page FROM chapters WHERE chat_id=? ORDER BY start_page", (chat_id,)).fetchall()
    finally:
        conn.close()
    if rows:
        ch_lines = [f"• {r['chapter_label'].title()} (pp. {r['start_page']}-{r['end_page']})" for r in rows]
        ch_txt = "\n".join(ch_lines)
    else:
        ch_txt = "(No chapters detected — try /quiz pages a-b)"
    msg = (
        f"✅ PDF loaded. Pages: {page_count}.\n"
        f"Chapters detected:\n{ch_txt}"
    )
    await update.message.reply_text(msg)
    await send_main_keyboard(update, context)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    doc = update.message.document
    if not doc:
        await update.message.reply_text("Please upload a PDF document.")
        return
    if doc.mime_type != "application/pdf" and not doc.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("Only PDF files are supported.")
        return

    status_msg = await update.message.reply_text("📥 Downloading and processing your PDF...")

    # Save under data/
    pdf_path = os.path.join(DATA_DIR, f"{chat_id}_uploaded.pdf")
    tg_file = await doc.get_file()
    await tg_file.download_to_drive(pdf_path)

    # Extract text per page
    pages_text, count = extract_pages_text(pdf_path, max_pages=PDF_MAX_PAGES)

    # Build simple chapter index
    chapters = build_chapter_index(pages_text)

    # Persist
    pages_json_path = save_pages_json(chat_id, pages_text)
    upsert_user_doc(chat_id, pdf_path, pages_json_path, count)
    if chapters:
        write_chapters(chat_id, chapters)

    await context.bot.edit_message_text(
        chat_id=update.effective_chat.id,
        message_id=status_msg.message_id,
        text=(
            "✅ PDF processed! Now try:\n\n"
            "/quiz\n"
            "/quiz chapter 1 Logic and Philosophy\n"
            "/quiz pages 5-25 Probability\n\n"
            "Tip: Use /status to see detected chapters."
        ),
    )
    # Send quick action buttons
    await send_main_keyboard(update, context)


def parse_quiz_args(args: List[str]) -> Tuple[Optional[Tuple[int, int]], Optional[str], str]:
    """
    Returns (pages_range, chapter_term, topic)
    pages_range: (start, end) 1-based inclusive, or None
    chapter_term: e.g., "1" or "chapter 1", or None
    topic: remaining topic string
    """
    if not args:
        return None, None, "General Content"
    text = " ".join(args).strip()

    # pages a-b
    m = re.search(r"pages\s+(\d+)\s*[-–]\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        topic = re.sub(m.group(0), "", text).strip() or "General Content"
        return (min(a, b), max(a, b)), None, topic

    # chapter n
    m2 = re.search(r"chapter\s+([0-9IVXLC]+)", text, flags=re.IGNORECASE)
    if m2:
        ch = m2.group(1)
        topic = re.sub(m2.group(0), "", text).strip() or "General Content"
        return None, ch, topic

    return None, None, text or "General Content"


async def generate_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_msg = update.message or (update.callback_query.message if getattr(update, 'callback_query', None) else None)
    if genai_client is None:
        if target_msg:
            await target_msg.reply_text("AI not configured. Please set GEMINI_API_KEY in .env and restart.")
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="AI not configured. Please set GEMINI_API_KEY in .env and restart.")
        return

    chat_id = str(update.effective_chat.id)
    meta = read_doc_meta(chat_id)
    if not meta:
        if target_msg:
            await target_msg.reply_text("⚠️ You haven't uploaded a PDF yet!")
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ You haven't uploaded a PDF yet!")
        return

    pages_text = load_pages_json(chat_id)
    if not pages_text:
        if target_msg:
            await target_msg.reply_text("⚠️ I couldn't load your PDF text. Please upload it again.")
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ I couldn't load your PDF text. Please upload it again.")
        return

    page_count = int(meta["page_count"]) if meta else len(pages_text)

    pages_range, chapter_term, topic = parse_quiz_args(context.args)

    start_page, end_page = 1, min(page_count, len(pages_text))
    source_desc = "Full document"

    if chapter_term:
        rng = read_chapter_range(chat_id, chapter_term)
        if rng:
            start_page, end_page = rng
            source_desc = f"Chapter {chapter_term} (pp. {start_page}-{end_page})"
        else:
            if target_msg:
                await target_msg.reply_text(
                    f"Couldn't find Chapter {chapter_term}. Falling back to full document."
                )
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Couldn't find Chapter {chapter_term}. Falling back to full document.")

    if pages_range:
        a, b = pages_range
        if 1 <= a <= page_count and 1 <= b <= page_count and a <= b:
            start_page, end_page = a, b
            source_desc = f"Pages {a}-{b}"
        else:
            if target_msg:
                await target_msg.reply_text(
                    f"Invalid page range {a}-{b}. Using 1-{page_count}."
                )
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Invalid page range {a}-{b}. Using 1-{page_count}.")

    selection = "\n\n".join(pages_text[start_page - 1:end_page])
    # Token safety: trim to ~15000 chars
    selection = selection[:15000]

    topic_safe = escape_markdown(topic, version=1)
    await reply_markdown(update, context, f"🧠 Generating a Hard Exam on: *{topic_safe}*\n_Source: {source_desc}_\nThis takes ~10–20 seconds…")
    output_fmt = get_output_format(chat_id)
    if output_fmt == "latex":
        prompt = f"""
You are a strict university professor. Generate a Midterm Exam in LaTeX ONLY, using valid LaTeX markup, based ONLY on the text below.
Focus Area: {topic}

LaTeX Requirements:
- Provide a self-contained LaTeX document using article class.
- Use sections for Multiple Choice, Short Answers, and a closing Motivational Quote.
- Do NOT include answers.

Context from PDF (pages {start_page}-{end_page}):
{selection}
"""
    else:
        prompt = f"""
You are a strict university professor.
Generate a Midterm Exam based ONLY on the text provided below.
Focus Area: {topic}

Format:
1. 5 Multiple Choice Questions (tricky/application-based)
2. 2 Short Answer Questions
3. 1 Motivational Quote at the end (Islamic or General)
4. Do NOT provide the answers yet

Context from PDF (pages {start_page}-{end_page}):
{selection}
"""
    try:
        # Ensure we have a working model name; if not, try to choose one
        if not current_model_name and not choose_model(GEMINI_MODEL):
            raise RuntimeError("No available Gemini model. Set GEMINI_MODEL in .env to a valid model.")
        resp = genai_client.models.generate_content(model=current_model_name, contents=prompt)
        text = (getattr(resp, "text", None) or "(No response) ").strip()
        if not text:
            text = "(The AI returned an empty response.)"
        if output_fmt == "latex":
            # Send as a .tex file so users can compile
            buf = io.BytesIO(text.encode("utf-8"))
            buf.name = "exam.tex"
            if target_msg:
                await target_msg.reply_document(document=buf, filename="exam.tex", caption="LaTeX exam file")
            else:
                await context.bot.send_document(chat_id=update.effective_chat.id, document=buf, filename="exam.tex", caption="LaTeX exam file")
        else:
            await reply_long(update, context, text)
        # Stash last quiz so we can reveal answers later
        context.chat_data["last_quiz"] = {
            "format": output_fmt,
            "topic": topic,
            "start_page": start_page,
            "end_page": end_page,
            "source_desc": source_desc,
            "questions": text,
            # keep a shorter slice of selection to save tokens on answer generation
            "selection": selection[:10000],
        }
        if target_msg:
            await target_msg.reply_text("🗝 Tap the button below to reveal the answers.")
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🗝 Reveal Answers", callback_data="ANSWERS")]])
            await target_msg.reply_text("Answer key:", reply_markup=kb)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="� Tap the button below to reveal the answers.")
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🗝 Reveal Answers", callback_data="ANSWERS")]])
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Answer key:", reply_markup=kb)
    except GoogleNotFound as e:
        # Fallback to another model if current is unavailable
        logger.warning(f"Model not found or unsupported: {current_model_name}. Trying fallback…")
        # Try another model and retry once
        if choose_model(None):
            try:
                resp = genai_client.models.generate_content(model=current_model_name, contents=prompt)
                text = (getattr(resp, "text", None) or "(No response) ").strip()
                if output_fmt == "latex":
                    buf = io.BytesIO(text.encode("utf-8"))
                    buf.name = "exam.tex"
                    if target_msg:
                        await target_msg.reply_document(document=buf, filename="exam.tex", caption="LaTeX exam file")
                    else:
                        await context.bot.send_document(chat_id=update.effective_chat.id, document=buf, filename="exam.tex", caption="LaTeX exam file")
                else:
                    await reply_long(update, context, text)
                # Stash last quiz post-fallback
                context.chat_data["last_quiz"] = {
                    "format": output_fmt,
                    "topic": topic,
                    "start_page": start_page,
                    "end_page": end_page,
                    "source_desc": source_desc,
                    "questions": text,
                    "selection": selection[:10000],
                }
                if target_msg:
                    await target_msg.reply_text("🗝 Tap the button below to reveal the answers.")
                    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🗝 Reveal Answers", callback_data="ANSWERS")]])
                    await target_msg.reply_text("Answer key:", reply_markup=kb)
                else:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="🗝 Tap the button below to reveal the answers.")
                    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🗝 Reveal Answers", callback_data="ANSWERS")]])
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="Answer key:", reply_markup=kb)
                return
            except Exception:
                pass
        # If still failing, inform user
        msg = (
            "⚠️ The AI model name seems unsupported for your API key. "
            "Set GEMINI_MODEL in .env to a valid model (e.g., gemini-1.5-pro or gemini-pro)."
        )
        if target_msg:
            await target_msg.reply_text(msg)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    except RuntimeError as e:
        # Typically: No available Gemini model
        msg = str(e) or "No available Gemini model. Run /models and set GEMINI_MODEL in .env."
        if target_msg:
            await target_msg.reply_text(msg)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    except Exception as e:
        logger.exception("AI Error")
        if target_msg:
            await target_msg.reply_text("⚠️ The AI is overloaded or the text was too long. Try again later.")
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ The AI is overloaded or the text was too long. Try again later.")


# ----------------------
# Keyboards & Callbacks
# ----------------------

def build_main_keyboard(chat_id: str) -> InlineKeyboardMarkup:
    fmt = get_output_format(chat_id)
    fmt_label = "LaTeX" if fmt == "latex" else "Markdown"
    toggle_label = "Format: LaTeX" if fmt != "latex" else "Format: Markdown"
    rows = [
        [InlineKeyboardButton("🧪 Generate Quiz", callback_data="QUIZ")],
        [InlineKeyboardButton("📚 List Chapters", callback_data="LIST_CHAPTERS")],
        [InlineKeyboardButton("🗝 Reveal Answers", callback_data="ANSWERS")],
        [InlineKeyboardButton(toggle_label, callback_data="FORMAT_LATEX" if fmt != "latex" else "FORMAT_MD")],
    ]
    return InlineKeyboardMarkup(rows)


async def send_main_keyboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    kb = build_main_keyboard(chat_id)
    await update.message.reply_text("Quick actions:", reply_markup=kb)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = str(update.effective_chat.id)
    data = query.data or ""

    if data == "QUIZ":
        # Run default quiz on current selection (full doc)
        # Reuse command handler by invoking with no args
        context.args = []
        await generate_quiz(update, context)
        return

    if data == "LIST_CHAPTERS":
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT chapter_label, start_page, end_page FROM chapters WHERE chat_id=? ORDER BY start_page LIMIT 12",
                (chat_id,),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            await query.edit_message_text("No chapters detected. Try /quiz pages a-b.")
            return
        buttons = [
            [InlineKeyboardButton(f"{r['chapter_label'].title()} (pp.{r['start_page']}-{r['end_page']})", callback_data=f"CHAPTER::{r['chapter_label']}")]
            for r in rows
        ]
        await query.edit_message_text("Choose a chapter:", reply_markup=InlineKeyboardMarkup(buttons))
        return

    if data.startswith("CHAPTER::"):
        chapter_term = data.split("::", 1)[1]
        # Invoke quiz generation focused on this chapter
        # Simulate args: ["chapter", term]
        context.args = ["chapter", chapter_term.replace("chapter", "", 1).strip()]
        await generate_quiz(update, context)
        return

    if data == "FORMAT_LATEX":
        set_output_format(chat_id, "latex")
        await query.edit_message_text("Output format set to LaTeX.")
        return
    if data == "FORMAT_MD":
        set_output_format(chat_id, "markdown")
        await query.edit_message_text("Output format set to Markdown.")
        return

    if data == "ANSWERS":
        # Defer to a shared handler so /answers reuses logic
        await answers_cmd(update, context)
        return


async def format_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if context.args:
        arg = context.args[0].lower()
        if arg in {"latex", "md", "markdown"}:
            set_output_format(chat_id, "latex" if arg == "latex" else "markdown")
    fmt = get_output_format(chat_id)
    await update.message.reply_text(f"Output format: {fmt.title()}.")


async def models_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List available models for your API key with generateContent capability."""
    try:
        if genai_client is None:
            await update.message.reply_text("No GEMINI_API_KEY configured or client init failed.")
            return
        items = genai_client.models.list()
        names = []
        for m in items:
            full = getattr(m, "name", "<unknown>")
            base = full.split("/")[-1]
            names.append(base)
        if not names:
            await update.message.reply_text("No models listed. Your key may not have access in this region.")
            return
        # Compact output
        preview = "\n".join(names[:30])
        await update.message.reply_text(f"Models (first {min(30, len(names))} shown):\n{preview}\n\nSet GEMINI_MODEL in .env to one of these.")
    except Exception as e:
        await update.message.reply_text(f"Failed to list models: {e}")


async def answers_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reveal answer key for the last generated quiz."""
    target_msg = update.message or (update.callback_query.message if getattr(update, 'callback_query', None) else None)
    last = context.chat_data.get("last_quiz")
    if not last:
        if target_msg:
            await target_msg.reply_text("No quiz found. Generate one first with /quiz.")
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="No quiz found. Generate one first with /quiz.")
        return

    fmt = last.get("format", "markdown")
    topic = last.get("topic", "General Content")
    start_page = last.get("start_page", 1)
    end_page = last.get("end_page", 1)
    questions_text = last.get("questions", "")
    selection = last.get("selection", "")

    if fmt == "latex":
        prompt = f"""
You previously created an exam in LaTeX (no answers) from the PDF excerpt below.
Now generate a LaTeX Answer Key as a stand-alone article document.

Requirements:
- Title the document "Answer Key".
- Match the numbering and ordering of the questions exactly.
- For Multiple Choice, provide the correct option letter and a one-line justification.
- For Short Answers, provide a concise model answer (2-4 sentences).
- Do NOT repeat full questions verbatim; keep answers concise.

Focus Area: {topic}
Questions (LaTeX source):
{questions_text}

Context excerpt from PDF (pages {start_page}-{end_page}):
{selection}
"""
    else:
        prompt = f"""
You previously created an exam (questions only, no answers) from the PDF excerpt below.
Now produce a clean Answer Key in Markdown matching the same numbering and options.

Requirements:
- For Multiple Choice, show the correct letter and a brief justification.
- For Short Answers, give a concise model answer (2-4 sentences).
- Do NOT repeat full questions; list answers in order.

Focus Area: {topic}
Questions:
{questions_text}

Context excerpt from PDF (pages {start_page}-{end_page}):
{selection}
"""

    try:
        if not current_model_name and not choose_model(GEMINI_MODEL):
            raise RuntimeError("No available Gemini model. Set GEMINI_MODEL in .env to a valid model.")
        resp = genai_client.models.generate_content(model=current_model_name, contents=prompt)
        text = (getattr(resp, "text", None) or "(No response)").strip()
        if not text:
            text = "(The AI returned an empty response.)"
        if fmt == "latex":
            buf = io.BytesIO(text.encode("utf-8"))
            buf.name = "answers.tex"
            if target_msg:
                await target_msg.reply_document(document=buf, filename="answers.tex", caption="LaTeX answer key")
            else:
                await context.bot.send_document(chat_id=update.effective_chat.id, document=buf, filename="answers.tex", caption="LaTeX answer key")
        else:
            await reply_long(update, context, text)
    except GoogleNotFound:
        logger.warning(f"Model not found or unsupported: {current_model_name}. Trying fallback…")
        if choose_model(None):
            try:
                resp = genai_client.models.generate_content(model=current_model_name, contents=prompt)
                text = (getattr(resp, "text", None) or "(No response)").strip()
                if fmt == "latex":
                    buf = io.BytesIO(text.encode("utf-8"))
                    buf.name = "answers.tex"
                    if target_msg:
                        await target_msg.reply_document(document=buf, filename="answers.tex", caption="LaTeX answer key")
                    else:
                        await context.bot.send_document(chat_id=update.effective_chat.id, document=buf, filename="answers.tex", caption="LaTeX answer key")
                else:
                    await reply_long(update, context, text)
                return
            except Exception:
                pass
        msg = (
            "⚠️ The AI model name seems unsupported for your API key. "
            "Set GEMINI_MODEL in .env to a valid model (e.g., gemini-2.5-flash)."
        )
        if target_msg:
            await target_msg.reply_text(msg)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    except RuntimeError as e:
        msg = str(e) or "No available Gemini model. Run /models and set GEMINI_MODEL in .env."
        if target_msg:
            await target_msg.reply_text(msg)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    except Exception:
        logger.exception("AI Error (answers)")
        if target_msg:
            await target_msg.reply_text("⚠️ The AI is overloaded or the prompt was too long. Try again later.")
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ The AI is overloaded or the prompt was too long. Try again later.")


# ----------------------
# Main
# ----------------------

def main():
    init_db()
    print("🤖 Bot is starting…")
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN missing. Create .env and set TELEGRAM_TOKEN, then rerun.")
        return
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_cmd))
    application.add_handler(CommandHandler('status', status))
    application.add_handler(CommandHandler('format', format_cmd))
    application.add_handler(CommandHandler('models', models_cmd))
    application.add_handler(CommandHandler('answers', answers_cmd))
    application.add_handler(CommandHandler('quiz', generate_quiz))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    application.add_handler(CallbackQueryHandler(on_button))

    print("✅ Bot is running! Go to Telegram.")

    if USE_WEBHOOK:
        # Build full webhook URL
        base = WEBHOOK_URL_BASE.rstrip("/")
        path = WEBHOOK_PATH if WEBHOOK_PATH.startswith("/") else "/" + WEBHOOK_PATH
        webhook_url = f"{base}{path}"
        print(f"Using webhook at {webhook_url} (listen {HOST}:{PORT}{path})")
        application.run_webhook(
            listen=HOST,
            port=PORT,
            url_path=path.lstrip('/'),
            webhook_url=webhook_url,
            drop_pending_updates=True,
        )
    else:
        application.run_polling()


if __name__ == '__main__':
    main()
