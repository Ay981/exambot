# StudyBot

A Telegram bot that reads a PDF handout and generates study quizzes using Google Gemini.

## Features
- Secure config via `.env`
- PDF upload and text extraction (pdfplumber)
- Topic-focused quiz generation with Gemini (google-genai)
- Token-safe selection: choose a chapter or a page range
- Persistence: remembers the last uploaded PDF per user (SQLite + local files)
- Pretty responses using Markdown (safely escaped)
- Inline buttons: Generate Quiz, List Chapters, Toggle output format
- LaTeX mode: generate a .tex exam file you can compile in Overleaf

## Quick start

1. Create and activate a virtual environment (Linux/macOS):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

- Copy `.env.example` to `.env`
- Fill the values:
  - `GEMINI_API_KEY=...`
  - `TELEGRAM_TOKEN=...`
  - (optional) `GEMINI_MODEL=gemini-2.5-flash` (use `/models` to discover available names)

4. Run the bot:

```bash
python main.py
```

## Deploy/Host options

You have two common ways to run Telegram bots in production:

1) Polling on a VPS (simple)
- Keep `USE_WEBHOOK=false` (default) and run the bot as a background service.
- Example with systemd on Ubuntu/Debian:

```bash
# as root
cat >/etc/systemd/system/studybot.service <<'UNIT'
[Unit]
Description=StudyBot Telegram Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/StudyBot
EnvironmentFile=/home/ubuntu/StudyBot/.env
ExecStart=/home/ubuntu/StudyBot/venv/bin/python /home/ubuntu/StudyBot/main.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now studybot
journalctl -u studybot -f
```

2) Webhook on a container platform (scalable)
- Recommended for Cloud Run, Render, Railway, Fly.io, etc.
- Set in `.env` (or platform env variables):

```
USE_WEBHOOK=true
WEBHOOK_URL_BASE=https://<your-public-url>
WEBHOOK_PATH=/tg-webhook
HOST=0.0.0.0
PORT=8080
```

- Build and run with Docker locally:

```bash
docker build -t studybot:latest .
docker run --rm -p 8080:8080 \
  -e TELEGRAM_TOKEN=*** \
  -e GEMINI_API_KEY=*** \
  -e USE_WEBHOOK=true \
  -e WEBHOOK_URL_BASE=https://<public-url> \
  -e WEBHOOK_PATH=/tg-webhook \
  studybot:latest
```

- Cloud Run (example):
  - Build and push the image to a registry (e.g., Artifact Registry or Docker Hub).
  - Deploy allowing unauthenticated invocations.
  - Set env vars (`TELEGRAM_TOKEN`, `GEMINI_API_KEY`, `USE_WEBHOOK=true`, `WEBHOOK_URL_BASE=https://<service-url>`, `WEBHOOK_PATH=/tg-webhook`).
  - The bot will expose `PORT` and set the Telegram webhook automatically.

Notes:
- When using webhook mode, Telegram must be able to reach your service over HTTPS.
- If you change `WEBHOOK_URL_BASE` or `WEBHOOK_PATH`, redeploy so the webhook is updated.
- Do not commit your `.env` file or secrets.

## Gemini SDK migration (google-genai)
- This project uses the new `google-genai` SDK.
- If you see a model not found error, run `/models` in Telegram to list what your key supports and set `GEMINI_MODEL` accordingly (e.g., `gemini-2.5-pro`).

## Usage
- Send `/start` to see instructions.
- Upload a PDF.
- Generate a quiz:
  - `/quiz` — uses the whole (first N pages) context
  - `/quiz chapter 1 Logic` — focuses on "Chapter 1" and topic "Logic"
  - `/quiz pages 5-25 Logic` — focuses on pages 5 to 25 and topic "Logic"
- Check status: `/status` (what's loaded, page count, detected chapters)
- Use inline buttons for quick actions (Generate Quiz, List Chapters, Toggle format)

### LaTeX output
- Toggle to LaTeX via the button (Format: LaTeX) or command:

```
/format latex
```

When LaTeX is enabled, the bot sends a self-contained `exam.tex` file you can compile in Overleaf or locally.

## Notes
- Large PDFs: You can limit extraction with `PDF_MAX_PAGES` in `.env`.
- Chapters: The bot heuristically detects chapter headings like "Chapter 1" or "CHAPTER 1". If headings differ, use `pages a-b`.
- Persistence: The last PDF per user is tracked in `data/bot.db` and page texts in `data/<chat_id>_pages.json`.

### Security
- Keep API keys in env vars or `.env`. Never hardcode them in code or commit them.
- Prefer a secret-ish `WEBHOOK_PATH` (e.g., `/tg-<random>`).

## Security
Do not hard-code your API keys. Keep `.env` local and private.

## Next ideas
- Add `/answers` to generate keys to previous quiz
- Improve chapter detection for varied heading styles
- Stream responses for very long outputs
