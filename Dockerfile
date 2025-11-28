# Root-level Dockerfile for Render
# Uses the StudyBot subdirectory as application source.

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps if needed (kept minimal here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY StudyBot/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY StudyBot /app

# Default env expectations (Render will inject real values)
# ENV USE_WEBHOOK=true
# ENV HOST=0.0.0.0
# Render sets PORT automatically

# Expose typical port (Render reads PORT env)
EXPOSE 8080

# Start the bot
CMD ["python", "main.py"]
