# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps for pdfplumber (depends on poppler via pdfminer.six pure python; no native extra needed, but add fonts/locale)
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
 && rm -rf /var/lib/apt/lists/* \
 && sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

WORKDIR /app

# Cache deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Default envs (override in deployment)
ENV HOST=0.0.0.0 \
    PORT=8080 \
    USE_WEBHOOK=false

EXPOSE 8080

CMD ["python", "main.py"]
