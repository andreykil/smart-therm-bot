FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-bot.txt ./requirements-bot.txt

RUN pip install --no-cache-dir -r requirements-bot.txt

COPY . .

RUN mkdir -p /app/data/runtime /app/data/indices /app/data/processed/chat

CMD ["python", "-m", "scripts.run_telegram_bot"]
