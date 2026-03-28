#!/bin/zsh

set -euo pipefail

PROJECT_DIR="/Users/andrey/study/smart-therm-bot"
BOT_PATTERN="python -m scripts.run_telegram_bot"

cd "$PROJECT_DIR"

if [[ -f ./.env ]]; then
  set -a
  source ./.env
  set +a
fi

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
  echo "TELEGRAM_BOT_TOKEN is not set in [`.env`](.env)" >&2
  exit 1
fi

pkill -9 -f "$BOT_PATTERN" 2>/dev/null || true
sleep 2

source ./.venv/bin/activate
exec python -m scripts.run_telegram_bot
