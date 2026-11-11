#!/bin/bash
# One-time setup for MiniFlow on macOS.
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null; then
  echo "Python 3 not found. Install it with: brew install python"
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env - open it and paste your GROQ_API_KEY (optional)."
fi

echo ""
echo "Done. Start it with: ./run.sh"
