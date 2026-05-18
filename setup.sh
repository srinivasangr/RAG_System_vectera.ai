#!/usr/bin/env bash
# Convenience setup for macOS / Linux.
# What it does:
#   1. Creates a Python venv under app/.venv
#   2. Installs requirements
#   3. Checks app/.env exists and has the required keys
#   4. Runs the Snowflake schema bootstrap
#
# Re-runnable: each step is idempotent.

set -euo pipefail

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP="$ROOT/app"

echo "─── 1/4  Python venv ──────────────────────────────────────────"
if [ ! -d "$APP/.venv" ]; then
    python3 -m venv "$APP/.venv"
    echo "   created $APP/.venv"
else
    echo "   already exists"
fi
# shellcheck disable=SC1091
source "$APP/.venv/bin/activate"

echo
echo "─── 2/4  install requirements ─────────────────────────────────"
pip install --quiet --upgrade pip
pip install --quiet -r "$APP/requirements.txt"

echo
echo "─── 3/4  check .env ───────────────────────────────────────────"
if [ ! -f "$APP/.env" ]; then
    echo "   missing $APP/.env"
    echo "   copy app/.env.example to app/.env and fill in your keys, then re-run."
    cp "$APP/.env.example" "$APP/.env"
    echo "   (a fresh app/.env was created from the example for you)"
    exit 1
fi

missing=()
for k in SNOWFLAKE_ACCOUNT SNOWFLAKE_USER SNOWFLAKE_PASSWORD CEREBRAS_API_KEY; do
    val=$(grep -E "^${k}=" "$APP/.env" | head -n1 | cut -d= -f2- | tr -d '"' | xargs)
    if [ -z "$val" ] || [[ "$val" == *YOUR_* ]] || [[ "$val" == *PASTE_* ]]; then
        missing+=("$k")
    fi
done
if [ ${#missing[@]} -ne 0 ]; then
    echo "   the following required keys are empty or placeholder in app/.env:"
    for k in "${missing[@]}"; do echo "     - $k"; done
    exit 1
fi
echo "   all required keys present"

echo
echo "─── 4/4  Snowflake schema ─────────────────────────────────────"
cd "$APP"
python -m rag_system.storage.init_snowflake

echo
echo "✓ setup complete"
echo
echo "Next:"
echo "   cd app && source .venv/bin/activate"
echo "   python scripts/ingest_all_pending.py --no-vision    # ingest the 10 PDFs"
echo "   streamlit run rag_system/ui/streamlit_app.py        # open the chat UI"
