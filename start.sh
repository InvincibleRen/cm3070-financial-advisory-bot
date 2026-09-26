#!/usr/bin/env bash
# One-step launcher: sets everything up (Python venv + backend deps + frontend
# deps) if needed, then starts the FastAPI backend and the Vite React dev server.
#
# Usage:   ./start.sh          (first run does the setup automatically)
# Stop:    Ctrl+C              (kills both servers)
#
# Requires Python 3.10+ and Node.js on PATH. Everything else is handled here.

set -e
cd "$(dirname "$0")"

VENV="venv"

# 1. Python 3.10+ check
if ! command -v python3 >/dev/null 2>&1; then
  echo "[start] ERROR: python3 not found on PATH. Install Python 3.10+ and retry." >&2
  exit 1
fi
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "[start] ERROR: Python 3.10+ required, found $(python3 -V). " >&2
  exit 1
fi

# 2. Create the virtual environment on first run
if [ ! -d "$VENV" ]; then
  echo "[start] Creating virtual environment in ./$VENV ..."
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 3. Install backend dependencies if anything is missing (fast no-op once installed)
if ! python -c 'import fastapi, uvicorn, httpx, xgboost, sklearn' >/dev/null 2>&1; then
  echo "[start] Installing backend dependencies (first run only)..."
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
fi

# 4. Install frontend dependencies if missing
if [ ! -d "frontend/node_modules" ] || [ ! -x "frontend/node_modules/.bin/vite" ]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "[start] ERROR: npm not found on PATH. Install Node.js and retry." >&2
    exit 1
  fi
  echo "[start] Installing frontend dependencies (first run only)..."
  (cd frontend && npm install)
fi

# 5. Launch both servers
echo "[start] Starting FastAPI backend on :8000..."
# --reload-dir src so the reloader watches only source, not venv/ or data/
uvicorn src.api.main:app --reload --reload-dir src --port 8000 &
PID_API=$!

echo "[start] Starting Vite dev server on :5173..."
(cd frontend && npm run dev) &
PID_VITE=$!

echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "  Press Ctrl+C to stop both."
echo ""

# Clean up both processes on exit
trap "kill $PID_API $PID_VITE 2>/dev/null; exit" INT TERM
wait
