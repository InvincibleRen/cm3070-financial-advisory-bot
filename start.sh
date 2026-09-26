#!/usr/bin/env bash
# Start both the FastAPI backend and the Vite React dev server.
# Usage:   ./start.sh
# Stop:    Ctrl+C (kills both processes)

set -e
cd "$(dirname "$0")"

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
  echo "[start] Installing frontend dependencies..."
  (cd frontend && npm install)
fi

echo "[start] Starting FastAPI backend on :8000..."
uvicorn src.api.main:app --reload --port 8000 &
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
