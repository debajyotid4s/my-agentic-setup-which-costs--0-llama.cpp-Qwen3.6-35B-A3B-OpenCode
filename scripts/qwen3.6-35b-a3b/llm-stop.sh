#!/usr/bin/env bash
# llm-stop.sh — Gracefully stop llama-server, with safe escalation.
#
# Usage:
#   llm-stop.sh              # graceful stop (SIGTERM, wait 30s, then SIGKILL)
#   llm-stop.sh --force      # immediate kill (no graceful wait)
#   llm-stop.sh --port 8080  # kill by port instead of process name
set -euo pipefail

PORT="${LLM_PORT:-8080}"
TIMEOUT=30
GRACEFUL=true

case "${1:-}" in
  --force) GRACEFUL=false ;;
  --port)  PORT="${2:-$PORT}" ;;
  --help|-h)
    echo "Usage: $0 [--force] [--port PORT]"
    echo "  --force    skip graceful wait, kill immediately"
    echo "  --port     use different port (default: $PORT)"
    exit 0
    ;;
esac

# Find the llama-server process
if [[ "$GRACEFUL" == "true" ]]; then
  echo "Stopping llama-server (graceful, ${TIMEOUT}s timeout)..."
else
  echo "Force-killing llama-server..."
fi

PIDS=$(ss -tlnp "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)

if [[ -z "$PIDS" ]]; then
  # Fallback: try by process name
  PIDS=$(pgrep -f "llama-server.*--port $PORT" 2>/dev/null || true)
  if [[ -z "$PIDS" ]]; then
    PIDS=$(pgrep -f "llama-server" 2>/dev/null || true)
  fi
fi

if [[ -z "$PIDS" ]]; then
  echo "No llama-server process found on port $PORT."
  exit 0
fi

echo "Found PIDs: $PIDS"

if [[ "$GRACEFUL" == "true" ]]; then
  # Send SIGTERM for graceful shutdown
  for PID in $PIDS; do
    echo "  Sending SIGTERM to $PID..."
    kill -SIGTERM "$PID" 2>/dev/null || true
  done

  # Wait for processes to exit
  WAITED=0
  while [[ $WAITED -lt $TIMEOUT ]]; do
    ALIVE=""
    for PID in $PIDS; do
      if kill -0 "$PID" 2>/dev/null; then
        ALIVE="$ALIVE $PID"
      fi
    done
    if [[ -z "$ALIVE" ]]; then
      echo "Server stopped cleanly."
      exit 0
    fi
    sleep 1
    WAITED=$((WAITED + 1))
    # Show progress every 5 seconds
    if (( WAITED % 5 == 0 )); then
      echo "  Waiting... ($WAITED/${TIMEOUT}s)"
    fi
  done

  echo "Graceful shutdown timed out after ${TIMEOUT}s. Force-killing..."
fi

# Force kill
for PID in $PIDS; do
  if kill -0 "$PID" 2>/dev/null; then
    echo "  Sending SIGKILL to $PID..."
    kill -SIGKILL "$PID" 2>/dev/null || true
  fi
done

# Brief wait for port to be released
sleep 1

# Verify port is free
if ss -tlnp "sport = :$PORT" 2>/dev/null | grep -q "$PORT"; then
  echo "WARN: port $PORT still in use. Check for stale processes."
  exit 1
fi

echo "Port $PORT is free."
