#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="qwen-openwhispr-asr"
UNIT_NAME="${SERVICE_NAME}.service"
HOST="${QWEN_ASR_HOST:-127.0.0.1}"
PORT="${QWEN_ASR_PORT:-8179}"

cd "$APP_DIR"

if [[ "${1:-}" == "--serve" ]]; then
  shift
  exec uv run python main.py --warmup --host "$HOST" --port "$PORT" "$@"
fi

echo "Stopping old ${UNIT_NAME} if it exists..."
systemctl --user stop "$UNIT_NAME" >/dev/null 2>&1 || true
systemctl --user reset-failed "$UNIT_NAME" >/dev/null 2>&1 || true

echo "Killing leftover Qwen ASR processes if any..."
pkill -TERM -f "${APP_DIR}/.venv/bin/python3 main.py" >/dev/null 2>&1 || true
pkill -TERM -f "uv run python main.py --warmup --host ${HOST} --port ${PORT}" >/dev/null 2>&1 || true
sleep 1
pkill -KILL -f "${APP_DIR}/.venv/bin/python3 main.py" >/dev/null 2>&1 || true
pkill -KILL -f "uv run python main.py --warmup --host ${HOST} --port ${PORT}" >/dev/null 2>&1 || true

echo "Starting ${UNIT_NAME}..."
systemd-run \
  --user \
  --unit="$SERVICE_NAME" \
  --collect \
  --working-directory="$APP_DIR" \
  "$APP_DIR/run_qwen_asr.sh" --serve "$@"

echo "Waiting for http://${HOST}:${PORT}/health ..."
for _ in $(seq 1 120); do
  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    curl -sS "http://${HOST}:${PORT}/health"
    echo
    echo "OpenWhispr Server URL: http://${HOST}:${PORT}/v1"
    exit 0
  fi
  sleep 1
done

echo "Server did not become healthy within 120 seconds." >&2
echo "Check logs with: journalctl --user-unit=${UNIT_NAME} -n 80 --no-pager" >&2
exit 1
