#!/usr/bin/env bash
set -euo pipefail 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_URL="${1:-https://github.com/kaengreg/ispras_hate_verification.git}"
BRANCH="${2:-main}"
APP_DIR="${3:-ispras-llm-stand}"

VLLM_BASE_URL="${VLLM_BASE_URL:-}"
VLLM_API_KEY="${VLLM_API_KEY:-}"
HTTP_TIMEOUT="${HTTP_TIMEOUT:-60}"

need_cmd() { command -v "$1" >/dev/null 2>&1; }

need_cmd git || { echo "git not found"; exit 1; }
need_cmd docker || { echo "docker not found"; exit 1; }

docker compose version >/dev/null 2>&1 || { echo "docker compose not available"; exit 1; }

if [ ! -d "$APP_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
else
  git -C "$APP_DIR" fetch origin "$BRANCH"
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" pull --ff-only
fi

cp .env "$APP_DIR/" 2>/dev/null || true

cd "$APP_DIR"

if [ ! -f .env ] && [ -f "$SCRIPT_DIR/.env" ]; then
  cp "$SCRIPT_DIR/.env" .env
fi

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ ! -f .env ]; then
  : "${VLLM_BASE_URL:=http://localhost:8000}"
  : "${HTTP_TIMEOUT:=60}"

  cat > .env <<EOF
VLLM_BASE_URL=${VLLM_BASE_URL}
VLLM_API_KEY=${VLLM_API_KEY:-}
HTTP_TIMEOUT=${HTTP_TIMEOUT}
EOF

  if [ -z "${VLLM_API_KEY:-}" ]; then
    echo "Created $APP_DIR/.env (VLLM_API_KEY is empty). Edit it and set VLLM_API_KEY if required."
  fi
fi

docker compose up -d --build

echo "OK. Open: http://localhost:${HOST_PORT:-8000}"