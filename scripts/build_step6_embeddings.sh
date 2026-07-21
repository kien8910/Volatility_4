#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/step6_naive_news.yaml}"
DEVICE="${DEVICE:-cuda}"

python -m src.news.run_step6 \
  --config "${CONFIG}" \
  --mode build-embeddings \
  --device "${DEVICE}" \
  --resume

