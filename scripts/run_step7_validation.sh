#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/step7_stock_specific_news.yaml}"
DEVICE="${DEVICE:-cuda}"

python -m src.stock_news_impact.run_step7 \
  --config "${CONFIG}" \
  --mode pilot \
  --device "${DEVICE}" \
  --resume

