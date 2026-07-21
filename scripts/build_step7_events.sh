#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/step7_stock_specific_news.yaml}"

python -m src.stock_news_impact.run_step7 \
  --config "${CONFIG}" \
  --mode build-events

python -m src.stock_news_impact.run_step7 \
  --config "${CONFIG}" \
  --mode build-event-stock-pairs

python -m src.stock_news_impact.run_step7 \
  --config "${CONFIG}" \
  --mode build-features

