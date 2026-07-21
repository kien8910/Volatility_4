#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/step5
echo "Step 5 final/test started at $(date -Iseconds)"
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode train-final "$@" 2>&1 | tee logs/step5/train_final.log
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode evaluate-test "$@" 2>&1 | tee logs/step5/evaluate_test.log
echo "Step 5 final/test finished at $(date -Iseconds)"

