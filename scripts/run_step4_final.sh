#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/step4
echo "Step 4 final/test started at $(date -Iseconds)"
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode train-final "$@" 2>&1 | tee logs/step4/train_final.log
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode reconstruction "$@" 2>&1 | tee logs/step4/reconstruction.log
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode evaluate-test "$@" 2>&1 | tee logs/step4/evaluate_test.log
echo "Step 4 final/test finished at $(date -Iseconds)"

