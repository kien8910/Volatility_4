#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/step4
start_ts="$(date -Iseconds)"
echo "Step 4 validation started at ${start_ts}"
python scripts/check_environment.py | tee logs/step4/environment_validation.log
pytest -q tests/test_step4_*.py | tee logs/step4/unit_tests_validation.log
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode train-validation "$@" 2>&1 | tee logs/step4/train_validation.log
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode select-config "$@" 2>&1 | tee logs/step4/select_config.log
echo "Step 4 validation finished at $(date -Iseconds)"

