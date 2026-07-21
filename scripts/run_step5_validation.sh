#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/step5
echo "Step 5 validation started at $(date -Iseconds)"
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode validate-data "$@" 2>&1 | tee logs/step5/validate_data.log
pytest -q tests/test_step5_*.py 2>&1 | tee logs/step5/unit_tests_validation.log
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode train-validation "$@" 2>&1 | tee logs/step5/train_validation.log
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode select-config "$@" 2>&1 | tee logs/step5/select_config.log
echo "Step 5 validation finished at $(date -Iseconds)"

