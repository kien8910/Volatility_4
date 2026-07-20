#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/step4
echo "Step 4 full pipeline started at $(date -Iseconds)"
python scripts/check_environment.py 2>&1 | tee logs/step4/environment_full.log
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode validate-data "$@" 2>&1 | tee logs/step4/validate_data.log
pytest -q tests/test_step4_*.py 2>&1 | tee logs/step4/unit_tests_full.log
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode full "$@" 2>&1 | tee logs/step4/full_pipeline.log
echo "Step 4 full pipeline finished at $(date -Iseconds)"

