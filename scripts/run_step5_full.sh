#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/step5
echo "Step 5 full pipeline started at $(date -Iseconds)"
pytest -q tests/test_step5_*.py 2>&1 | tee logs/step5/unit_tests_full.log
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode full "$@" 2>&1 | tee logs/step5/full_pipeline.log
echo "Step 5 full pipeline finished at $(date -Iseconds)"

