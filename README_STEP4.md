# Step 4 — Static Residual Graph Learning

This step trains static graph models on the 11 semiconductor tickers using Step 3 residual artifacts. It does not create or assume any metrics until you run it on your server.

## 1. Required inputs

The following files must already exist:

- `data/processed/step3_residual_state.parquet`
- `data/processed/step3_residual_targets.parquet`
- `results/step3/oos_p_predictions.parquet`
- `data/splits/split_manifest.csv`
- `data/splits/expanding_folds.csv`

The fixed ticker order is:

`ADI, AMAT, AMD, AVGO, INTC, KLAC, LRCX, MU, NVDA, QCOM, TXN`

## 2. Environment setup

Using venv:

```bash
bash scripts/setup_environment.sh
```

Using conda:

```bash
conda env create -f environment.yml
conda activate semiconductor-step4-graph
python scripts/check_environment.py
```

## 3. Validate data only

```bash
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode validate-data
```

## 4. Validation search

CPU:

```bash
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode train-validation \
  --device cpu
```

GPU:

```bash
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode train-validation \
  --device cuda \
  --num-workers 8
```

Resume:

```bash
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode train-validation \
  --device cuda \
  --num-workers 8 \
  --resume
```

Then select the best validation configuration:

```bash
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode select-config
```

## 5. Final locked-test workflow

After validation selection:

```bash
python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode train-final \
  --device cuda

python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode reconstruction \
  --device cuda

python -m src.graph.run_step4 \
  --config configs/step4_static_graph.yaml \
  --mode evaluate-test \
  --device cuda
```

Or run the wrapper:

```bash
bash scripts/run_step4_final.sh --device cuda --num-workers 8
```

## 6. Full run

```bash
bash scripts/run_step4_full.sh --device cuda --num-workers 8
```

Slurm:

```bash
sbatch scripts/step4_slurm.sh
```

## 7. Outputs

When executed, the pipeline writes:

- `results/step4/predictions_validation.parquet`
- `results/step4/predictions_test.parquet`
- `results/step4/residual_predictions.parquet`
- `results/step4/metrics_by_model.csv`
- `results/step4/metrics_by_ticker.csv`
- `results/step4/metrics_by_horizon.csv`
- `results/step4/metrics_by_fold_seed.csv`
- `results/step4/masked_reconstruction.csv`
- `results/step4/graph_edges.csv`
- `results/step4/graph_stability.csv`
- `results/step4/failures.csv`
- `results/step4/best_static_graph_config.yaml`
- `reports/step4_static_graph_report.md`

Figures are written to `results/figures/step4_*.png`.

## 8. File map

- `src/graph/data_loader.py`: reads and audits Step 3 artifacts.
- `src/graph/panel_builder.py`: builds complete `[date, ticker]` panels and graph windows.
- `src/graph/datasets.py`: PyTorch datasets.
- `src/graph/scalers.py`: train-only standard scaling.
- `src/graph/temporal_encoder.py`: `TemporalLinear` and `SmallTCN`.
- `src/graph/adjacency.py`: identity, correlation, random, and learned static adjacency.
- `src/graph/graph_layers.py`: normalized GCN layer.
- `src/graph/models.py`: residual forecast and masked reconstruction models.
- `src/graph/losses.py`: MSE/Huber residual loss.
- `src/graph/metrics.py`: QLIKE and summary metrics.
- `src/graph/trainer.py`: train loop, AMP, early stopping, checkpointing.
- `src/graph/evaluator.py`: prediction flattening, metrics, figures, report decisions.
- `src/graph/graph_diagnostics.py`: graph edges, stability, graph plots.
- `src/graph/masked_reconstruction.py`: masked-stock reconstruction diagnostic.
- `src/graph/checkpointing.py`: save/load checkpoint and git hash capture.
- `src/graph/reproducibility.py`: seed and device utilities.
- `src/graph/run_step4.py`: CLI entry point.

## 9. Assumptions to check before running

- Step 3 was run successfully and its residual files are from the same ticker universe.
- `is_oos == 1` for all rows used in Step 4.
- `target_date > date` for every target row.
- The server has enough time for the full grid search. The default YAML is intentionally broad.
- Validation selection is completed before `evaluate-test`.
- Locked test results are inspected only once after `best_static_graph_config.yaml` has been written.

No graph edge should be interpreted as causal evidence.
