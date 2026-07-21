# Step 5 — EMA Stabilization and Regime-Aware Graph Bank

Step 5 extends the locked Step 4 model:

`G5__L22__small_tcn__mse__learned__k3__dg8`

It keeps the Step 4 backbone fixed and only varies EMA and regime graph-bank parameters.

## Inputs

- `data/processed/step3_residual_state.parquet`
- `data/processed/step3_residual_targets.parquet`
- `results/step3/oos_p_predictions.parquet`
- `results/step4/best_static_graph_config.yaml`

## Validate

```bash
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode validate-data
```

## Reproduce Step 4 baseline

```bash
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode reproduce-step4 \
  --device cuda \
  --num-workers 8 \
  --resume
```

## Pilot validation

```bash
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode train-validation \
  --device cuda \
  --num-workers 8 \
  --resume
```

Restrict models:

```bash
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode train-ema \
  --device cuda \
  --include-models S5-B0,S5-E \
  --resume
```

## Select, final train, locked test

```bash
python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode select-config

python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode train-final \
  --device cuda \
  --num-workers 8 \
  --resume

python -m src.regime_graph.run_step5 \
  --config configs/step5_regime_graph.yaml \
  --mode evaluate-test \
  --device cuda \
  --num-workers 8
```

## Shell wrappers

```bash
bash scripts/run_step5_validation.sh --device cuda --num-workers 8 --resume
bash scripts/run_step5_final.sh --device cuda --num-workers 8 --resume
sbatch scripts/step5_slurm.sh
```

## Outputs

- `results/step5/predictions_validation.parquet`
- `results/step5/predictions_test.parquet`
- `results/step5/residual_predictions.parquet`
- `results/step5/metrics_by_model.csv`
- `results/step5/metrics_by_ticker.csv`
- `results/step5/metrics_by_horizon.csv`
- `results/step5/metrics_by_regime.csv`
- `results/step5/metrics_by_fold_seed.csv`
- `results/step5/metrics_by_market_state.csv`
- `results/step5/ema_stability.csv`
- `results/step5/regime_usage.csv`
- `results/step5/regime_entropy.csv`
- `results/step5/graph_diversity.csv`
- `results/step5/graph_edges.csv`
- `results/step5/state_features.parquet`
- `results/step5/failures.csv`
- `results/step5/best_step5_config.yaml`
- `reports/step5_regime_graph_report.md`

## Notes

- The locked test is evaluated only after `best_step5_config.yaml` is selected from validation.
- EMA is updated only during training.
- State features use only information through the forecast origin.
- Graph/regime patterns are predictive diagnostics, not causal relations.

