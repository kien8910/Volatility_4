# Step 7 — Stock-Specific News Impact Pilot

This is a pilot implementation for checking whether stock-specific gates can suppress harmful Step 6 news corrections. It is intentionally not a full grid.

The default config [configs/step7_stock_specific_news.yaml](configs/step7_stock_specific_news.yaml) uses:

- one seed: `42`
- small model set: `S0_StockOnly_G5`, `S2_FixedSmallGate`, `S5_UtilityFactorizedGate`
- small fixed-gate probabilities: `0.03`, `0.05`
- one loss/time-aware gate, `S5_UtilityFactorizedGate`, which uses historical stock/news loss context and auxiliary utility labels
- balanced per-hierarchy event caps, so macro, sector, target-company, and related-company events are all represented
- capped event-stock pairs
- no locked-test evaluation
- no full placebo suite unless requested

## Required previous artifacts

- `results/step6/predictions_validation.parquet`
- `results/step6/news_features.parquet` or Step 0 panel/news files
- `results/step4/best_static_graph_config.yaml`

Step 7 reads Step 6 validation predictions and chooses the least harmful Step 6 news branch as a correction proxy. If Step 6 selected stock-only, the pilot can still run with zero correction proxy, but it will mostly test protective-gate plumbing.

## Run pilot

```bash
python -m src.stock_news_impact.run_step7 \
  --config configs/step7_stock_specific_news.yaml \
  --mode pilot \
  --device cuda \
  --resume \
  --force-rebuild
```

Even smaller smoke run:

```bash
python -m src.stock_news_impact.run_step7 \
  --config configs/step7_stock_specific_news.yaml \
  --mode pilot \
  --device cuda \
  --max-runs 2 \
  --max-pairs 30000 \
  --seeds 42 \
  --max-epochs 5
```

CPU smoke run:

```bash
python -m src.stock_news_impact.run_step7 \
  --config configs/step7_stock_specific_news.yaml \
  --mode pilot \
  --device cpu \
  --max-runs 2 \
  --max-pairs 10000 \
  --max-epochs 3
```

## Step-by-step modes

```bash
python -m src.stock_news_impact.run_step7 --config configs/step7_stock_specific_news.yaml --mode validate-data
python -m src.stock_news_impact.run_step7 --config configs/step7_stock_specific_news.yaml --mode build-events
python -m src.stock_news_impact.run_step7 --config configs/step7_stock_specific_news.yaml --mode build-event-stock-pairs
python -m src.stock_news_impact.run_step7 --config configs/step7_stock_specific_news.yaml --mode build-features
python -m src.stock_news_impact.run_step7 --config configs/step7_stock_specific_news.yaml --mode train-gate --device cuda --resume
python -m src.stock_news_impact.run_step7 --config configs/step7_stock_specific_news.yaml --mode run-placebos
python -m src.stock_news_impact.run_step7 --config configs/step7_stock_specific_news.yaml --mode select-config
```

## Outputs

- `data/processed/step7_news_events.parquet`
- `data/processed/step7_event_stock_pairs.parquet`
- `results/step7/selected_news_branch.yaml`
- `results/step7/abnormal_volatility_response.parquet`
- `results/step7/gate_features.parquet`
- `results/step7/utility_labels_train.parquet`
- `results/step7/event_stock_gate_values.parquet`
- `results/step7/event_stock_corrections.parquet`
- `results/step7/predictions_validation.parquet`
- `results/step7/metrics_by_model.csv`
- `results/step7/metrics_by_ticker.csv`
- `results/step7/metrics_by_horizon.csv`
- `results/step7/metrics_by_fold_seed.csv`
- `results/step7/oracle_diagnostics.csv`
- `results/step7/did_diagnostics.csv`
- `results/step7/common_news_impact.csv`
- `results/step7/common_news_impact.parquet`
- `results/step7/placebo_results.csv`
- `reports/step7_stock_specific_news_report.md`

## Interpretation

Use this pilot only to decide whether Step 7 is worth expanding. A promising pilot should show:

- gated model close to or better than `S0_StockOnly_G5`;
- gates lower than naive always-on news correction;
- reasonable gate distribution, not all 1;
- real assignment better than simple placebo diagnostics.

If the best gated model is still worse than stock-only, keep the final model as Step 4 G5.

## Macro/sector common-impact diagnostic

Macro and sector events are broadcast to all 11 semiconductor stocks, but their impact is evaluated cross-sectionally rather than treated as 11 independent firm-specific events.

`common_news_impact.csv` reports, by model/config/hierarchy/horizon:

- number of common events;
- average common gated correction across stocks;
- average absolute cross-stock correction;
- same-sign rate across stocks;
- commonality ratio: `abs(mean correction across stocks) / mean(abs correction across stocks))`;
- mean utility and abnormal response.

A macro/sector event looks more like a real common industry shock when many stocks move in the same direction and the commonality ratio is high. This is still a predictive diagnostic, not causal evidence.

## Loss/time-aware gating

`S5_UtilityFactorizedGate` is designed for per-news and per-time adaptive gating. It does not use the current validation target as an input feature. Instead, it learns from:

- historical ticker-level stock-only loss, news-adjusted loss, utility, and abnormal response;
- historical market/time-level loss and utility;
- historical macro/sector event utility across the stock universe;
- row-level utility labels during training;
- macro/sector common utility labels during training, so common news is judged by its cross-stock effect rather than by one ticker at a time.

The historical context is causal: only observations whose realized `target_date` is strictly before the current event `date` are used as features. Current event utility is used only as a training label inside the training fold.
