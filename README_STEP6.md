# Step 6 — Naive Hierarchical News Fusion

Step 6 tests whether fixed-weight FinTexTS news fusion improves the locked Step 4 stock-only G5 backbone.

The official stock-only backbone remains:

- `G5`
- `SmallTCN` residual temporal encoder
- learned static GCN graph
- selected from `results/step4/best_static_graph_config.yaml`
- checkpoints loaded from `checkpoints/step4/`

Step 6 intentionally does not use reliability gates, regime gates, event gates, oracle gates, placebo tests, or text-encoder fine-tuning.

## Main commands

Validate data:

```bash
python -m src.news.run_step6 --config configs/step6_naive_news.yaml --mode validate-data
```

Build/resume embedding cache:

```bash
python -m src.news.run_step6 --config configs/step6_naive_news.yaml --mode build-embeddings --device cuda --resume
```

Run validation grid:

```bash
python -m src.news.run_step6 --config configs/step6_naive_news.yaml --mode train-validation --device cuda --resume
```

Select validation config:

```bash
python -m src.news.run_step6 --config configs/step6_naive_news.yaml --mode select-config
```

Refit development and evaluate locked test:

```bash
python -m src.news.run_step6 --config configs/step6_naive_news.yaml --mode train-final --device cuda --resume
python -m src.news.run_step6 --config configs/step6_naive_news.yaml --mode evaluate-test --device cuda
```

Quick smoke grid:

```bash
python -m src.news.run_step6 --config configs/step6_naive_news.yaml --mode train-validation --quick-grid --text-model hashing-test --device cpu
```

`hashing-test` is deterministic and only intended for local pipeline smoke tests. The research configuration uses `ProsusAI/finbert`.

## Output

Step 6 writes to `results/step6/`:

- `predictions_validation.parquet`
- `predictions_test.parquet`
- `metrics_by_model.csv`
- `metrics_by_ticker.csv`
- `metrics_by_horizon.csv`
- `metrics_by_news_group.csv`
- `metrics_by_fold_seed.csv`
- `hierarchy_ablation.csv`
- `news_corrections.parquet`
- `news_coverage.csv`
- `embedding_statistics.csv`
- `trainable_parameters.csv`
- `failures.csv`
- `best_naive_news_config.yaml`

The report is written to `reports/step6_naive_news_report.md`.

## Schema assumptions

The wide Step 0 panel must contain `date`, `ticker`, and FinTexTS hierarchy columns such as `macro_category1`, `sector_category1`, `targetCompany_category1`, `relatedCompany_category1`, and filing columns.

If `data/processed/fintexts_news_long.parquet` exists, it is expected to have:

- `date`
- `ticker`
- `hierarchy`
- `category`
- `text`

If that long file is missing or unusable, Step 6 rebuilds hierarchy features from the Step 0 panel.

## Leakage rules enforced

- Forecast origin is `date=t`.
- Target date must be strictly greater than forecast origin; this is inherited from Step 4 target validation.
- No same-day `News_t -> v_t` target is created.
- No dynamic news forward/backward fill is performed.
- Text encoder/cache is frozen; only projection and fusion layers train in the main Step 6 experiment.
- Locked test is not used by `select-config`.

