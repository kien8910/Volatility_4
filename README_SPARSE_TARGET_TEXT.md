# Sparse target-company text integration

This pilot keeps the selected Step 4 G5 stock model frozen and learns only a small,
bounded late correction for horizons 1 and 5. It treats FinTexTS target-company
rows as **category-level summaries**, not individual articles.

## Information and split rules

- Step 4 OOF folds 1–3 are analysis-train; fold 4 is validation.
- Targets crossing the next split boundary are purged.
- Locked test data are inaccessible to development modes and require an explicit command.
- The source has no publication timestamps. The default makes news available at the
  next forecast session (`news_lag_sessions: 1`). Do not set this to zero unless the
  server data have a justified after-close cutoff.
- Quarantined leakage hashes are excluded when the quarantine artifact exists.
- Current realized volatility, baseline error, spike flag, and utility labels are never model inputs.
- The current FinBERT path supplies hidden-state embeddings, not sentiment logits; no
  sentiment feature is fabricated.

## Models

- `M0_stock_only`: unchanged G5 prediction.
- `T1_all_target`: all valid target-company category items.
- `T2_hard_top1`: catalyst filter plus deterministic top-1 (primarily semantic novelty
  with the current timestamp-free/direct-target data).
- `T3_sparse_hurdle`: catalyst filter, learned top-k edge gate, ex-ante hurdle, and a
  horizon-specific correction bounded by `alpha_max`.
- Each real text variant has a matching within-day semantic-payload placebo.

The embedding builder tokenizes and expands long category summaries into chunks that
fit the configured FinBERT `max_length`; it never silently relies on encoder truncation.

## Server commands

Run these in order. Development commands do not read locked-test rows.

```bash
python -m src.sparse_target_text.run_sparse_target_text \
  --config configs/sparse_target_text.yaml --mode validate-data

python -m src.sparse_target_text.run_sparse_target_text \
  --config configs/sparse_target_text.yaml --mode build-embedding-cache --device cuda

python -m src.sparse_target_text.run_sparse_target_text \
  --config configs/sparse_target_text.yaml --mode run-placebo --device cuda

python -m src.sparse_target_text.run_sparse_target_text \
  --config configs/sparse_target_text.yaml --mode train-validation --device cuda

python -m src.sparse_target_text.run_sparse_target_text \
  --config configs/sparse_target_text.yaml --mode select-config
```

Only after `selected_variant.yaml` is frozen:

```bash
python -m src.sparse_target_text.run_sparse_target_text \
  --config configs/sparse_target_text.yaml --mode build-locked-test-cache --device cuda

python -m src.sparse_target_text.run_sparse_target_text \
  --config configs/sparse_target_text.yaml --mode evaluate-locked-test --device cuda
```

The locked-test command verifies both the frozen config fingerprint and checkpoint
SHA-256 before evaluation.
