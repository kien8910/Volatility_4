# FinTexTS Step-0

Run `python -m src.preprocessing.run_step0 --config configs/data.yaml`, then `pytest -q`.

The raw Hugging Face snapshot is immutable. Conflicts and invalid OHLC are quarantined; prices are never filled or interpolated. External OHLC and historical sector classification are deliberately not downloaded by the core pipeline.
