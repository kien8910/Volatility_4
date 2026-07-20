# FinTexTS Step-0 Data Audit

## Dataset provenance
See `data/raw/provenance.json`. Raw rows: 130400; processed rows: 130390.
## Schema
Required schema validated. Shape: 130390 rows, 124 columns.
## Panel completeness
100 tickers, 1304 dates, 2019-01-01–2023-12-29.
## OHLC validity
Invalid rows removed from modeling panel: 10; corporate-action candidates: 0.
## Date and market-calendar alignment
Weekend/date diagnostics generated locally; external alignment was not used to overwrite FinTexTS.
## Volatility targets
GK nonpositive: 2; RS nonpositive: 54. Raw and clipped values retained.
## News coverage
See coverage tables. Macro/sector/related coverage is separately deduplicated by date, hierarchy, category and hash.
## Filing persistence
Filings are treated as persistent company context, not daily events.
## Leakage checks
Target-date violations: 0. No backward fill is implemented. Manual content review remains pending.
## Temporal splits
- Fold 1: train 2019-01-01–2021-02-04, validation 2021-02-05–2021-07-30
- Fold 2: train 2019-01-01–2021-07-30, validation 2021-08-02–2022-01-24
- Fold 3: train 2019-01-01–2022-01-24, validation 2022-01-25–2022-07-19
- Fold 4: train 2019-01-01–2022-07-19, validation 2022-07-20–2023-01-11
- Locked test: 2023-01-12–2023-12-29 (252 observed dates)
## Known limitations
No volume, adjusted close, corporate-action field, publication time, article ID, or native ticker-sector label. Adjusted status and external date alignment remain unverified. Ticker-sector mapping remains unresolved without verified external sources.
## Go/No-Go decision
**CONDITIONAL GO**. Automated hard checks pass only when `hard_ok=true`; manual news review and independently verified historical sector mapping remain required. Dataset is suitable for a market-wide cross-stock relational graph. Sector-specific graph claims require a separately verified ticker-sector mapping.
