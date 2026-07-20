# Leakage Report

Forecast origin: OHLC through day t and text dated ≤t predict volatility at t+h. Target-date violations: 0. No backward-fill is used. Filing is persistent context. Candidate rows are in quarantine. **Automated leakage checks passed, but manual content audit remains pending.** Publication-time and semantic look-ahead risk cannot be excluded. Main design `News_t → volatility_t+1` is conditionally usable after manual review.
