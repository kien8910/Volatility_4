# Data Dictionary

`date` is row/feature date; `target_date_h*` is the future observed row date. OHLC are float64 and available after close. `log_return=ln(C_t/C_{t-1})`; GK and Rogers–Satchell raw variances preserve estimator output, clipped variants use configured epsilon, and `logvol=0.5 ln(variance)`. Text fields preserve hierarchy; missing text is empty with category masks. Filing is persistent context, not daily news. Targets are never features. News publication time is unknown, so same-day use risks leakage.
