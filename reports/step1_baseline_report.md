# Step 1 Baseline Report

Group: Industry-level Semiconductor. Fixed tickers: ADI, AMAT, AMD, AVGO, INTC, KLAC, LRCX, MU, NVDA, QCOM, TXN.

## Data audit

- Target check rows: 14344
- Max absolute GK log-volatility recomputation error: 0.000e+00
- GK variance raw <= 0 count: 0

## Ticker observations

| ticker   |   observations | first_date          | last_date           |   missing_ohlc |   missing_logvol |
|:---------|---------------:|:--------------------|:--------------------|---------------:|-----------------:|
| ADI      |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |
| AMAT     |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |
| AMD      |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |
| AVGO     |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |
| INTC     |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |
| KLAC     |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |
| LRCX     |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |
| MU       |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |
| NVDA     |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |
| QCOM     |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |
| TXN      |           1304 | 2019-01-01 00:00:00 | 2023-12-29 00:00:00 |              0 |                0 |

## Test result

- Best baseline by test QLIKE: B4_HAR_Ridge at horizon 1, QLIKE=0.254904.
- Mean HAR-OLS QLIKE: 0.281277; Historical Mean QLIKE: 0.312222; Last Value QLIKE: 0.534564.
- Failure rows logged: 0.

## Decision

GO