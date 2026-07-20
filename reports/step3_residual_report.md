# Step 3 Residual Report

- P model: HAR-Ridge
- Fallback to HAR-Ridge: Yes
- Residual construction: pseudo-out-of-sample expanding walk-forward.
- Failure rows: 0

## Overall Diagnostics

|    n |   median_acf_energy_reduction |   median_lfr_reduction |   median_variance_ratio |   max_abs_mean_residual |
|-----:|------------------------------:|-----------------------:|------------------------:|------------------------:|
| 8789 |                        0.3459 |                0.75325 |                0.871489 |               0.0629556 |

## Diagnostics By Split

| base_split   |    n |   median_acf_energy_reduction |   median_lfr_reduction |   median_variance_ratio |   max_abs_mean_residual |
|:-------------|-----:|------------------------------:|-----------------------:|------------------------:|------------------------:|
| test         | 2772 |                     0.342045  |               0.75126  |                0.913117 |               0.0443066 |
| train        |  473 |                     0.0237548 |               0.66721  |                0.881313 |               0.0629556 |
| validation   | 5544 |                     0.84963   |               0.826379 |                0.746934 |               0.0321301 |

## P Model QLIKE By Horizon

|   horizon |    qlike |
|----------:|---------:|
|         1 | 0.315885 |
|         5 | 0.352551 |
|        10 | 0.369383 |
|        22 | 0.435066 |

## Cross-Stock Dependency

- mean_abs_offdiag_residual_corr: 0.4549326843881496
- mean_abs_offdiag_raw_corr: 0.5311746196411827
- significant_lagged_pairs_p05: 30
- max_abs_lagged_corr: 0.5155998927124751

## Decision

GO