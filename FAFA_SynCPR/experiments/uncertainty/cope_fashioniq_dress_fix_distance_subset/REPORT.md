# CoPE / FAFA Uncertainty Analysis

- Domain: `FashionIQ`
- Category: `dress`
- Count: `64`

## Key Metrics

- `query_logvar_mean`: mean=-7.306854, median=-7.306854, p05=-7.306854, p95=-7.306854
- `query_std_mean`: mean=0.025902, median=0.025902, p05=0.025902, p95=0.025902
- `query_var_trace`: mean=0.515270, median=0.515270, p05=0.515270, p95=0.515270
- `target_logvar_mean`: mean=-8.000000, median=-8.000000, p05=-8.000000, p95=-8.000000
- `target_std_mean`: mean=0.018316, median=0.018316, p05=0.018316, p95=0.018316
- `target_var_trace`: mean=0.257635, median=0.257635, p05=0.257635, p95=0.257635
- `rank`: mean=3.015625, median=1.000000, p05=1.000000, p95=10.849998
- `hit10`: mean=0.937500, median=1.000000, p05=0.150000, p95=1.000000
- `hit50`: mean=1.000000, median=1.000000, p05=1.000000, p95=1.000000
- `top1_margin`: mean=0.035132, median=0.023919, p05=0.001411, p95=0.101585
- `top10_entropy`: mean=2.301872, median=2.301989, p05=2.300546, p95=2.302484
- `gt_distance`: mean=1.401604, median=1.385973, p05=1.284175, p95=1.510629
- `top1_distance`: mean=1.384222, median=1.376642, p05=1.281135, p95=1.490252

## Recall

- `R10`: 93.750
- `R50`: 100.000

## Correlations

- `query_std_mean_vs_rank`: -0.04388793930411339
- `query_std_mean_vs_hit10`: 0.04442616179585457
- `query_var_trace_vs_rank`: None
- `top10_entropy_vs_rank`: 0.20635069906711578
- `top1_margin_vs_hit10`: 0.1684475839138031
