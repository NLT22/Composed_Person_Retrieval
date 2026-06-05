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
- `rank`: mean=3.609375, median=1.000000, p05=1.000000, p95=12.000000
- `hit10`: mean=0.906250, median=1.000000, p05=0.000000, p95=1.000000
- `hit50`: mean=1.000000, median=1.000000, p05=1.000000, p95=1.000000
- `top1_margin`: mean=0.074993, median=0.055803, p05=0.002633, p95=0.178071
- `top10_entropy`: mean=2.299553, median=2.299911, p05=2.295405, p95=2.301951
- `gt_distance`: mean=1.385268, median=1.386459, p05=1.196093, p95=1.591491
- `top1_distance`: mean=1.338889, median=1.349424, p05=1.196093, p95=1.445475

## Recall

- `R10`: 90.625
- `R50`: 100.000

## Correlations

- `query_std_mean_vs_rank`: 0.0236163679510355
- `query_std_mean_vs_hit10`: -0.03198595345020294
- `query_var_trace_vs_rank`: 0.03639807179570198
- `top10_entropy_vs_rank`: 0.414817750453949
- `top1_margin_vs_hit10`: 0.22046023607254028
