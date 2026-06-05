# CoPE / FAFA Uncertainty Analysis

- Domain: `SynCPR/FAFA proxy`
- Category: `None`
- Count: `16384`

## Key Metrics

- `positive_score`: mean=0.676188, median=0.680217, p05=0.602657, p95=0.735870
- `hard_negative_score`: mean=0.678041, median=0.680186, p05=0.572651, p95=0.771885
- `margin`: mean=-0.001854, median=0.000288, p05=-0.114176, p95=0.100308
- `top10_entropy`: mean=2.297810, median=2.298103, p05=2.293533, p95=2.301095
- `positive_token_std`: mean=0.015703, median=0.014573, p05=0.005878, p95=0.029544
- `rank_within_shard`: mean=1.706238, median=1.000000, p05=1.000000, p95=4.000000
- `hit1_within_shard`: mean=0.501770, median=1.000000, p05=0.000000, p95=1.000000

## Hit

- `hit1_within_shard`: 50.177

## Correlations

- `margin_vs_hit1`: 0.7465330958366394
- `top10_entropy_vs_hit1`: 0.3241562843322754
- `positive_token_std_vs_hit1`: 0.23013973236083984
- `positive_token_std_vs_margin`: 0.24367128312587738
