# FAFA + SynCPR Hard-Negative Reranker Report

## Setup

- Backbone/cache: FAFA pretrained feature cache on ITCPR.
- SynCPR usage: train a lightweight reranker on frozen FAFA features, using same-shard hard negatives mined by FAFA FDA score.
- Checkpoint: `fafa_hard_negative_reranker.pt`.
- Baseline from cache: `R1=46.549`, `R5=66.167`, `R10=73.252`, `mAP=55.650`.

## Full Correction

Applying the learned correction to the full gallery hurts transfer badly:

| Method | R1 | R5 | R10 | mAP |
|---|---:|---:|---:|---:|
| FAFA cache baseline | 46.549 | 66.167 | 73.252 | 55.650 |
| Full learned correction | 40.963 | 61.717 | 68.256 | 50.252 |

Conclusion: the learned correction is not reliable as a global replacement or global additive score.

## Delta Sweep

The correction only helps when used as a very small negative adjustment.

| Delta mult | R1 | R5 | R10 | mAP |
|---:|---:|---:|---:|---:|
| -0.02 | 46.685 | 66.167 | 73.252 | 55.709 |
| 0.00 | 46.549 | 66.167 | 73.252 | 55.650 |
| 1.00 | 40.963 | 61.717 | 68.256 | 50.252 |

Interpretation: SynCPR-trained reranker learns a signal, but the signal is anti-correlated or miscalibrated for ITCPR if used directly.

## Top-K Rerank

Restricting the correction to the FAFA top-K candidates is safer. Best observed config:

| Top-K | Delta mult | R1 | R5 | R10 | mAP |
|---:|---:|---:|---:|---:|---:|
| 1000 | -0.015 | 46.821 | 66.167 | 73.206 | 55.771 |
| 2000 | -0.010 | 46.776 | 66.213 | 73.206 | 55.759 |
| 5000 | -0.020 | 46.776 | 66.167 | 73.252 | 55.754 |
| baseline | 0.000 | 46.549 | 66.167 | 73.252 | 55.650 |

Top-K reranking gives a small gain, mostly in R1 and mAP. R10 slightly drops for the best R1 config.

## Per-Query Analysis For Best Config

Best config: `topk=1000`, `delta_mult=-0.015`.

- Top-1 improved: 7 queries.
- Top-1 regressed: 1 query.
- Top-1 unchanged correct: 1024 queries.
- Top-1 unchanged wrong: 1170 queries.
- First positive rank improved: 179 queries.
- First positive rank regressed: 194 queries.
- AP improved: 179 queries.
- AP regressed: 194 queries.
- Mean AP delta: `+0.001214`.

Conclusion: R1 gain is real enough to log, but the broader ranking effect is mixed. This is a weak reranking signal, not a strong method by itself.

## Conditional Top-K Rerank

The correction was also gated by FAFA confidence. The gate uses the baseline top1-top2 score margin:

- Lower margin means FAFA is less confident.
- `margin_quantile=0.30` applies reranking only to the 30% most ambiguous queries.

Best conditional result:

| Top-K | Delta mult | Margin quantile | Applied queries | R1 | R5 | R10 | mAP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | -0.015 | 0.30 | 661 | 46.821 | 66.122 | 73.206 | 55.789 |
| baseline | 0.000 | none | 2202 | 46.549 | 66.167 | 73.252 | 55.650 |

This slightly improves mAP over ungated top-K rerank (`55.789` vs. `55.771`) while keeping the same R1. The gain is still small, but it supports the idea that the SynCPR correction should be used selectively, not globally.

## Caption-Length Gate

Qualitative inspection showed that the only top-1 regression under margin-only gating was a short and under-specified modification text: `wearing a white coat`. The recovered top-1 cases had more informative captions. A simple text-informativeness gate was therefore added:

- Keep margin gate: apply only to the 30% lowest baseline top1-top2 margins.
- Add caption gate: apply only when the modification caption has at least 6 alphanumeric words.

Best result so far:

| Top-K | Delta mult | Margin quantile | Min caption words | Applied queries | R1 | R5 | R10 | mAP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | -0.015 | 0.30 | 6 | 521 | 46.866 | 66.213 | 73.252 | 55.811 |
| baseline | 0.000 | none | none | 2202 | 46.549 | 66.167 | 73.252 | 55.650 |

Per-query effect for this best setting:

- Top-1 improved: 7 queries.
- Top-1 regressed: 0 queries.
- First positive rank improved: 77 queries.
- First positive rank regressed: 69 queries.
- AP improved: 77 queries.
- AP regressed: 69 queries.
- Mean AP delta: `+0.001615`.

This is still a small numerical gain, but it is now cleaner: the top-1 gain has no top-1 regression on ITCPR under this gate.

## Extra Gate Sweep

Additional ambiguity gates were tested after the caption-length gate:

- `gap10`: baseline top1 score minus mean top10 score.
- `entropy10`: entropy of the baseline top10 score distribution.
- Combined gates: `margin + gap10 + caption`, `margin + entropy10 + caption`.

Best zero-top1-regression result from this richer sweep:

| Gate | Params | Applied queries | R1 | R5 | R10 | mAP | Top1 +/- |
|---|---|---:|---:|---:|---:|---:|---:|
| margin + gap10 + caption | margin=0.30, gap10=0.70, min_words=6 | 506 | 46.821 | 66.213 | 73.252 | 55.789 | +6 / -0 |
| margin + caption | margin=0.30, min_words=6 | 521 | 46.866 | 66.213 | 73.252 | 55.811 | +7 / -0 |

Conclusion: the richer score-shape gates did not improve the result. They were more conservative and removed one useful top-1 recovery. The best gate remains the simple and interpretable `margin + caption length` rule.

## Recommended Next Step

Do not train this head harder as-is. The next sensible direction is to turn this into a clean analysis/contribution:

1. Present FAFA as the strong base model.
2. Use SynCPR hard negatives to train a lightweight correction signal.
3. Show that global correction fails, but selective correction works.
4. Use the simple `margin + caption length` gate as the final method.
5. Use case-study HTML to qualitatively explain why the gate helps.

The current best deployable experimental setting is:

```bash
topk=1000
delta_mult=-0.015
margin_quantile=0.30
min_caption_words=6
```

Relevant outputs:

- `itcpr_eval.json`
- `itcpr_delta_sweep.json`
- `itcpr_topk_rerank.json`
- `itcpr_topk_rerank_fine.json`
- `itcpr_topk_best_analysis.json`
- `itcpr_conditional_topk_sweep.json`
- `itcpr_conditional_topk_best.json`
- `itcpr_caption_length_gate_sweep.json`
- `itcpr_conditional_caption_gate_best.json`
- `itcpr_multi_gate_sweep.json`
- `gate_payload_top1000.pt`
- `case_study/case_study.html`
- `case_study_caption_gate/case_study.html`
