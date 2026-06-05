# ITCPR-C Text Robustness Pilot

This pilot adapts the robustness benchmark idea to composed person retrieval.
It keeps the gallery image features clean and corrupts only query captions, so it is cheap enough to run without retraining FAFA or CoPE.

## Setup

- ITCPR root: `../data`
- FAFA checkpoint: `tuned_recall_at1_step.pt`
- Gallery cache: `experiments/fafa_pretrained/itcpr_fafa_features.pt`
- Queries evaluated: `32`
- Gallery size: `20510`

## Results

| corruption | severity | method | R1 | R5 | R10 | mAP | RR@1 | RR@mAP | applied |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0 | FAFA | 25.000 | 53.125 | 62.500 | 36.933 | 1.000 | 1.000 |  |
| character_filter | 1 | FAFA | 18.750 | 50.000 | 62.500 | 31.787 | 0.750 | 0.861 |  |
| character_filter | 1 | FAFA+SynCPR-gated-rerank | 18.750 | 50.000 | 62.500 | 31.788 | 0.750 | 0.861 | 10 |
| RemoveChar_filter | 1 | FAFA | 12.500 | 43.750 | 53.125 | 28.246 | 0.500 | 0.765 |  |
| RemoveChar_filter | 1 | FAFA+SynCPR-gated-rerank | 15.625 | 43.750 | 53.125 | 29.810 | 0.625 | 0.807 | 9 |

## Interpretation

- RR is relative robustness against the clean FAFA baseline on the same query subset.
- If corruption drops FAFA but the gated SynCPR reranker recovers R@1/mAP, that supports the robust-and-ambiguity-aware CPR direction.
- If reranking hurts under corruptions, keep the reranker as an uncertainty diagnostic rather than a deployed correction.
