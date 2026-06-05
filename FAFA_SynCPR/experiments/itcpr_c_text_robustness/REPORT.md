# ITCPR-C Text Robustness Pilot

This pilot adapts the robustness benchmark idea to composed person retrieval.
It keeps the gallery image features clean and corrupts only query captions, so it is cheap enough to run without retraining FAFA or CoPE.

## Research Framing

- Deep-research framing: test whether robustness benchmarking gives a cleaner contribution than forcing a direct FAFA-CoPE model fusion.
- Academic-pipeline framing: keep the experiment reproducible with fixed corruptions, fixed seed, fixed checkpoint, JSON output, and a markdown report.
- Karpathy-guidelines framing: run the smallest meaningful benchmark first before adding training-heavy components.

## Setup

- ITCPR root: `../data`
- FAFA checkpoint: `tuned_recall_at1_step.pt`
- Gallery cache: `experiments/fafa_pretrained/itcpr_fafa_features.pt`
- Queries evaluated: `2202`
- Gallery size: `20510`

## Results

| corruption | severity | method | R1 | R5 | R10 | mAP | RR@1 | RR@mAP | applied |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0 | FAFA | 46.549 | 66.167 | 73.252 | 55.650 | 1.000 | 1.000 |  |
| character_filter | 1 | FAFA | 40.009 | 61.081 | 68.483 | 49.823 | 0.860 | 0.895 |  |
| character_filter | 1 | FAFA+SynCPR-gated-rerank | 39.782 | 61.081 | 68.483 | 49.713 | 0.855 | 0.893 | 528 |
| character_filter | 3 | FAFA | 28.429 | 48.411 | 56.449 | 38.077 | 0.611 | 0.684 |  |
| character_filter | 3 | FAFA+SynCPR-gated-rerank | 28.247 | 48.501 | 56.449 | 37.983 | 0.607 | 0.683 | 545 |
| RemoveChar_filter | 1 | FAFA | 40.191 | 61.762 | 69.391 | 50.261 | 0.863 | 0.903 |  |
| RemoveChar_filter | 1 | FAFA+SynCPR-gated-rerank | 40.145 | 61.853 | 69.301 | 50.255 | 0.862 | 0.903 | 522 |
| RemoveChar_filter | 3 | FAFA | 29.791 | 50.318 | 57.720 | 39.492 | 0.640 | 0.710 |  |
| RemoveChar_filter | 3 | FAFA+SynCPR-gated-rerank | 29.791 | 50.363 | 57.720 | 39.486 | 0.640 | 0.710 | 546 |
| qwerty_filter | 1 | FAFA | 39.555 | 60.536 | 67.938 | 49.392 | 0.850 | 0.888 |  |
| qwerty_filter | 1 | FAFA+SynCPR-gated-rerank | 39.555 | 60.445 | 67.847 | 49.391 | 0.850 | 0.888 | 538 |
| qwerty_filter | 3 | FAFA | 28.520 | 48.638 | 55.904 | 38.107 | 0.613 | 0.685 |  |
| qwerty_filter | 3 | FAFA+SynCPR-gated-rerank | 28.610 | 48.592 | 55.904 | 38.150 | 0.615 | 0.686 | 537 |
| remove_space_filter | 1 | FAFA | 42.779 | 62.579 | 70.300 | 52.208 | 0.919 | 0.938 |  |
| remove_space_filter | 1 | FAFA+SynCPR-gated-rerank | 42.688 | 62.489 | 70.300 | 52.156 | 0.917 | 0.937 | 505 |
| remove_space_filter | 3 | FAFA | 32.516 | 53.179 | 60.627 | 42.195 | 0.699 | 0.758 |  |
| remove_space_filter | 3 | FAFA+SynCPR-gated-rerank | 32.607 | 53.179 | 60.763 | 42.245 | 0.700 | 0.759 | 379 |
| repetition_filter | 1 | FAFA | 45.731 | 66.303 | 72.661 | 55.155 | 0.982 | 0.991 |  |
| repetition_filter | 1 | FAFA+SynCPR-gated-rerank | 45.731 | 66.349 | 72.661 | 55.149 | 0.982 | 0.991 | 578 |
| repetition_filter | 3 | FAFA | 44.005 | 65.077 | 72.389 | 53.750 | 0.945 | 0.966 |  |
| repetition_filter | 3 | FAFA+SynCPR-gated-rerank | 43.688 | 65.032 | 72.434 | 53.604 | 0.939 | 0.963 | 661 |

## Key Takeaways

- Clean FAFA is R1=46.549, mAP=55.650.
- Worst tested corruption is `character_filter` severity 3: R1 drops by -18.120, mAP drops by -17.573.
- Mildest tested corruption is `repetition_filter` severity 1: R1 drops by -0.817, mAP drops by -0.495.
- Best gated-rerank movement is `qwerty_filter` severity 3: R1 +0.091, mAP +0.043.
- Worst gated-rerank movement is `repetition_filter` severity 3: R1 -0.318, mAP -0.146.
- Current SynCPR reranker is not a robust correction module under text noise; it is better treated as an ambiguity diagnostic until trained or gated on corrupted-query data.

## Interpretation

- RR is relative robustness against the clean FAFA baseline on the same query subset.
- Character-level corruptions are the clearest weakness: character swap, remove char, and QWERTY errors reduce R@1 much more than word repetition.
- The stronger paper direction is now ITCPR-C: a robustness benchmark for composed person retrieval, with FAFA as a pretrained CPR model and CoPE/robustness benchmark ideas as methodology.
- A model-combination direction still exists, but it should target text normalization or corruption-aware query composition before ranker correction.
