# ITCPR-C Image Robustness Pilot

This pilot adapts the robustness benchmark idea to composed person retrieval.
It keeps the gallery image features clean and corrupts only query image, so it is cheap enough to run without retraining FAFA or CoPE.

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
| gaussian_noise_filter | 3 | FAFA | 34.378 | 55.177 | 63.488 | 44.220 | 0.739 | 0.795 |  |
| gaussian_noise_filter | 3 | FAFA+SynCPR-gated-rerank | 34.514 | 55.132 | 63.533 | 44.269 | 0.741 | 0.795 | 523 |
| motion_blur_filter | 3 | FAFA | 43.052 | 62.307 | 70.254 | 52.042 | 0.925 | 0.935 |  |
| motion_blur_filter | 3 | FAFA+SynCPR-gated-rerank | 43.097 | 62.262 | 70.209 | 52.071 | 0.926 | 0.936 | 527 |
| brightness_filter | 3 | FAFA | 44.187 | 63.851 | 71.571 | 53.491 | 0.949 | 0.961 |  |
| brightness_filter | 3 | FAFA+SynCPR-gated-rerank | 44.051 | 63.987 | 71.617 | 53.436 | 0.946 | 0.960 | 541 |
| jpeg_compression | 3 | FAFA | 42.779 | 64.668 | 71.798 | 52.918 | 0.919 | 0.951 |  |
| jpeg_compression | 3 | FAFA+SynCPR-gated-rerank | 42.825 | 64.668 | 71.798 | 52.938 | 0.920 | 0.951 | 547 |

## Key Takeaways

- Clean FAFA is R1=46.549, mAP=55.650.
- Worst tested corruption is `gaussian_noise_filter` severity 3: R1 drops by -12.171, mAP drops by -11.430.
- Mildest tested corruption is `brightness_filter` severity 3: R1 drops by -2.361, mAP drops by -2.159.
- Best gated-rerank movement is `gaussian_noise_filter` severity 3: R1 +0.136, mAP +0.050.
- Worst gated-rerank movement is `brightness_filter` severity 3: R1 -0.136, mAP -0.055.
- Current SynCPR reranker is not a robust correction module under image corruption; it is better treated as an ambiguity diagnostic until trained or gated on corrupted-query data.

## Interpretation

- RR is relative robustness against the clean FAFA baseline on the same query subset.
- Per-corruption robustness is more informative than a single aggregate score; the benchmark exposes which composed-query input channel is brittle.
- The stronger paper direction is now ITCPR-C: a robustness benchmark for composed person retrieval, with FAFA as a pretrained CPR model and CoPE/robustness benchmark ideas as methodology.
- A model-combination direction still exists, but it should be driven by measured failure modes rather than assumed fixes.
