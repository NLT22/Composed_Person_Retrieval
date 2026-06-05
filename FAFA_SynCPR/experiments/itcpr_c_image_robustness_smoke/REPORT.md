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
- Queries evaluated: `32`
- Gallery size: `20510`

## Results

| corruption | severity | method | R1 | R5 | R10 | mAP | RR@1 | RR@mAP | applied |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0 | FAFA | 25.000 | 53.125 | 62.500 | 36.933 | 1.000 | 1.000 |  |
| gaussian_noise_filter | 3 | FAFA | 18.750 | 43.750 | 53.125 | 27.941 | 0.750 | 0.757 |  |
| gaussian_noise_filter | 3 | FAFA+SynCPR-gated-rerank | 18.750 | 43.750 | 53.125 | 27.784 | 0.750 | 0.752 | 9 |
| motion_blur_filter | 3 | FAFA | 21.875 | 46.875 | 62.500 | 34.772 | 0.875 | 0.941 |  |
| motion_blur_filter | 3 | FAFA+SynCPR-gated-rerank | 21.875 | 46.875 | 62.500 | 34.777 | 0.875 | 0.942 | 10 |

## Key Takeaways

- Clean FAFA is R1=25.000, mAP=36.933.
- Worst tested corruption is `gaussian_noise_filter` severity 3: R1 drops by -6.250, mAP drops by -8.992.
- Mildest tested corruption is `motion_blur_filter` severity 3: R1 drops by -3.125, mAP drops by -2.161.
- Best gated-rerank movement is `gaussian_noise_filter` severity 3: R1 +0.000, mAP -0.156.
- Worst gated-rerank movement is `gaussian_noise_filter` severity 3: R1 +0.000, mAP -0.156.
- Current SynCPR reranker is not a robust correction module under image corruption; it is better treated as an ambiguity diagnostic until trained or gated on corrupted-query data.

## Interpretation

- RR is relative robustness against the clean FAFA baseline on the same query subset.
- Per-corruption robustness is more informative than a single aggregate score; the benchmark exposes which composed-query input channel is brittle.
- The stronger paper direction is now ITCPR-C: a robustness benchmark for composed person retrieval, with FAFA as a pretrained CPR model and CoPE/robustness benchmark ideas as methodology.
- A model-combination direction still exists, but it should be driven by measured failure modes rather than assumed fixes.
