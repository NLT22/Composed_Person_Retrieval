# ITCPR-C Synthesis

## Position

This direction combines the existing projects without inventing a fake error-and-fix story:

- FAFA provides a real composed person retrieval model with pretrained weights.
- The robustness benchmark project provides the evaluation protocol and corruption taxonomy for composed retrieval.
- CoPE contributes the composed image retrieval perspective: the query is intrinsically multimodal, so robustness should be measured on both reference image and modification text.
- SynCPR remains useful as synthetic CPR data and as a learned reranking/ambiguity probe, but current results do not support presenting the existing reranker as a robust correction module.

The cleanest research direction is therefore ITCPR-C: a robustness benchmark and analysis suite for composed person retrieval.

## Experiments Run

All runs used:

- FAFA checkpoint: `tuned_recall_at1_step.pt`
- ITCPR query count: `2202`
- Gallery count: `20510`
- Clean FAFA baseline: R1=`46.549`, mAP=`55.650`
- Gallery features kept clean; only query-side input was corrupted.

Source reports:

- Text: `experiments/itcpr_c_text_robustness/REPORT.md`
- Image: `experiments/itcpr_c_image_robustness/REPORT.md`
- SynCPR reranker: `experiments/fafa_hard_negative_reranker/REPORT.md`

## Text Robustness

| corruption | severity | R1 | mAP | R1 drop | RR@1 |
|---|---:|---:|---:|---:|---:|
| character_filter | 1 | 40.009 | 49.823 | -6.540 | 0.860 |
| character_filter | 3 | 28.429 | 38.077 | -18.120 | 0.611 |
| RemoveChar_filter | 1 | 40.191 | 50.261 | -6.358 | 0.863 |
| RemoveChar_filter | 3 | 29.791 | 39.492 | -16.757 | 0.640 |
| qwerty_filter | 1 | 39.555 | 49.392 | -6.994 | 0.850 |
| qwerty_filter | 3 | 28.520 | 38.107 | -18.029 | 0.613 |
| remove_space_filter | 1 | 42.779 | 52.208 | -3.769 | 0.919 |
| remove_space_filter | 3 | 32.516 | 42.195 | -14.033 | 0.699 |
| repetition_filter | 1 | 45.731 | 55.155 | -0.817 | 0.982 |
| repetition_filter | 3 | 44.005 | 53.750 | -2.543 | 0.945 |

Text takeaway: FAFA is much more brittle to character-level perturbations than to redundant wording. This is a real measured failure mode, not a made-up typo-fixing problem.

## Image Robustness

| corruption | severity | R1 | mAP | R1 drop | RR@1 |
|---|---:|---:|---:|---:|---:|
| gaussian_noise_filter | 3 | 34.378 | 44.220 | -12.171 | 0.739 |
| motion_blur_filter | 3 | 43.052 | 52.042 | -3.497 | 0.925 |
| brightness_filter | 3 | 44.187 | 53.491 | -2.361 | 0.949 |
| jpeg_compression | 3 | 42.779 | 52.918 | -3.769 | 0.919 |

Image takeaway: Gaussian noise is clearly more damaging than motion blur, brightness, or JPEG compression at severity 3. The image channel matters, but the strongest text corruptions are even more damaging than the sampled image corruptions.

## SynCPR Reranker Role

The gated SynCPR reranker previously gave a small clean ITCPR improvement:

- clean FAFA: R1=`46.549`, mAP=`55.650`
- best gated rerank: R1=`46.866`, mAP=`55.811`

Under corruption, its movement is tiny:

| domain | best observed movement | worst observed movement |
|---|---:|---:|
| text | +0.091 R1, +0.050 mAP | -0.318 R1, -0.146 mAP |
| image | +0.136 R1, +0.050 mAP | -0.136 R1, -0.055 mAP |

Conclusion: current SynCPR reranking is not a robustness solution. It can still be used as an ambiguity diagnostic or as a training source for a future corruption-aware gate, but not as the main contribution.

## Paper Direction

Recommended title-level direction:

**ITCPR-C: Benchmarking Robustness of Composed Person Retrieval**

Core claim:

Existing composed retrieval robustness work focuses on composed image retrieval. CPR has a different target domain and stronger identity/attribute constraints, so robustness should be evaluated separately. ITCPR-C adapts the composed retrieval corruption protocol to CPR and reveals modality-specific brittleness in FAFA.

Minimum solid contribution:

- Define ITCPR-C from ITCPR by corrupting query reference image and query modification text.
- Evaluate FAFA clean versus corrupted.
- Report recall, mAP, and relative robustness.
- Analyze which corruption families hurt CPR most.
- Show that a SynCPR-trained reranker helps clean ambiguity slightly but is not enough for robustness.

Next experimental step:

- Add the remaining benchmark image corruptions at severity 3 first.
- Then add severity sweep 1-5 for the top failure families: text `character_filter`/`qwerty_filter`, image `gaussian_noise_filter`.
- Only after that consider a model-side method, because the benchmark now tells us where the model actually fails.
