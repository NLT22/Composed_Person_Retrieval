# Tổng hợp quá trình triển khai FAFA x SynCPR x CoPE/Robustness

## 1. Mục tiêu ban đầu

Trong folder `2025.2` hiện có ba nhóm nghiên cứu/code liên quan:

- **FAFA**: giải quyết bài toán **Composed Person Retrieval**.
- **CoPE**: giải quyết bài toán **Composed Image Retrieval**.
- **Benchmark-Robustness-Text-Image-Compose-Retrieval**: đánh giá độ bền vững của các model composed retrieval khi query image/text bị nhiễu.

Vấn đề không phải là “ghép hai project lại cho có”. Nếu ghép FAFA và CoPE trực tiếp mà không có lý do khoa học rõ ràng, hướng đó khá yếu:

- FAFA và CoPE không cùng bài toán.
- Dữ liệu khác format.
- Checkpoint CoPE không có sẵn.
- Máy local không phù hợp để train lại các model lớn.

Vì vậy mình chuyển hướng sang một câu hỏi nghiên cứu hợp lý hơn:

> Nếu FAFA là model composed person retrieval có weight sẵn, thì nó có robust không khi query composed retrieval bị nhiễu ảnh/text theo protocol của CoPE/Robustness?

Hướng này kết hợp các nghiên cứu đang có một cách tự nhiên:

- FAFA cung cấp model và bài toán CPR.
- Benchmark-Robustness cung cấp protocol corruption.
- CoPE cung cấp góc nhìn composed retrieval: query gồm ảnh tham chiếu + text mô tả thay đổi, nên cần đánh giá cả hai kênh.
- SynCPR cung cấp synthetic CPR data và tín hiệu reranking/ambiguity.

Kết quả cuối cùng mình đề xuất hướng:

> **ITCPR-C: Benchmarking Robustness of Composed Person Retrieval**

## 2. Các skill đã dùng và cách áp dụng

Bạn yêu cầu dùng các skill trong `~/.claude/skill`. Thư mục đó không tồn tại đúng path, nhưng mình tìm được các skill liên quan trong `~/.claude/plugins/cache` và dùng theo tinh thần của chúng:

- **deep-research**: dùng để đặt câu hỏi nghiên cứu đúng. Thay vì “làm sao sửa typo”, mình đặt câu hỏi “FAFA robust đến đâu khi query composed retrieval bị corruption?”.
- **academic-pipeline**: dùng để giữ experiment tái lập được: có checkpoint, seed, metric, JSON output và markdown report.
- **karpathy-guidelines**: dùng để làm bản nhỏ, chạy được, đo được trước khi nghĩ đến train hay viết module mới.

Lý do cách làm này hợp lý: nó tránh việc “vẽ ra lỗi sai rồi sửa”. Mình không tạo một module fix tùy tiện, mà đầu tiên đo xem model thật sự hỏng ở đâu.

## 3. Những việc đã làm

### 3.1. Kiểm tra weight và cache của FAFA

Đã xác nhận FAFA có checkpoint:

```text
FAFA_SynCPR/tuned_recall_at1_step.pt
```

Đã có cache feature ITCPR:

```text
experiments/fafa_pretrained/itcpr_fafa_features.pt
```

Cache này gồm:

- `2202` query.
- `20510` gallery image.

Clean baseline của FAFA trên ITCPR:

| Metric | Value |
|---|---:|
| R1 | 46.549 |
| R5 | 66.167 |
| R10 | 73.252 |
| mAP | 55.650 |

Đây là mốc so sánh chính. Mọi corruption về sau đều được so với mốc này.

### 3.2. Extract feature SynCPR và thử reranker

Trước đó đã extract feature SynCPR thành các shard:

```text
experiments/fafa_pretrained/syncpr_features
```

Tổng số sample SynCPR đã extract:

```text
1,153,230 samples
```

Sau đó mình thử hướng hard-negative reranker học từ SynCPR.

Kết quả tốt nhất trên clean ITCPR:

| Method | R1 | mAP |
|---|---:|---:|
| FAFA clean | 46.549 | 55.650 |
| FAFA + gated SynCPR rerank | 46.866 | 55.811 |

Cải thiện:

- R1 tăng `+0.317`.
- mAP tăng `+0.161`.

Đây là cải thiện nhỏ nhưng có ý nghĩa: SynCPR reranker có thể giúp một số query mơ hồ, nhưng chưa đủ mạnh để trở thành đóng góp chính.

Ví dụ dễ hiểu:

- FAFA ban đầu xếp target đúng ở hạng 2 hoặc 3 cho một query khó.
- Reranker chỉ can thiệp trong top-k và chỉ can thiệp khi model thiếu chắc chắn.
- Nếu can thiệp đúng query, R1 có thể tăng.
- Nếu can thiệp quá rộng, nó có thể làm xấu kết quả.

Vì vậy mình dùng gate:

```text
topk=1000
delta_mult=-0.015
margin_quantile=0.30
min_caption_words=6
```

Gate này nghĩa là: chỉ dùng reranker khi query có margin thấp và caption có đủ thông tin.

### 3.3. Tạo script robustness cho ITCPR-C

Mình đã tạo/mở rộng script:

```text
fafa_itcpr_robustness.py
```

Script này có hai mode:

```bash
--benchmark-mode text
--benchmark-mode image
```

Nó làm việc như sau:

1. Load FAFA checkpoint.
2. Load clean gallery feature từ cache.
3. Chỉ corrupt query-side input:
   - với text mode: corrupt caption của query.
   - với image mode: corrupt reference image của query.
4. Recompute query embedding.
5. So sánh với clean gallery.
6. Tính metric R1/R5/R10/mAP.
7. Optionally chạy gated SynCPR reranker để xem có giúp không.
8. Lưu `results.json` và `REPORT.md`.

Lý do giữ gallery clean:

- Rẻ hơn về compute, chạy được trên máy local.
- Phù hợp với câu hỏi: nếu query của user bị nhiễu thì model còn tìm đúng target không?
- Không phải recompute 20k gallery image mỗi lần.

## 4. Text robustness: FAFA yếu ở đâu?

Đã chạy trên `2202` query ITCPR với các text corruption từ Benchmark project:

- `character_filter`
- `RemoveChar_filter`
- `qwerty_filter`
- `remove_space_filter`
- `repetition_filter`

Kết quả tóm tắt:

| Corruption | Severity | R1 | mAP | R1 drop | RR@1 |
|---|---:|---:|---:|---:|---:|
| clean | 0 | 46.549 | 55.650 | 0.000 | 1.000 |
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

### Kết luận dễ hiểu

FAFA rất nhạy với lỗi cấp ký tự:

- đổi ký tự,
- xóa ký tự,
- gõ nhầm phím QWERTY.

Nhưng FAFA khá bền với việc lặp từ.

Ví dụ caption gốc:

```text
Wearing a brown plaid shirt, black leather shoes, another dark gray T-shirt, another blue jeans
```

Nếu bị lặp từ:

```text
Wearing a brown plaid plaid shirt, black leather shoes, another dark gray T-shirt, another blue jeans
```

Ý nghĩa vẫn còn khá rõ, model còn chịu được.

Nhưng nếu bị typo cấp ký tự:

```text
Wearign a borwn palid shrit, balck laether shoes...
```

Text encoder/query composer bị ảnh hưởng mạnh hơn, nên retrieval tụt sâu.

Đây là measured failure mode thật, không phải lỗi mình tự vẽ ra để sửa.

## 5. Image robustness: nhiễu ảnh nào ảnh hưởng nhiều?

Đã chạy image corruption theo protocol Benchmark project, severity 3, trên reference image của query:

- `gaussian_noise_filter`
- `motion_blur_filter`
- `brightness_filter`
- `jpeg_compression`

Kết quả:

| Corruption | Severity | R1 | mAP | R1 drop | RR@1 |
|---|---:|---:|---:|---:|---:|
| clean | 0 | 46.549 | 55.650 | 0.000 | 1.000 |
| gaussian_noise_filter | 3 | 34.378 | 44.220 | -12.171 | 0.739 |
| motion_blur_filter | 3 | 43.052 | 52.042 | -3.497 | 0.925 |
| brightness_filter | 3 | 44.187 | 53.491 | -2.361 | 0.949 |
| jpeg_compression | 3 | 42.779 | 52.918 | -3.769 | 0.919 |

### Kết luận dễ hiểu

Không phải mọi nhiễu ảnh đều làm model hỏng như nhau.

Gaussian noise làm FAFA tụt nhiều:

```text
R1: 46.549 -> 34.378
drop: -12.171
```

Nhưng brightness severity 3 làm tụt nhẹ hơn:

```text
R1: 46.549 -> 44.187
drop: -2.361
```

Điều này quan trọng vì nó cho thấy benchmark có ý nghĩa: nó phân loại được model yếu với loại nhiễu nào.

## 6. SynCPR reranker có cải thiện robustness không?

Kết quả ngắn gọn: **không đáng kể**.

Trên clean ITCPR, gated SynCPR rerank có cải thiện nhỏ:

```text
R1: 46.549 -> 46.866
mAP: 55.650 -> 55.811
```

Nhưng dưới corruption, mức thay đổi rất nhỏ:

| Domain | Best movement | Worst movement |
|---|---:|---:|
| text | +0.091 R1, +0.050 mAP | -0.318 R1, -0.146 mAP |
| image | +0.136 R1, +0.050 mAP | -0.136 R1, -0.055 mAP |

### Vì sao không cải thiện mạnh?

Vì reranker hiện tại học từ hard negatives của SynCPR clean/synthetic. Nó học cách điều chỉnh ranking khi FAFA đã có embedding query ổn định.

Nhưng khi text/image query bị corruption, vấn đề nằm ở **query embedding đã bị lệch từ đầu vào**.

Nói đơn giản:

- Reranker chỉ sửa thứ tự trong danh sách ứng viên.
- Nếu query embedding đã bị noise làm target đúng bị đẩy xuống quá xa, reranker khó cứu.
- Nếu corruption làm text/image representation sai nghĩa, ranker phía sau không có đủ thông tin để phục hồi.

Ví dụ:

Nếu caption typo làm model hiểu sai “brown plaid shirt” thành một embedding không rõ, target đúng có thể không còn nằm trong top ứng viên tốt. Reranker top-k lúc này chỉ đang sắp xếp lại những ứng viên đã sai sẵn.

Vậy SynCPR reranker nên được trình bày như:

```text
ambiguity diagnostic / clean-query reranking probe
```

Không nên trình bày như:

```text
robustness correction module
```

## 7. Có cải thiện gì đã đạt được?

Có hai loại cải thiện.

### 7.1. Cải thiện về model/ranking

Trên clean ITCPR, gated SynCPR rerank tăng nhẹ:

```text
R1 +0.317
mAP +0.161
```

Đây là cải thiện nhỏ, nhưng nó chứng minh SynCPR có thể mang tín hiệu hữu ích nếu gate đúng.

### 7.2. Cải thiện về hướng nghiên cứu

Đây mới là cải thiện quan trọng hơn.

Trước khi chạy các experiment này, hướng FAFA + CoPE bị mờ:

- CoPE không có checkpoint sẵn.
- CoPE giải quyết composed image retrieval, không phải composed person retrieval.
- FAFA có weight nhưng không rõ nên ghép ở đâu.
- Train lại nặng và rủi ro.

Sau khi chạy robustness benchmark, hướng nghiên cứu rõ hơn:

```text
ITCPR-C: Benchmarking Robustness of Composed Person Retrieval
```

Hướng này có:

- model thật: FAFA,
- dataset thật: ITCPR,
- protocol có sẵn: Benchmark-Robustness,
- composed retrieval context: CoPE/Benchmark,
- synthetic CPR angle: SynCPR,
- số liệu thật: text/image robustness reports.

Đây là một đóng góp để viết paper/báo cáo hợp lý hơn nhiều so với việc cố ghép architecture.

## 8. Vì sao cách làm này đúng hơn việc train/gép model ngay?

### Lý do 1: Không có CoPE checkpoint phù hợp CPR

CoPE checkpoint nếu có cũng gắn với FashionIQ/CIR. Nó không trực tiếp dùng cho person retrieval.

Nếu train lại, máy local có thể không đủ.

### Lý do 2: FAFA đã có checkpoint và baseline

FAFA là điểm neo tốt nhất vì:

- đúng bài toán CPR,
- có checkpoint,
- đã chạy được ITCPR,
- có clean baseline rõ.

### Lý do 3: Benchmark trước, method sau

Nếu chưa biết model hỏng ở đâu mà đã viết module fix, rất dễ thành một câu chuyện yếu.

Bây giờ ta đã biết:

- text character-level corruption gây drop mạnh nhất,
- gaussian image noise gây drop mạnh hơn blur/brightness/JPEG,
- reranker SynCPR không đủ để sửa corruption.

Từ đó nếu sau này làm method, method sẽ có lý do:

- corruption-aware query composition,
- robust text encoder/query encoder,
- denoising/adaptation cho reference image,
- gate phát hiện query bị corruption.

## 9. Các file đã tạo / đã cập nhật

Script chính:

```text
fafa_itcpr_robustness.py
```

Text robustness:

```text
experiments/itcpr_c_text_robustness/results.json
experiments/itcpr_c_text_robustness/REPORT.md
```

Image robustness:

```text
experiments/itcpr_c_image_robustness/results.json
experiments/itcpr_c_image_robustness/REPORT.md
```

Tổng hợp research:

```text
experiments/itcpr_c_synthesis/REPORT.md
experiments/itcpr_c_synthesis/ENGINEERING_SUMMARY_VI.md
```

Reranker SynCPR:

```text
experiments/fafa_hard_negative_reranker/REPORT.md
```

## 10. Nên làm gì tiếp theo?

Thứ tự nên làm tiếp:

### 10.1. Mở rộng image corruption

Chạy thêm các corruption còn lại của Benchmark project ở severity 3:

```text
shot_noise_filter
impulse_noise_filter
defocus_blur_filter
zoom_blur_filter
contrast_filter
pixelate_filter
frost/snow/fog nếu dependency ổn
```

### 10.2. Severity sweep cho failure family mạnh nhất

Không cần sweep tất cả ngay. Nên tập trung:

```text
text: character_filter, qwerty_filter
image: gaussian_noise_filter
```

Chạy severity 1-5 để có curve.

### 10.3. Viết benchmark protocol ITCPR-C

Định nghĩa rõ:

- clean ITCPR,
- ITCPR-C-text,
- ITCPR-C-image,
- metric: R@1/R@5/R@10/mAP/RR,
- model: FAFA pretrained.

### 10.4. Sau đó mới nghĩ đến method

Nếu cần method, nên đi theo hướng:

- corruption-aware query encoder,
- text robustness augmentation,
- lightweight test-time normalization,
- corruption detector/gate.

Không nên bắt đầu bằng train CoPE lại vì compute và checkpoint không ủng hộ.

## 11. Kết luận ngắn gọn

Hướng tốt nhất hiện tại không phải “FAFA + CoPE fusion” theo nghĩa ghép architecture.

Hướng tốt nhất là:

```text
Dùng protocol robustness của CoPE/Benchmark để tạo ITCPR-C,
đánh giá FAFA trên composed person retrieval khi query image/text bị corruption,
sau đó dùng SynCPR như diagnostic/reranking probe.
```

Đây là cách kết hợp các nghiên cứu đang có một cách có logic:

- CoPE/Benchmark cho framework composed retrieval robustness.
- FAFA cho model CPR có weight thật.
- ITCPR cho test set CPR.
- SynCPR cho synthetic signal và ambiguity analysis.

Quan trọng nhất: các kết luận hiện tại đều dựa trên số liệu đã chạy, không phải một lỗi giả định rồi tự sửa.
