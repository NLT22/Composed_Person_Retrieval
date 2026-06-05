# Phân tích uncertainty của CoPE và khả năng áp dụng cho FAFA

## 1. Câu hỏi cần trả lời

Mục tiêu là kiểm tra xem uncertainty trong CoPE có thể dùng để làm rõ hoặc cải thiện FAFA không.

Cụ thể:

- CoPE tính uncertainty như thế nào?
- Có tính được số liệu cụ thể trên FashionIQ không?
- Có tính được số liệu cụ thể trên SynCPR không?
- Dựa trên số liệu đó, có nên áp dụng uncertainty cho FAFA không?

## 2. CoPE tính uncertainty như thế nào?

CoPE không dùng entropy ranking làm uncertainty chính. CoPE học một embedding xác suất:

```text
feature = {
  mean: vector embedding,
  var: log(variance) theo từng chiều embedding
}
```

Trong code:

- `ACL25-CoPE/models/encoders.py`: `get_image_features()` và `get_text_features()` trả về `mean` và `var`.
- `ACL25-CoPE/models/utils.py`: `UncertaintyPooler` sinh ra `log(variance)`.
- `ACL25-CoPE/models/model.py`: `encode_query()` cộng uncertainty của text và reference image bằng `logsumexp`.
- `ACL25-CoPE/models/loss.py` và `ACL25-CoPE/engine.py`: score/distance dùng cả mean và standard deviation.

Distance của CoPE gồm 3 phần:

```text
distance =
  ||mu_query - mu_target||^2
  + ||sigma_query - sigma_target||^2
  + 2 * D * mean(sigma_query) * mean(sigma_target)
```

Trong đó:

```text
sigma = exp(0.5 * log_var)
```

Nếu distance nhỏ hơn thì target được coi là giống query hơn.

## 3. Điểm quan trọng trong code: uncertainty bị clamp rất hẹp

Trong `UncertaintyPooler.forward()`:

```python
x_clamped = soft_clamp_tanh(x, min_val=-10.0, max_val=-8.0)
```

Nghĩa là `log(var)` bị ép vào khoảng:

```text
[-10, -8]
```

Tương ứng:

```text
std = exp(0.5 * log_var)
```

Nếu `log_var = -8`:

```text
std ≈ 0.018316
```

Với query, CoPE cộng uncertainty text và image:

```python
log_combined_var = logsumexp([log_var_text, log_var_img])
```

Nếu cả text và image đều saturate ở `-8`, query sẽ thành:

```text
logsumexp(-8, -8) = -7.30685
std ≈ 0.025902
```

Đây chính xác là số mình đo được.

## 4. Số liệu FashionIQ đã tính được

Do venv của `ACL25-CoPE` hiện báo `cuda=False`, mình chạy subset CPU để kiểm chứng pipeline:

```text
category: dress
queries: 64
targets: 64
```

Đã chạy 2 checkpoint:

```text
ACL25-CoPE/checkpoints/fiq_dress/best_model
ACL25-CoPE/checkpoints/fiq_dress_paper_fix_distance/best_model
```

Output:

```text
experiments/uncertainty/cope_fashioniq_dress_subset/
experiments/uncertainty/cope_fashioniq_dress_fix_distance_subset/
```

### 4.1. Checkpoint `fiq_dress/best_model`

| Metric | Mean | Std | p05 | p95 |
|---|---:|---:|---:|---:|
| query_std_mean | 0.025902 | 0.0000000038 | 0.025902 | 0.025902 |
| query_var_trace | 0.515270 | 0.000000083 | 0.515270 | 0.515270 |
| top1_margin | 0.074993 | 0.062555 | 0.002633 | 0.178071 |
| top10_entropy | 2.299553 | 0.002098 | 2.295405 | 2.301951 |

Recall trên subset:

```text
R10 = 90.625
R50 = 100.000
```

Correlation:

| Correlation | Value |
|---|---:|
| query_std_mean vs rank | 0.024 |
| query_std_mean vs hit10 | -0.032 |
| query_var_trace vs rank | 0.036 |
| top10_entropy vs rank | 0.415 |
| top1_margin vs hit10 | 0.220 |

### 4.2. Checkpoint `fiq_dress_paper_fix_distance/best_model`

| Metric | Mean | Std | p05 | p95 |
|---|---:|---:|---:|---:|
| query_std_mean | 0.025902 | 0.0000000020 | 0.025902 | 0.025902 |
| query_var_trace | 0.515270 | 0.0000000000 | 0.515270 | 0.515270 |
| top1_margin | 0.035132 | 0.035252 | 0.001411 | 0.101585 |
| top10_entropy | 2.301872 | 0.000649 | 2.300546 | 2.302484 |

Recall trên subset:

```text
R10 = 93.750
R50 = 100.000
```

Correlation:

| Correlation | Value |
|---|---:|
| query_std_mean vs rank | -0.044 |
| query_std_mean vs hit10 | 0.044 |
| query_var_trace vs rank | None |
| top10_entropy vs rank | 0.206 |
| top1_margin vs hit10 | 0.168 |

## 5. Kết luận từ FashionIQ

CoPE uncertainty head trong các checkpoint đã thử gần như **không phân biệt query khó/dễ**.

Lý do:

- `query_std_mean` gần như hằng số.
- `query_var_trace` gần như hằng số.
- Correlation giữa uncertainty learned và rank/hit rất gần 0.
- Trong khi đó `top10_entropy` và `top1_margin` vẫn có tín hiệu tốt hơn.

Điều này nghĩa là:

```text
Không nên bê trực tiếp CoPE learned variance sang FAFA.
```

Nếu dùng uncertainty cho FAFA, nên ưu tiên các uncertainty proxy có tín hiệu thực nghiệm:

- margin giữa top1 và top2,
- entropy của top-k scores,
- độ phân tán token-level score,
- caption/image corruption indicators.

## 6. Có tính được uncertainty trên SynCPR không?

Có, nhưng cần phân biệt rõ:

### 6.1. Không tính được CoPE-native uncertainty trên SynCPR một cách đúng nghĩa

Lý do:

- CoPE được train/eval cho composed image retrieval như FashionIQ.
- SynCPR là composed person retrieval synthetic data.
- FAFA feature cache không có `log_var`.
- CoPE checkpoint không được train trên SynCPR/CPR.

Vì vậy nếu ép CoPE chạy SynCPR thì số uncertainty không còn đúng domain.

### 6.2. Có thể tính FAFA/SynCPR uncertainty proxy

Mình đã tính proxy trên SynCPR feature cache:

```text
experiments/fafa_pretrained/syncpr_features
```

Sample đã chạy:

```text
8 shards x 2048 rows = 16384 pairs
```

Output:

```text
experiments/uncertainty/fafa_syncpr_proxy/
```

Các proxy đã tính:

- `margin`: score positive - score hard negative.
- `top10_entropy`: entropy của top hard negatives.
- `positive_token_std`: độ lệch chuẩn token-level score của positive pair.
- `hit1_within_shard`: positive target có đứng top1 trong shard không.

Kết quả:

| Metric | Mean | Std | p05 | p95 |
|---|---:|---:|---:|---:|
| margin | -0.001854 | 0.066130 | -0.114176 | 0.100308 |
| top10_entropy | 2.297810 | 0.002350 | 2.293533 | 2.301095 |
| positive_token_std | 0.015703 | 0.007368 | 0.005878 | 0.029544 |
| hit1_within_shard | 50.177% | - | - | - |

Correlation:

| Correlation | Value |
|---|---:|
| margin vs hit1 | 0.747 |
| top10_entropy vs hit1 | 0.324 |
| positive_token_std vs hit1 | 0.230 |
| positive_token_std vs margin | 0.244 |

## 7. Kết luận từ SynCPR/FAFA proxy

Với FAFA, tín hiệu tốt nhất hiện tại là **margin**, không phải learned variance kiểu CoPE.

`margin_vs_hit1 = 0.747` là rất mạnh:

- margin cao: positive target thường đứng top1.
- margin thấp/âm: hard negative dễ vượt positive.

`positive_token_std` cũng có tín hiệu, nhưng yếu hơn:

```text
positive_token_std_vs_hit1 = 0.230
```

Điều này hợp với các thử nghiệm trước đó:

- Gated rerank dựa trên margin/caption length giúp clean ITCPR tăng nhẹ.
- Rerank không cứu được corruption vì query embedding đã lệch.

## 8. Có nên áp dụng uncertainty cho FAFA không?

Câu trả lời ngắn:

> Có thể áp dụng uncertainty cho FAFA, nhưng **không nên áp dụng bằng cách copy learned variance của CoPE**.

Nên áp dụng theo hướng:

```text
uncertainty-aware gating / ambiguity-aware reranking
```

Không nên áp dụng theo hướng:

```text
thêm CoPE variance head rồi kỳ vọng tự cải thiện
```

## 9. Vì sao không nên copy CoPE uncertainty?

Vì số liệu FashionIQ cho thấy learned variance đang gần như hằng số:

```text
query_std_mean ≈ 0.025902 cho mọi query
target_std_mean ≈ 0.018316 cho mọi target
```

Nếu variance gần như hằng, phần uncertainty trong distance gần như chỉ là một hằng số hoặc rất ít phân biệt target/query.

Khi đó nó không giúp biết:

- query nào khó,
- query nào dễ,
- khi nào nên rerank,
- khi nào model đang không chắc chắn.

Trong khi đó FAFA proxy cho thấy margin có tín hiệu rõ ràng hơn nhiều.

## 10. Hướng nên làm tiếp

### 10.1. Dùng uncertainty proxy cho FAFA

Các proxy nên dùng:

- top1-top2 margin,
- top-k entropy,
- token-level score std,
- caption length/informativeness,
- corruption score nếu đang ở ITCPR-C.

### 10.2. Làm gate thay vì làm module lớn

Ví dụ logic:

```text
Nếu margin thấp và caption đủ thông tin:
    bật SynCPR reranker
Nếu margin thấp và entropy cao:
    đánh dấu query uncertain
Nếu query bị text/image corruption:
    không tin reranker clean, cần robust query encoder hoặc test-time normalization
```

### 10.3. Nếu muốn learned uncertainty thật sự

Khi đó cần train một head riêng cho FAFA, nhưng phải có loss rõ ràng:

- uncertainty cao khi hard negative gần positive,
- uncertainty cao khi corruption làm query score bất ổn,
- uncertainty calibration bằng clean/corrupted ITCPR-C.

Không nên train chỉ vì CoPE có uncertainty head.

## 11. Kết luận cuối cùng

Kết luận thực nghiệm hiện tại:

```text
CoPE learned variance: tính được trên FashionIQ, nhưng gần như hằng số trên subset đã thử.
FAFA/SynCPR proxy uncertainty: tính được và margin có tương quan mạnh với hit1.
```

Vì vậy hướng hợp lý cho FAFA là:

```text
Áp dụng uncertainty như một cơ chế gating/diagnostic dựa trên margin, entropy và token-level statistics.
Không copy trực tiếp probabilistic variance head của CoPE.
```

Nói ngắn gọn:

> Uncertainty đáng dùng cho FAFA, nhưng nên là uncertainty được đo từ hành vi ranking/token score của FAFA, không phải variance head của CoPE.
