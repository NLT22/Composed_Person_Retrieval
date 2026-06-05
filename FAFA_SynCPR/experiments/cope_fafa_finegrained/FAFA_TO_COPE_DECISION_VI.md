# Áp Dụng Ngược FAFA Vào CoPE: Kết Quả Và Quyết Định Kỹ Thuật

## 1. Câu hỏi nghiên cứu

CoPE giải bài toán composed image retrieval trên FashionIQ bằng embedding xác suất: mỗi query/target được biểu diễn bằng mean và variance toàn cục. Limitation hợp lý của hướng này là thiếu fine-grained alignment: mô hình có thể biết "ảnh này nói chung giống query", nhưng không có cơ chế rõ ràng để kiểm tra từng chi tiết như tay áo, cổ áo, màu, hoa văn, độ dài váy.

FAFA giải composed person retrieval bằng một ý tưởng ngược lại: ngoài embedding toàn cục, FAFA dùng FDA (Fine-grained Dynamic Alignment), tức là so query với các token/patch chi tiết của ảnh target, chọn top-k vùng khớp nhất rồi cộng vào score.

Giả thuyết thí nghiệm:

> Nếu CoPE thiếu fine-grained, ta có thể giữ nguyên checkpoint CoPE và thêm FDA kiểu FAFA như một reranker để cải thiện ranking trên FashionIQ.

## 2. Tôi đã triển khai gì

File chính:

- `Composed_Person_Retrieval/FAFA_SynCPR/cope_fafa_finegrained.py`

Script này không sửa code gốc của CoPE/FAFA. Nó load checkpoint CoPE đã có, encode FashionIQ, rồi tính thêm fine-grained score.

Các biến thể đã thử:

1. `mean -> target tokens`
   - Query vẫn là composed embedding toàn cục của CoPE.
   - Target là patch tokens từ vision transformer.
   - Score FDA = query so với từng patch target, lấy top-k patch giống nhất.

2. `query tokens -> target tokens`
   - Query lấy patch tokens của reference image sau khi đã được text-modulate bởi CoPE.
   - Mỗi query patch so với target patches, lấy top-k target patches, rồi lấy tiếp top-k query patches.
   - Biến thể này gần tinh thần FAFA hơn vì dùng nhiều token chi tiết ở cả hai phía.

3. Rerank toàn bộ gallery và rerank trong top-K
   - Rerank toàn bộ dễ gây nhiễu vì patch similarity có thể kéo nhầm ảnh có texture giống nhau.
   - Rerank top-10/top-50 giữ CoPE làm bộ lọc chính, FDA chỉ sắp lại ứng viên gần nhất.

4. Chọn alpha trên nửa đầu query, báo cáo trên nửa sau
   - Để tránh tự lừa mình khi sweep alpha.
   - Score kết hợp: `CoPE_z + alpha * FDA_z`.

## 3. Kết quả chính

### Subset 256 query / 256 target

| Biến thể | Baseline eval R@10 | FDA eval R@10 | Baseline eval R@50 | FDA eval R@50 | Mean rank thay đổi |
|---|---:|---:|---:|---:|---:|
| mean, rerank all | 75.781 | 75.781 | 92.969 | 92.969 | 11.852 -> 11.969 |
| mean, rerank top-10 | 75.781 | 75.781 | 92.969 | 92.969 | 11.852 -> 11.867 |
| mean, rerank top-50 | 75.781 | 75.781 | 92.969 | 92.969 | 11.852 -> 11.883 |
| token-token, rerank top-10 | 75.781 | 75.781 | 92.969 | 92.969 | 11.852 -> 11.805 |

Subset nhỏ cho thấy FDA có thể đổi rank một chút, nhưng chưa đổi recall.

Ví dụ có cải thiện:

- `B00G6TRZE8` + "A low cut black gown..." -> target `B003ZJ6GCO`: baseline rank 10 -> FDA rank 5.
- `B00CAMMV5S` + "Is more purple and elegant..." -> target `B00CAMJMRS`: baseline rank 4 -> FDA rank 2.

Ví dụ bị kéo sai:

- `B002OEYH3Q` + "Has long sleeves and patterned..." -> target `B008YIGWOS`: baseline rank 2 -> FDA rank 7.
- `B003M4Z9OS` + "Patterned dress with zig zags..." -> target `B009XHND2I`: baseline rank 2 -> FDA rank 6.

Diễn giải đơn giản: patch-level score bắt được chi tiết màu/texture/cổ áo ở vài case, nhưng cũng rất dễ bị lừa bởi hoa văn hoặc vùng ảnh giống nhau mà không đúng composed intent.

### 256 query / full target gallery

Full gallery có 3,817 target ảnh, sát bài toán retrieval thật hơn.

| Biến thể | Baseline eval R@10 | FDA eval R@10 | Baseline eval R@50 | FDA eval R@50 | Mean rank thay đổi |
|---|---:|---:|---:|---:|---:|
| mean, rerank top-10 | 33.594 | 33.594 | 61.719 | 61.719 | 160.703 -> 160.680 |
| token-token, rerank top-10 | 33.594 | 33.594 | 61.719 | 61.719 | 160.703 -> 160.719 |

Kết luận từ full gallery: FDA post-hoc gần như không cải thiện recall. Biến thể mean có mean rank tốt hơn cực nhỏ, nhưng mức này quá bé để gọi là cải thiện thật.

## 4. Vì sao chưa nên train lớn ngay

Không nên train lớn ngay vì post-hoc signal quá yếu.

Nếu FDA score có chất lượng tốt, ta thường kỳ vọng ít nhất một trong các dấu hiệu sau xuất hiện trên subset:

- R@10 tăng dù nhỏ.
- R@50 tăng.
- Mean rank giảm rõ ràng và ổn định giữa các biến thể.
- Alpha lớn hơn 0 được chọn và không làm eval tệ đi.

Thực tế:

- R@10/R@50 không tăng ở cả subset lẫn full gallery.
- Alpha tốt nhất rất nhỏ (`0.05` hoặc `0.1`), nghĩa là FDA chỉ chịu được trọng số nhẹ.
- Khi tăng alpha, kết quả thường xấu đi.
- FDA cải thiện vài case nhưng cũng làm hỏng vài case tương tự.

Nói ngắn gọn: fine-grained là đúng vấn đề, nhưng cách thêm FDA ngoài mô hình chưa đủ. Nếu train một head lớn ngay trên score này, nguy cơ cao là overfit hoặc chỉ học lại CoPE score.

## 5. Điều này nói gì về hướng kết hợp FAFA + CoPE

Kết quả không phủ định ý tưởng "CoPE thiếu fine-grained". Nó chỉ nói rằng:

> Không thể giải limitation fine-grained của CoPE chỉ bằng cách lấy patch tokens có sẵn rồi cộng FDA score post-hoc.

Lý do:

1. Patch tokens của CoPE/CLIP chưa được train để làm composed local alignment.
   - Chúng tốt cho representation tổng quát, nhưng chưa chắc một patch "tay áo dài" sẽ align đúng với phrase "long sleeves".

2. Query token hiện tại vẫn chưa thật sự là phrase-level query.
   - Text instruction trong FashionIQ có nhiều thuộc tính: màu, form, độ dài, họa tiết.
   - FDA kiểu FAFA cần biết token nào đại diện cho thuộc tính nào. Script hiện tại mới chỉ dùng text-modulated image tokens, chưa tách phrase/attribute.

3. FashionIQ khác composed person retrieval.
   - Person retrieval có cấu trúc người/quần áo/vùng thân thể rõ hơn.
   - FashionIQ có sản phẩm đa dạng, nhiều ảnh bị texture/màu đánh lừa.

## 6. Hướng khả thi tiếp theo

### Hướng A: Attribute-aware fine-grained CoPE

Đây là hướng đáng làm nhất nếu muốn thành đóng góp nghiên cứu.

Ý tưởng:

- Tách text instruction thành các attribute phrase: màu, độ dài, tay áo, cổ áo, họa tiết, style.
- Dùng text phrase embedding làm query local tokens.
- Align từng phrase với target patch tokens.
- Train nhẹ một contrastive/ranking loss để phrase đúng kéo target đúng lên.

Ví dụ:

Instruction: "is black, longer, has long sleeves"

Thay vì một vector query toàn cục, tạo 3 local queries:

- `black`
- `longer`
- `long sleeves`

Rồi mỗi local query tìm vùng ảnh liên quan. Đây mới là fine-grained đúng nghĩa cho FashionIQ.

### Hướng B: Gated reranker, không train backbone

Nếu máy không đủ train lớn, dùng hướng nhẹ:

- Freeze CoPE.
- Cache query/target embeddings và patch tokens.
- Train một gate nhỏ hoặc logistic calibrator trên feature:
  - CoPE margin top-1/top-2.
  - FDA margin.
  - Entropy top-10.
  - Độ đồng thuận giữa mean-FDA và token-FDA.
- Chỉ bật FDA khi CoPE đang uncertain.

Hướng này rẻ hơn nhiều so với train lại CoPE, và hợp với kết quả trước đó về uncertainty: learned variance của CoPE không hữu ích, nhưng margin/entropy có tín hiệu hơn.

### Hướng C: Dùng FDA làm công cụ phân tích lỗi, không làm scorer chính

FDA hiện tại chưa nâng recall, nhưng nó hữu ích để phát hiện case fine-grained:

- Query mà FDA cải thiện mạnh: có thể là case CoPE thiếu chi tiết.
- Query mà FDA làm tệ: thường là case texture/màu gây nhiễu.

Có thể dùng các case này để viết phần analysis trong báo cáo nghiên cứu.

## 7. Quyết định hiện tại

Không nên train lớn ngay với FDA post-hoc hiện tại.

Nên đi theo hướng:

1. Giữ script `cope_fafa_finegrained.py` làm baseline fine-grained post-hoc.
2. Viết tiếp module cache feature để chạy nhiều sweep rẻ hơn.
3. Nếu triển khai tiếp, ưu tiên `Attribute-aware fine-grained CoPE`, không phải cộng FDA thô.
4. Nếu cần kết quả nhanh ít GPU, train một gated reranker nhỏ trên cached features.

## 8. Các file kết quả

- `dress_q256_full_mean_rerank10/REPORT.md`
- `dress_q256_full_mean_rerank10/results.json`
- `dress_q256_full_token_rerank10/REPORT.md`
- `dress_q256_full_token_rerank10/results.json`
- `dress_q256_t256_token_rerank10/REPORT.md`
- `dress_q256_t256_token_rerank10/results.json`

Kết luận thực dụng: hướng kết hợp FAFA + CoPE vẫn có logic, nhưng phải đưa fine-grained vào quá trình học hoặc học gate có điều kiện. Bản rerank FDA post-hoc hiện tại chưa đủ mạnh để gọi là cải thiện.
