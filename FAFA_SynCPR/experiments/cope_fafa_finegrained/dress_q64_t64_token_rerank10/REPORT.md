# CoPE + FAFA Fine-Grained Dynamic Alignment

## Mục tiêu

Thử áp dụng ngược ý tưởng fine-grained của FAFA vào CoPE/FashionIQ. CoPE gốc dùng embedding mean/variance toàn cục; thí nghiệm này giữ nguyên checkpoint CoPE rồi thêm điểm FDA kiểu FAFA trên patch tokens của target.

## Cách làm

- Baseline: score xác suất gốc của CoPE.
- FDA: lấy vector query composed của CoPE, so với từng patch token của target image, chọn top-k patch giống nhất rồi lấy trung bình.
- Kết hợp: chuẩn hoá score theo từng query, sau đó dùng `score = CoPE_z + alpha * FDA_z`.
- Nếu bật `rerank_topk`, FDA chỉ được áp dụng trong top-K ứng viên CoPE, phần còn lại giữ nguyên score CoPE.
- Chọn `alpha` trên nửa đầu tập query, báo cáo trên nửa sau để giảm overfit do sweep.

## Kết quả chính

- Category: `dress`
- Queries hợp lệ: `64`
- Targets: `64`
- FDA mode: `tokens`
- FDA top-k: `6`
- Query token top-k: `8`
- CoPE top-K được rerank: `10`
- Alpha tốt nhất trên dev: `0.05`
- Baseline all: R@10=93.750, R@50=100.000, mean rank=3.02
- Baseline eval-half: R@10=87.500, R@50=100.000, mean rank=3.84
- CoPE+FDA eval-half: R@10=87.500, R@50=100.000, mean rank=3.84
- Chênh eval-half: ΔR@10=+0.000, ΔR@50=+0.000

## Diễn giải

FDA có ảnh hưởng nhưng chưa cải thiện metric eval-half. Điều này thường xảy ra khi score fine-grained bắt đúng vài chi tiết nhưng cũng kéo nhầm các ảnh có texture/patch tương tự.

## Ví dụ thay đổi rank

- `B000WIRE54` + "Has longer sleeves and a different color and has longer sleeves and is more casual" -> `B007UYY43I`: baseline rank 5 -> FDA rank 3
- `B00F4WI9H0` + "Is longer and more elegant and is longer and sleeveless" -> `B0077SM1Z0`: baseline rank 3 -> FDA rank 4
- `B009KQVCAM` + "Its blue with higher neckline and is more blue and higher at the neckline" -> `B00A0I6GSC`: baseline rank 2 -> FDA rank 1
- `B004VJ217Q` + "Has more flair and smaller print and is shorter" -> `B00DC6UJ0K`: baseline rank 7 -> FDA rank 6
- `B00BQVAVRY` + "Is shorter with longer sleeves and is shorter has three quarter length sleeves" -> `B005SMZW7Q`: baseline rank 4 -> FDA rank 3

## File sinh ra

- `results.json`: số liệu đầy đủ, alpha sweep, rank từng query mẫu.
- `REPORT.md`: báo cáo ngắn này.
