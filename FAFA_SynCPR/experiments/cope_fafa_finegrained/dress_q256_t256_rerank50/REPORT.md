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
- Queries hợp lệ: `256`
- Targets: `256`
- FDA top-k: `6`
- CoPE top-K được rerank: `50`
- Alpha tốt nhất trên dev: `0.1`
- Baseline all: R@10=78.906, R@50=93.750, mean rank=11.18
- Baseline eval-half: R@10=75.781, R@50=92.969, mean rank=11.85
- CoPE+FDA eval-half: R@10=75.781, R@50=92.969, mean rank=11.88
- Chênh eval-half: ΔR@10=+0.000, ΔR@50=+0.000

## Diễn giải

FDA có ảnh hưởng nhưng chưa cải thiện metric eval-half. Điều này thường xảy ra khi score fine-grained bắt đúng vài chi tiết nhưng cũng kéo nhầm các ảnh có texture/patch tương tự.

## Ví dụ thay đổi rank

- `B008LTJG3E` + "Has a brighter color and is more of a boot with shorter heels and is a shoe" -> `B00D4JNBC8`: baseline rank 9 -> FDA rank 16
- `B0081JJ9FO` + "Is a black dress with a shade of grey and is similar in black" -> `B0097C8UAO`: baseline rank 33 -> FDA rank 39
- `B00B923WCQ` + "One shouldered grey dress and Is more blue and has a slit" -> `B005GQPE4K`: baseline rank 36 -> FDA rank 31
- `B0092QK9AY` + "Is solid nude color and has straps instead of long sleeves and has two colors" -> `B007ORZ1PG`: baseline rank 19 -> FDA rank 15
- `B0016LHFE6` + "Is sexier and more revealing and it has no collar and has a shorter sleeve" -> `B009LIHN3O`: baseline rank 47 -> FDA rank 43
- `B004VJ217Q` + "Has more flair and smaller print and is shorter" -> `B00DC6UJ0K`: baseline rank 32 -> FDA rank 29
- `B005FLWZCA` + "Is a lighter color and is solid white" -> `B0087PLHB6`: baseline rank 9 -> FDA rank 6
- `B00BTT3R0Q` + "Is no sleeves and darker color and has markings in black and red flowers" -> `B00CCBVN1K`: baseline rank 34 -> FDA rank 37
- `B00CW6HT7W` + "Is off the shoulder with one shorter sleeve and has longer sleeves" -> `B00DVH1QS4`: baseline rank 8 -> FDA rank 6
- `B007SV9A5A` + "Is more floral print and less striped and white" -> `B005EOD18K`: baseline rank 12 -> FDA rank 10

## File sinh ra

- `results.json`: số liệu đầy đủ, alpha sweep, rank từng query mẫu.
- `REPORT.md`: báo cáo ngắn này.
