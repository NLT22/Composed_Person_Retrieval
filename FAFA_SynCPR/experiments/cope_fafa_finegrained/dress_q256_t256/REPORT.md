# CoPE + FAFA Fine-Grained Dynamic Alignment

## Mục tiêu

Thử áp dụng ngược ý tưởng fine-grained của FAFA vào CoPE/FashionIQ. CoPE gốc dùng embedding mean/variance toàn cục; thí nghiệm này giữ nguyên checkpoint CoPE rồi thêm điểm FDA kiểu FAFA trên patch tokens của target.

## Cách làm

- Baseline: score xác suất gốc của CoPE.
- FDA: lấy vector query composed của CoPE, so với từng patch token của target image, chọn top-k patch giống nhất rồi lấy trung bình.
- Kết hợp: chuẩn hoá score theo từng query, sau đó dùng `score = CoPE_z + alpha * FDA_z`.
- Chọn `alpha` trên nửa đầu tập query, báo cáo trên nửa sau để giảm overfit do sweep.

## Kết quả chính

- Category: `dress`
- Queries hợp lệ: `256`
- Targets: `256`
- FDA top-k: `6`
- Alpha tốt nhất trên dev: `0.1`
- Baseline all: R@10=78.906, R@50=93.750, mean rank=11.18
- Baseline eval-half: R@10=75.781, R@50=92.969, mean rank=11.85
- CoPE+FDA eval-half: R@10=75.781, R@50=92.969, mean rank=11.97
- Chênh eval-half: ΔR@10=+0.000, ΔR@50=+0.000

## Diễn giải

FDA có ảnh hưởng nhưng chưa cải thiện metric eval-half. Điều này thường xảy ra khi score fine-grained bắt đúng vài chi tiết nhưng cũng kéo nhầm các ảnh có texture/patch tương tự.

## Ví dụ thay đổi rank

- `B002TKJL40` + "A solid and is much shorter" -> `B001HK23TS`: baseline rank 63 -> FDA rank 79
- `B00F150ZXQ` + "Is solid white and high low and is slightly shorter and more colorful" -> `B00BOXD234`: baseline rank 88 -> FDA rank 80
- `B008LTJG3E` + "Has a brighter color and is more of a boot with shorter heels and is a shoe" -> `B00D4JNBC8`: baseline rank 9 -> FDA rank 16
- `B007ZDYK2E` + "Shorter skirt and na" -> `B00BFPGAGM`: baseline rank 116 -> FDA rank 122
- `B0081JJ9FO` + "Is a black dress with a shade of grey and is similar in black" -> `B0097C8UAO`: baseline rank 33 -> FDA rank 39
- `B00B923WCQ` + "One shouldered grey dress and Is more blue and has a slit" -> `B005GQPE4K`: baseline rank 36 -> FDA rank 31
- `B0092QK9AY` + "Is solid nude color and has straps instead of long sleeves and has two colors" -> `B007ORZ1PG`: baseline rank 19 -> FDA rank 15
- `B00BT8N1DK` + "Is below the knee and darker and and long" -> `B00DVG348U`: baseline rank 174 -> FDA rank 170
- `B004VJ217Q` + "Has more flair and smaller print and is shorter" -> `B00DC6UJ0K`: baseline rank 32 -> FDA rank 29
- `B0054N0S6E` + "Is darker and has shorter sleeves and does not cover the sholders" -> `B0095WSNZ8`: baseline rank 103 -> FDA rank 100

## File sinh ra

- `results.json`: số liệu đầy đủ, alpha sweep, rank từng query mẫu.
- `REPORT.md`: báo cáo ngắn này.
