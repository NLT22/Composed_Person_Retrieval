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
- CoPE top-K được rerank: `10`
- Alpha tốt nhất trên dev: `0.05`
- Baseline all: R@10=78.906, R@50=93.750, mean rank=11.18
- Baseline eval-half: R@10=75.781, R@50=92.969, mean rank=11.85
- CoPE+FDA eval-half: R@10=75.781, R@50=92.969, mean rank=11.87
- Chênh eval-half: ΔR@10=+0.000, ΔR@50=+0.000

## Diễn giải

FDA có ảnh hưởng nhưng chưa cải thiện metric eval-half. Điều này thường xảy ra khi score fine-grained bắt đúng vài chi tiết nhưng cũng kéo nhầm các ảnh có texture/patch tương tự.

## Ví dụ thay đổi rank

- `B005FLWZCA` + "Is a lighter color and is solid white" -> `B0087PLHB6`: baseline rank 9 -> FDA rank 6
- `B00CW6HT7W` + "Is off the shoulder with one shorter sleeve and has longer sleeves" -> `B00DVH1QS4`: baseline rank 8 -> FDA rank 6
- `B009CMY4BS` + "Is gold and strapless and button front longer sleeves" -> `B0091PLEKA`: baseline rank 5 -> FDA rank 4
- `B00F4WI9H0` + "Is longer and more elegant and is longer and sleeveless" -> `B0077SM1Z0`: baseline rank 9 -> FDA rank 10
- `B008UEW5RO` + "Is more transparent and is more solid colored" -> `B00AIBYV1K`: baseline rank 3 -> FDA rank 2
- `B007G6BODI` + "Is darker and shorter in length and is sleeveless" -> `B008PZ0Y62`: baseline rank 8 -> FDA rank 7
- `B0080N9HNU` + "Is shorter and has animal print and the dress has cheetah print and more revealing" -> `B006ZO5GC2`: baseline rank 2 -> FDA rank 1
- `B00AOJVGYQ` + "Is brighter in color and is a light floral pattern" -> `B00AAOD3LO`: baseline rank 2 -> FDA rank 1
- `B00A7N4DC6` + "Is a short dress with a v neck and is a tan coloring with fabric overlapping on top" -> `B00B5VXRTO`: baseline rank 8 -> FDA rank 7
- `B0066NI17E` + "Is longer and has more pink and is pink with sleeves" -> `B00DC0XN0O`: baseline rank 8 -> FDA rank 7

## File sinh ra

- `results.json`: số liệu đầy đủ, alpha sweep, rank từng query mẫu.
- `REPORT.md`: báo cáo ngắn này.
