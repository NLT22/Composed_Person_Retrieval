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
- FDA mode: `tokens`
- FDA top-k: `6`
- Query token top-k: `8`
- CoPE top-K được rerank: `10`
- Alpha tốt nhất trên dev: `0.05`
- Baseline all: R@10=78.906, R@50=93.750, mean rank=11.18
- Baseline eval-half: R@10=75.781, R@50=92.969, mean rank=11.85
- CoPE+FDA eval-half: R@10=75.781, R@50=92.969, mean rank=11.80
- Chênh eval-half: ΔR@10=+0.000, ΔR@50=+0.000

## Diễn giải

FDA có ảnh hưởng nhưng chưa cải thiện metric eval-half. Điều này thường xảy ra khi score fine-grained bắt đúng vài chi tiết nhưng cũng kéo nhầm các ảnh có texture/patch tương tự.

## Ví dụ thay đổi rank

- `B0066NI17E` + "Is longer and has more pink and is pink with sleeves" -> `B00DC0XN0O`: baseline rank 8 -> FDA rank 6
- `B008LTJG3E` + "Has a brighter color and is more of a boot with shorter heels and is a shoe" -> `B00D4JNBC8`: baseline rank 9 -> FDA rank 7
- `B0036DE0R2` + "Has less sleeves with red and gray and is blue with a higher neckline and black belt" -> `B00A16DXWK`: baseline rank 5 -> FDA rank 3
- `B004VM5W1K` + "Is black and strapless and does not cover shoulders" -> `B004VERF8Q`: baseline rank 9 -> FDA rank 7
- `B00F4WI9H0` + "Is longer and more elegant and is longer and sleeveless" -> `B0077SM1Z0`: baseline rank 9 -> FDA rank 10
- `B007G6BODI` + "Is darker and shorter in length and is sleeveless" -> `B008PZ0Y62`: baseline rank 8 -> FDA rank 9
- `B00466ICP4` + "Is black with no straps and red designs and shorter" -> `B00A0TSV2U`: baseline rank 3 -> FDA rank 2
- `B00967AMPG` + "Less full skirt with slit to thigh and has a slit and is more form fitting" -> `B00AWM67OE`: baseline rank 1 -> FDA rank 2
- `B00A7N4DC6` + "Is a short dress with a v neck and is a tan coloring with fabric overlapping on top" -> `B00B5VXRTO`: baseline rank 8 -> FDA rank 7
- `B007FKA7E2` + "Plain black and sleeveless and is darker in color and solid colored" -> `B0029XKWKO`: baseline rank 4 -> FDA rank 5

## File sinh ra

- `results.json`: số liệu đầy đủ, alpha sweep, rank từng query mẫu.
- `REPORT.md`: báo cáo ngắn này.
