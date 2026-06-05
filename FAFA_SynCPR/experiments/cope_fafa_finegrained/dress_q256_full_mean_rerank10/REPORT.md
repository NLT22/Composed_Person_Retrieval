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
- Targets: `3817`
- FDA mode: `mean`
- FDA top-k: `6`
- Query token top-k: `8`
- CoPE top-K được rerank: `10`
- Alpha tốt nhất trên dev: `0.05`
- Baseline all: R@10=36.719, R@50=63.672, mean rank=151.80
- Baseline eval-half: R@10=33.594, R@50=61.719, mean rank=160.70
- CoPE+FDA eval-half: R@10=33.594, R@50=61.719, mean rank=160.68
- Chênh eval-half: ΔR@10=+0.000, ΔR@50=+0.000

## Diễn giải

FDA có ảnh hưởng nhưng chưa cải thiện metric eval-half. Điều này thường xảy ra khi score fine-grained bắt đúng vài chi tiết nhưng cũng kéo nhầm các ảnh có texture/patch tương tự.

## Ví dụ thay đổi rank

- `B00G6TRZE8` + "A low cut black gown and is in solid black and a lot longer" -> `B003ZJ6GCO`: baseline rank 10 -> FDA rank 5
- `B00FXZX2I4` + "Is solid red and is red" -> `B00997L4QE`: baseline rank 7 -> FDA rank 5
- `B008OKGFF2` + "Is longer sleeved with black skirt and is long sleeve and darker pattern with black skirt" -> `B00ETGEI1I`: baseline rank 2 -> FDA rank 4
- `B0080N9HNU` + "Is shorter and has animal print and the dress has cheetah print and more revealing" -> `B006ZO5GC2`: baseline rank 2 -> FDA rank 1
- `B00967AMPG` + "Less full skirt with slit to thigh and has a slit and is more form fitting" -> `B00AWM67OE`: baseline rank 4 -> FDA rank 3
- `B0094QX5SK` + "Is offwhite with black belt and longer seethrough skirt and is white with a black belr" -> `B00B1AVTWG`: baseline rank 2 -> FDA rank 3
- `B001HK23KM` + "And black and the shoulder straps more resemble a crop top" -> `B002TKJL40`: baseline rank 5 -> FDA rank 6
- `B007W9ZSEA` + "Has more color contrast and is pink and black" -> `B00BU5GGXE`: baseline rank 6 -> FDA rank 5
- `B007Y37P22` + "More revealing and shorter in length and is more strappy and emerald" -> `B006ZR2U8W`: baseline rank 3 -> FDA rank 4
- `B006330SY6` + "Brown and flowers thin scrapes and look very casual and is darker and shorter" -> `B002EANP9W`: baseline rank 1 -> FDA rank 2

## File sinh ra

- `results.json`: số liệu đầy đủ, alpha sweep, rank từng query mẫu.
- `REPORT.md`: báo cáo ngắn này.
