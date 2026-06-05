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
- FDA mode: `tokens`
- FDA top-k: `6`
- Query token top-k: `8`
- CoPE top-K được rerank: `10`
- Alpha tốt nhất trên dev: `0.1`
- Baseline all: R@10=36.719, R@50=63.672, mean rank=151.80
- Baseline eval-half: R@10=33.594, R@50=61.719, mean rank=160.70
- CoPE+FDA eval-half: R@10=33.594, R@50=61.719, mean rank=160.72
- Chênh eval-half: ΔR@10=+0.000, ΔR@50=+0.000

## Diễn giải

FDA có ảnh hưởng nhưng chưa cải thiện metric eval-half. Điều này thường xảy ra khi score fine-grained bắt đúng vài chi tiết nhưng cũng kéo nhầm các ảnh có texture/patch tương tự.

## Ví dụ thay đổi rank

- `B00G6TRZE8` + "A low cut black gown and is in solid black and a lot longer" -> `B003ZJ6GCO`: baseline rank 10 -> FDA rank 5
- `B002OEYH3Q` + "Has long sleeves and patterned and is in brownish coloring" -> `B008YIGWOS`: baseline rank 2 -> FDA rank 7
- `B003M4Z9OS` + "Patterned dress with zig zags and chevron patern" -> `B009XHND2I`: baseline rank 2 -> FDA rank 6
- `B001HK23KM` + "And black and the shoulder straps more resemble a crop top" -> `B002TKJL40`: baseline rank 5 -> FDA rank 8
- `B00BFGBJ3K` + "Is white in color with short sleeves and is more plain" -> `B006ZKN7UY`: baseline rank 5 -> FDA rank 3
- `B008OKGFF2` + "Is longer sleeved with black skirt and is long sleeve and darker pattern with black skirt" -> `B00ETGEI1I`: baseline rank 2 -> FDA rank 4
- `B00EOW7882` + "Is more revealing and longer and It is longer and has no sleeves" -> `B00D61ZCGW`: baseline rank 10 -> FDA rank 8
- `B00CAMMV5S` + "Is more purple and elegant and is darker purple in colour" -> `B00CAMJMRS`: baseline rank 4 -> FDA rank 2
- `B00GCR5OPG` + "Is lighter and is coral and less revealing" -> `B008TS39QC`: baseline rank 6 -> FDA rank 8
- `B001PKTT7E` + "Is solid colored with straps and is darker and has straps" -> `B004X0VYKI`: baseline rank 3 -> FDA rank 5

## File sinh ra

- `results.json`: số liệu đầy đủ, alpha sweep, rank từng query mẫu.
- `REPORT.md`: báo cáo ngắn này.
