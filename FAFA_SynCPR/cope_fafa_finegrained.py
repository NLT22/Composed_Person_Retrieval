#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
COPE_DIR = ROOT / "ACL25-CoPE"
sys.path.insert(0, str(COPE_DIR))


def zscore_per_query(scores):
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True).clamp_min(1e-6)


def probabilistic_score(query_mean, query_var, target_mean, target_var, chunk_size=512):
    chunks = []
    for start in range(0, target_mean.shape[0], chunk_size):
        end = min(start + chunk_size, target_mean.shape[0])
        t_mean = target_mean[start:end]
        t_var = target_var[start:end]
        q_std = torch.exp(0.5 * query_var).unsqueeze(1)
        t_std = torch.exp(0.5 * t_var).unsqueeze(0)
        mu_dist = ((query_mean.unsqueeze(1) - t_mean.unsqueeze(0)) ** 2).sum(-1)
        sigma_dist = ((q_std - t_std) ** 2).sum(-1)
        cross = 2 * query_mean.shape[-1] * q_std.mean(-1) * t_std.mean(-1)
        chunks.append(-(mu_dist + sigma_dist + cross))
    return torch.cat(chunks, dim=1)


def fafa_fda_score(query_mean, target_tokens, fda_k=6, chunk_size=256):
    chunks = []
    k = min(fda_k, target_tokens.shape[1])
    for start in range(0, target_tokens.shape[0], chunk_size):
        end = min(start + chunk_size, target_tokens.shape[0])
        sims = torch.einsum("bd,nld->bnl", query_mean.float(), target_tokens[start:end].float())
        chunks.append(torch.topk(sims, k=k, dim=-1).values.mean(dim=-1))
    return torch.cat(chunks, dim=1)


def fafa_token_fda_score(query_tokens, target_tokens, fda_k=6, query_k=8, q_chunk_size=16, target_chunk_size=32):
    rows = []
    target_k = min(fda_k, target_tokens.shape[1])
    query_k = min(query_k, query_tokens.shape[1])
    for q_start in range(0, query_tokens.shape[0], q_chunk_size):
        q_end = min(q_start + q_chunk_size, query_tokens.shape[0])
        q = query_tokens[q_start:q_end].float()
        chunks = []
        for t_start in range(0, target_tokens.shape[0], target_chunk_size):
            t_end = min(t_start + target_chunk_size, target_tokens.shape[0])
            sims = torch.einsum("bqd,nld->bnql", q, target_tokens[t_start:t_end].float())
            per_query_token = torch.topk(sims, k=target_k, dim=-1).values.mean(dim=-1)
            chunks.append(torch.topk(per_query_token, k=query_k, dim=-1).values.mean(dim=-1))
        rows.append(torch.cat(chunks, dim=1))
    return torch.cat(rows, dim=0)


def ranks_from_scores(scores, gt_indices):
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = []
    for row, gt_idx in enumerate(gt_indices):
        pos = (order[row] == gt_idx).nonzero(as_tuple=False)
        ranks.append(int(pos[0, 0].item()) + 1 if pos.numel() else None)
    return ranks


def recall(ranks, k):
    valid = [r for r in ranks if r is not None]
    if not valid:
        return 0.0
    return 100.0 * sum(r <= k for r in valid) / len(valid)


def mean_rank(ranks):
    valid = [r for r in ranks if r is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def metrics_for_scores(scores, gt_indices):
    ranks = ranks_from_scores(scores, gt_indices)
    return {
        "num_queries": len([r for r in ranks if r is not None]),
        "ranks": ranks,
        "R10": recall(ranks, 10),
        "R50": recall(ranks, 50),
        "mean_rank": mean_rank(ranks),
    }


@torch.no_grad()
def target_token_features(model, images):
    out = model.backbone.vision_model(pixel_values=images)
    mean = F.normalize(model.backbone.visual_projection(out.pooler_output), dim=-1, eps=1e-6)
    var = model.backbone.vision_uncertainty_pooler(out.last_hidden_state_var)
    tokens = model.backbone.visual_projection(out.last_hidden_state[:, 1:, :])
    tokens = F.normalize(tokens, dim=-1, eps=1e-6)
    return {"mean": mean, "var": var, "tokens": tokens}


@torch.no_grad()
def query_features(model, ref_imgs, input_ids):
    text = model.backbone.get_text_features(input_ids)
    ref_out = model.backbone.vision_model(pixel_values=ref_imgs, feat_modulate=text["mean"])
    ref_mean = model.backbone.visual_projection(ref_out.pooler_output)
    query_mean = F.normalize(text["mean"] + ref_mean, dim=-1, eps=1e-6)
    query_tokens = model.backbone.visual_projection(ref_out.last_hidden_state[:, 1:, :])
    query_tokens = F.normalize(query_tokens + text["mean"].unsqueeze(1), dim=-1, eps=1e-6)
    query_var = torch.logsumexp(
        torch.stack([text["var"], model.backbone.vision_uncertainty_pooler(ref_out.last_hidden_state_var)], dim=0),
        dim=0,
    )
    return {"mean": query_mean, "var": query_var, "tokens": query_tokens}


def combine_scores(cope_scores, fda_scores, alpha, rerank_topk=0):
    cope_z = zscore_per_query(cope_scores)
    fda_z = zscore_per_query(fda_scores)
    if not rerank_topk:
        return cope_z + alpha * fda_z

    combined = cope_z.clone()
    k = min(rerank_topk, cope_scores.shape[1])
    top_idx = torch.topk(cope_scores, k=k, dim=1).indices
    row_idx = torch.arange(cope_scores.shape[0], device=cope_scores.device).unsqueeze(1)
    combined[row_idx, top_idx] = cope_z[row_idx, top_idx] + alpha * fda_z[row_idx, top_idx]
    return combined


def choose_alpha_on_dev(cope_scores, fda_scores, gt_indices, alphas, rerank_topk=0):
    split = max(1, len(gt_indices) // 2)
    dev = list(range(split))
    eval_idx = list(range(split, len(gt_indices)))
    if not eval_idx:
        eval_idx = dev

    rows = []
    best = None
    for alpha in alphas:
        combined = combine_scores(cope_scores, fda_scores, alpha, rerank_topk)
        dev_metrics = metrics_for_scores(combined[dev], [gt_indices[i] for i in dev])
        eval_metrics = metrics_for_scores(combined[eval_idx], [gt_indices[i] for i in eval_idx])
        row = {
            "alpha": alpha,
            "dev_R10": dev_metrics["R10"],
            "dev_R50": dev_metrics["R50"],
            "dev_mean_rank": dev_metrics["mean_rank"],
            "eval_R10": eval_metrics["R10"],
            "eval_R50": eval_metrics["R50"],
            "eval_mean_rank": eval_metrics["mean_rank"],
        }
        rows.append(row)
        key = (dev_metrics["R10"], dev_metrics["R50"], -(dev_metrics["mean_rank"] or 1e9))
        if best is None or key > best[0]:
            best = (key, alpha, row)
    return best[1], rows, dev, eval_idx


def write_report(out_dir, payload):
    baseline = payload["baseline"]
    fda = payload["best_fda"]
    delta_r10 = fda["eval_R10"] - payload["baseline_eval"]["R10"]
    delta_r50 = fda["eval_R50"] - payload["baseline_eval"]["R50"]
    lines = [
        "# CoPE + FAFA Fine-Grained Dynamic Alignment",
        "",
        "## Mục tiêu",
        "",
        "Thử áp dụng ngược ý tưởng fine-grained của FAFA vào CoPE/FashionIQ. CoPE gốc dùng embedding mean/variance toàn cục; thí nghiệm này giữ nguyên checkpoint CoPE rồi thêm điểm FDA kiểu FAFA trên patch tokens của target.",
        "",
        "## Cách làm",
        "",
        "- Baseline: score xác suất gốc của CoPE.",
        "- FDA: lấy vector query composed của CoPE, so với từng patch token của target image, chọn top-k patch giống nhất rồi lấy trung bình.",
        "- Kết hợp: chuẩn hoá score theo từng query, sau đó dùng `score = CoPE_z + alpha * FDA_z`.",
        "- Nếu bật `rerank_topk`, FDA chỉ được áp dụng trong top-K ứng viên CoPE, phần còn lại giữ nguyên score CoPE.",
        "- Chọn `alpha` trên nửa đầu tập query, báo cáo trên nửa sau để giảm overfit do sweep.",
        "",
        "## Kết quả chính",
        "",
        f"- Category: `{payload['settings']['category']}`",
        f"- Queries hợp lệ: `{baseline['num_queries']}`",
        f"- Targets: `{payload['num_targets']}`",
        f"- FDA mode: `{payload['settings']['fda_mode']}`",
        f"- FDA top-k: `{payload['settings']['fda_k']}`",
        f"- Query token top-k: `{payload['settings']['query_k']}`",
        f"- CoPE top-K được rerank: `{payload['settings']['rerank_topk'] or 'all targets'}`",
        f"- Alpha tốt nhất trên dev: `{payload['best_alpha']}`",
        f"- Baseline all: R@10={baseline['R10']:.3f}, R@50={baseline['R50']:.3f}, mean rank={baseline['mean_rank']:.2f}",
        f"- Baseline eval-half: R@10={payload['baseline_eval']['R10']:.3f}, R@50={payload['baseline_eval']['R50']:.3f}, mean rank={payload['baseline_eval']['mean_rank']:.2f}",
        f"- CoPE+FDA eval-half: R@10={fda['eval_R10']:.3f}, R@50={fda['eval_R50']:.3f}, mean rank={fda['eval_mean_rank']:.2f}",
        f"- Chênh eval-half: ΔR@10={delta_r10:+.3f}, ΔR@50={delta_r50:+.3f}",
        "",
        "## Diễn giải",
        "",
    ]
    if delta_r10 > 0 or delta_r50 > 0:
        lines.append("FDA có tín hiệu hữu ích: nó đang thêm thông tin cục bộ mà CoPE global score chưa khai thác hết. Hướng kế tiếp hợp lý là mở rộng sang full val và/hoặc train một gate rất nhỏ để học khi nào nên tin FDA.")
    elif payload["best_alpha"] == 0:
        lines.append("FDA chưa thắng baseline trên split này: sweep chọn alpha=0, nghĩa là patch-level similarity thô chưa đủ ổn để thay ranking CoPE. Hướng kế tiếp không nên train lớn ngay; cần cải thiện biểu diễn query token hoặc chỉ dùng FDA như reranker top-K có điều kiện.")
    else:
        lines.append("FDA có ảnh hưởng nhưng chưa cải thiện metric eval-half. Điều này thường xảy ra khi score fine-grained bắt đúng vài chi tiết nhưng cũng kéo nhầm các ảnh có texture/patch tương tự.")
    lines += [
        "",
        "## Ví dụ thay đổi rank",
        "",
    ]
    for item in payload["rank_changes"][:10]:
        lines.append(
            f"- `{item['ref']}` + \"{item['text']}\" -> `{item['target']}`: "
            f"baseline rank {item['baseline_rank']} -> FDA rank {item['fda_rank']}"
        )
    lines += [
        "",
        "## File sinh ra",
        "",
        "- `results.json`: số liệu đầy đủ, alpha sweep, rank từng query mẫu.",
        "- `REPORT.md`: báo cáo ngắn này.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def run(args):
    from models.model import ProbCLIP_CIR
    from omegaconf import OmegaConf
    from util.data import build_data

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(args.config)
    config.device = str(device)
    config.data.data_path = str((ROOT / args.data_path).resolve())
    config.data.category = args.category
    config.validation.query_batch_size = args.batch_size
    config.validation.target_batch_size = args.batch_size
    config.validation.num_workers = args.num_workers
    config.validation.pin_memory = device.type == "cuda"
    config.data.val_max_samples = None if args.max_queries <= 0 else args.max_queries
    config.data.target_max_samples = None if args.max_targets <= 0 else args.max_targets

    model = ProbCLIP_CIR(path=config.model.path, local_files_only=args.local_files_only).to(device)
    state_dict = torch.load(Path(args.checkpoint) / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    data = build_data(config, preprocess=model.preprocess)
    target_loader = data["target_loader"]
    val_loader = data["val_loader"]

    target_names = []
    target_parts = []
    for batch in tqdm(target_loader, desc="targets"):
        feats = target_token_features(model, batch["img"].to(device, non_blocking=True))
        target_names.extend(batch["img_name"])
        target_parts.append({k: v.detach().cpu() for k, v in feats.items()})

    target_mean = torch.cat([x["mean"] for x in target_parts]).to(device)
    target_var = torch.cat([x["var"] for x in target_parts]).to(device)
    target_tokens = torch.cat([x["tokens"] for x in target_parts]).to(device)
    target_index = {name: idx for idx, name in enumerate(target_names)}

    query_parts = []
    meta = []
    gt_indices = []
    for batch in tqdm(val_loader, desc="queries"):
        ref = batch["ref_img"].to(device, non_blocking=True)
        input_ids = model.tokenize(batch["text_instruction"], padding="max_length", return_tensors="pt").input_ids.to(device)
        feats = query_features(model, ref, input_ids)
        query_parts.append({k: v.detach().cpu() for k, v in feats.items()})
        for i, tgt in enumerate(batch["tgt_img_name"]):
            idx = target_index.get(tgt)
            if idx is None:
                continue
            gt_indices.append(idx)
            meta.append(
                {
                    "ref": batch["ref_img_name"][i],
                    "target": tgt,
                    "text": batch["text_instruction"][i],
                }
            )

    query_mean = torch.cat([x["mean"] for x in query_parts]).to(device)[: len(gt_indices)]
    query_var = torch.cat([x["var"] for x in query_parts]).to(device)[: len(gt_indices)]
    query_tokens = torch.cat([x["tokens"] for x in query_parts]).to(device)[: len(gt_indices)]

    cope_scores = probabilistic_score(query_mean, query_var, target_mean, target_var, args.score_chunk)
    if args.fda_mode == "tokens":
        fda_scores = fafa_token_fda_score(
            query_tokens,
            target_tokens,
            args.fda_k,
            args.query_k,
            args.query_score_chunk,
            args.target_score_chunk,
        )
    else:
        fda_scores = fafa_fda_score(query_mean, target_tokens, args.fda_k, args.score_chunk)

    baseline = metrics_for_scores(cope_scores, gt_indices)
    alphas = [float(x) for x in args.alphas.split(",")]
    best_alpha, sweep_rows, dev_idx, eval_idx = choose_alpha_on_dev(
        cope_scores, fda_scores, gt_indices, alphas, args.rerank_topk
    )
    combined = combine_scores(cope_scores, fda_scores, best_alpha, args.rerank_topk)
    fda_metrics = metrics_for_scores(combined, gt_indices)
    baseline_eval = metrics_for_scores(cope_scores[eval_idx], [gt_indices[i] for i in eval_idx])

    base_ranks = baseline["ranks"]
    fda_ranks = fda_metrics["ranks"]
    changes = []
    for i, item in enumerate(meta):
        if base_ranks[i] is None or fda_ranks[i] is None or base_ranks[i] == fda_ranks[i]:
            continue
        row = dict(item)
        row["baseline_rank"] = base_ranks[i]
        row["fda_rank"] = fda_ranks[i]
        row["rank_delta"] = base_ranks[i] - fda_ranks[i]
        changes.append(row)
    changes.sort(key=lambda x: abs(x["rank_delta"]), reverse=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {k: v for k, v in vars(args).items()},
        "checkpoint": str(args.checkpoint),
        "num_targets": len(target_names),
        "baseline": {k: v for k, v in baseline.items() if k != "ranks"},
        "baseline_eval": {k: v for k, v in baseline_eval.items() if k != "ranks"},
        "best_alpha": best_alpha,
        "best_fda": next(row for row in sweep_rows if row["alpha"] == best_alpha),
        "all_fda": {k: v for k, v in fda_metrics.items() if k != "ranks"},
        "alpha_sweep": sweep_rows,
        "rank_changes": changes,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir, payload)
    print(json.dumps(payload["best_fda"], indent=2, ensure_ascii=False))
    print(f"Wrote {out_dir / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description="Apply FAFA-style fine-grained dynamic alignment to CoPE FashionIQ retrieval.")
    parser.add_argument("--config", default=str(COPE_DIR / "config_fiq_paper.yaml"))
    parser.add_argument("--checkpoint", default=str(COPE_DIR / "checkpoints/fiq_dress_paper_fix_distance/best_model"))
    parser.add_argument("--data-path", default="ACL25-CoPE/fashionIQ_dataset")
    parser.add_argument("--category", default="dress", choices=["dress", "shirt", "toptee"])
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "experiments/cope_fafa_finegrained/dress_subset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-queries", type=int, default=64)
    parser.add_argument("--max-targets", type=int, default=64)
    parser.add_argument("--fda-k", type=int, default=6)
    parser.add_argument("--query-k", type=int, default=8)
    parser.add_argument("--fda-mode", default="mean", choices=["mean", "tokens"])
    parser.add_argument("--score-chunk", type=int, default=128)
    parser.add_argument("--query-score-chunk", type=int, default=16)
    parser.add_argument("--target-score-chunk", type=int, default=32)
    parser.add_argument("--rerank-topk", type=int, default=0, help="Only apply FDA inside CoPE top-K candidates. 0 means combine all targets.")
    parser.add_argument("--alphas", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1,1.5,2")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
