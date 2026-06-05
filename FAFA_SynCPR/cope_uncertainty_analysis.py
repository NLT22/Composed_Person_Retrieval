#!/usr/bin/env python
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
COPE_DIR = ROOT / "ACL25-CoPE"
sys.path.insert(0, str(COPE_DIR))


def summarize(values):
    x = torch.as_tensor(values, dtype=torch.float32)
    if x.numel() == 0:
        return {}
    qs = torch.quantile(x, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "p05": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
        "max": float(x.max()),
    }


def pearson(x, y):
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    if x.numel() < 2 or y.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return None
    return float((x @ y) / denom)


def feature_uncertainty(feat):
    logvar = feat["var"].float()
    std = torch.exp(0.5 * logvar)
    var = torch.exp(logvar)
    return {
        "logvar_mean": logvar.mean(dim=-1),
        "std_mean": std.mean(dim=-1),
        "var_trace": var.sum(dim=-1),
    }


@torch.no_grad()
def analyze_cope_fashioniq(args):
    from models.model import ProbCLIP_CIR
    from util.data import build_data
    from omegaconf import OmegaConf

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = OmegaConf.load(args.config)
    config.device = str(device)
    config.data.data_path = str((ROOT / args.data_path).resolve())
    config.data.category = args.category
    config.validation.query_batch_size = args.batch_size
    config.validation.target_batch_size = args.batch_size
    config.validation.num_workers = args.num_workers
    config.validation.pin_memory = device.type == "cuda"
    if args.max_queries:
        config.data.val_max_samples = args.max_queries
    if args.max_targets:
        config.data.target_max_samples = args.max_targets

    model = ProbCLIP_CIR(path=config.model.path, local_files_only=args.local_files_only).to(device)
    state_dict = torch.load(Path(args.checkpoint) / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    data = build_data(config, preprocess=model.preprocess)
    val_loader = data["val_loader"]
    target_loader = data["target_loader"]

    target_names = []
    target_feats = []
    print("Encoding FashionIQ targets.")
    for batch in tqdm(target_loader, desc="targets"):
        feat = model.encode_target(batch["img"].to(device, non_blocking=True))
        target_names.extend(batch["img_name"])
        target_feats.append({k: v.detach().cpu() for k, v in feat.items()})
    target_mean = torch.cat([f["mean"] for f in target_feats]).to(device)
    target_var = torch.cat([f["var"] for f in target_feats]).to(device)
    target_unc = feature_uncertainty({"var": target_var.cpu()})
    target_index = {name: idx for idx, name in enumerate(target_names)}

    rows = []
    print("Encoding FashionIQ queries.")
    for batch in tqdm(val_loader, desc="queries"):
        ref = batch["ref_img"].to(device, non_blocking=True)
        input_ids = model.tokenize(batch["text_instruction"], padding="max_length", return_tensors="pt").input_ids.to(device)
        query = model.encode_query(ref, input_ids)
        q_unc = feature_uncertainty({k: v.detach().cpu() for k, v in query.items()})
        q_mean = query["mean"]
        q_var = query["var"]

        q_std = torch.exp(0.5 * q_var).unsqueeze(1)
        t_std = torch.exp(0.5 * target_var).unsqueeze(0)
        mu_dist = ((q_mean.unsqueeze(1) - target_mean.unsqueeze(0)) ** 2).sum(-1)
        sigma_dist = ((q_std - t_std) ** 2).sum(-1)
        cross = 2 * q_mean.shape[-1] * q_std.mean(-1) * t_std.mean(-1)
        dist = mu_dist + sigma_dist + cross
        sorted_idx = torch.argsort(dist, dim=-1)
        top_vals = torch.gather(dist, 1, sorted_idx[:, : min(50, dist.shape[1])])
        margin = top_vals[:, 1] - top_vals[:, 0] if top_vals.shape[1] > 1 else torch.zeros(top_vals.shape[0], device=device)
        entropy = torch.softmax(-top_vals[:, : min(10, top_vals.shape[1])], dim=-1)
        entropy = -(entropy * entropy.clamp_min(1e-12).log()).sum(-1)

        for i, tgt_name in enumerate(batch["tgt_img_name"]):
            gt_idx = target_index.get(tgt_name)
            if gt_idx is None:
                continue
            rank = int((sorted_idx[i] == gt_idx).nonzero(as_tuple=False)[0, 0].item()) + 1
            rows.append(
                {
                    "query_logvar_mean": float(q_unc["logvar_mean"][i]),
                    "query_std_mean": float(q_unc["std_mean"][i]),
                    "query_var_trace": float(q_unc["var_trace"][i]),
                    "target_logvar_mean": float(target_unc["logvar_mean"][gt_idx]),
                    "target_std_mean": float(target_unc["std_mean"][gt_idx]),
                    "target_var_trace": float(target_unc["var_trace"][gt_idx]),
                    "rank": rank,
                    "hit10": 1.0 if rank <= 10 else 0.0,
                    "hit50": 1.0 if rank <= 50 else 0.0,
                    "top1_margin": float(margin[i]),
                    "top10_entropy": float(entropy[i]),
                    "gt_distance": float(dist[i, gt_idx]),
                    "top1_distance": float(top_vals[i, 0]),
                }
            )

    out = summarize_rows(rows, domain="FashionIQ", category=args.category)
    out["checkpoint"] = str(args.checkpoint)
    out["num_targets"] = len(target_names)
    out["settings"] = {k: v for k, v in vars(args).items() if k != "func"}
    write_outputs(out, args.output_dir)


def summarize_rows(rows, domain, category=None):
    result = {"domain": domain, "category": category, "num_queries": len(rows)}
    if not rows:
        return result
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    result["metrics"] = {k: summarize([r[k] for r in rows]) for k in keys}
    result["recall"] = {
        "R10": 100.0 * sum(r.get("hit10", 0.0) for r in rows) / len(rows),
        "R50": 100.0 * sum(r.get("hit50", 0.0) for r in rows) / len(rows),
    }
    result["correlations"] = {
        "query_std_mean_vs_rank": pearson([r["query_std_mean"] for r in rows], [r["rank"] for r in rows]),
        "query_std_mean_vs_hit10": pearson([r["query_std_mean"] for r in rows], [r["hit10"] for r in rows]),
        "query_var_trace_vs_rank": pearson([r["query_var_trace"] for r in rows], [r["rank"] for r in rows]),
        "top10_entropy_vs_rank": pearson([r["top10_entropy"] for r in rows], [r["rank"] for r in rows]),
        "top1_margin_vs_hit10": pearson([r["top1_margin"] for r in rows], [r["hit10"] for r in rows]),
    }
    return result


def fda_pair_scores(qfeats, target_tokens, fda_k=6):
    scores = torch.einsum("bd,gld->bgl", qfeats.float(), target_tokens.float())
    topk = torch.topk(scores, k=fda_k, dim=-1).values
    return topk.mean(-1), topk.std(-1, unbiased=False)


@torch.no_grad()
def analyze_fafa_syncpr(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows = []
    shard_paths = sorted(Path(args.syncpr_features).glob("shard_*.pt"))
    if args.max_shards:
        shard_paths = shard_paths[: args.max_shards]
    for shard_path in tqdm(shard_paths, desc="SynCPR shards"):
        shard = torch.load(shard_path, map_location="cpu")
        qfeats = shard["qfeats"]
        target = shard["target_tokens"]
        if args.max_rows_per_shard:
            qfeats = qfeats[: args.max_rows_per_shard]
            target = target[: args.max_rows_per_shard]
        q = qfeats.to(device)
        t = target.to(device)
        scores, token_std = fda_pair_scores(q, t, args.fda_k)
        diag = torch.diag(scores)
        hard_scores = scores.clone()
        hard_scores.fill_diagonal_(-1e4)
        top_vals, top_idx = torch.topk(hard_scores, k=min(args.topk, hard_scores.shape[1] - 1), dim=1)
        margin = diag - top_vals[:, 0]
        entropy = torch.softmax(top_vals[:, : min(10, top_vals.shape[1])], dim=-1)
        entropy = -(entropy * entropy.clamp_min(1e-12).log()).sum(-1)
        pos_token_std = torch.diag(token_std)
        rank = 1 + (scores > diag.unsqueeze(1)).sum(dim=1)
        hit1 = (rank == 1).float()
        for i in range(q.shape[0]):
            rows.append(
                {
                    "positive_score": float(diag[i].cpu()),
                    "hard_negative_score": float(top_vals[i, 0].cpu()),
                    "margin": float(margin[i].cpu()),
                    "top10_entropy": float(entropy[i].cpu()),
                    "positive_token_std": float(pos_token_std[i].cpu()),
                    "rank_within_shard": float(rank[i].cpu()),
                    "hit1_within_shard": float(hit1[i].cpu()),
                }
            )
    out = summarize_rows_syncpr(rows)
    out["settings"] = {k: v for k, v in vars(args).items() if k != "func"}
    write_outputs(out, args.output_dir)


def summarize_rows_syncpr(rows):
    result = {"domain": "SynCPR/FAFA proxy", "num_pairs": len(rows)}
    if not rows:
        return result
    keys = rows[0].keys()
    result["metrics"] = {k: summarize([r[k] for r in rows]) for k in keys}
    result["hit1_within_shard"] = 100.0 * sum(r["hit1_within_shard"] for r in rows) / len(rows)
    result["correlations"] = {
        "margin_vs_hit1": pearson([r["margin"] for r in rows], [r["hit1_within_shard"] for r in rows]),
        "top10_entropy_vs_hit1": pearson([r["top10_entropy"] for r in rows], [r["hit1_within_shard"] for r in rows]),
        "positive_token_std_vs_hit1": pearson([r["positive_token_std"] for r in rows], [r["hit1_within_shard"] for r in rows]),
        "positive_token_std_vs_margin": pearson([r["positive_token_std"] for r in rows], [r["margin"] for r in rows]),
    }
    return result


def write_outputs(payload, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# CoPE / FAFA Uncertainty Analysis",
        "",
        f"- Domain: `{payload.get('domain')}`",
        f"- Category: `{payload.get('category')}`",
        f"- Count: `{payload.get('num_queries', payload.get('num_pairs'))}`",
        "",
        "## Key Metrics",
        "",
    ]
    for name, stats in payload.get("metrics", {}).items():
        if not stats:
            continue
        lines.append(f"- `{name}`: mean={stats['mean']:.6f}, median={stats['median']:.6f}, p05={stats['p05']:.6f}, p95={stats['p95']:.6f}")
    if "recall" in payload:
        lines += ["", "## Recall", ""]
        for k, v in payload["recall"].items():
            lines.append(f"- `{k}`: {v:.3f}")
    if "hit1_within_shard" in payload:
        lines += ["", "## Hit", "", f"- `hit1_within_shard`: {payload['hit1_within_shard']:.3f}"]
    lines += ["", "## Correlations", ""]
    for k, v in payload.get("correlations", {}).items():
        lines.append(f"- `{k}`: {v}")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_dir / 'results.json'}")
    print(f"Saved {out_dir / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    fiq = sub.add_parser("cope-fashioniq")
    fiq.add_argument("--config", default=str(COPE_DIR / "config_fiq_paper.yaml"))
    fiq.add_argument("--data-path", default="ACL25-CoPE/fashionIQ_dataset")
    fiq.add_argument("--checkpoint", default=str(COPE_DIR / "checkpoints/fiq_dress/best_model"))
    fiq.add_argument("--category", default="dress", choices=["dress", "shirt", "toptee"])
    fiq.add_argument("--device", default="cuda:0")
    fiq.add_argument("--batch-size", type=int, default=32)
    fiq.add_argument("--num-workers", type=int, default=4)
    fiq.add_argument("--max-queries", type=int)
    fiq.add_argument("--max-targets", type=int)
    fiq.add_argument("--local-files-only", action="store_true")
    fiq.add_argument("--output-dir", default="experiments/uncertainty/cope_fashioniq_dress")
    fiq.set_defaults(func=analyze_cope_fashioniq)

    syncpr = sub.add_parser("fafa-syncpr")
    syncpr.add_argument("--syncpr-features", default="experiments/fafa_pretrained/syncpr_features")
    syncpr.add_argument("--device", default="cuda")
    syncpr.add_argument("--fda-k", type=int, default=6)
    syncpr.add_argument("--topk", type=int, default=32)
    syncpr.add_argument("--max-shards", type=int, default=4)
    syncpr.add_argument("--max-rows-per-shard", type=int, default=2048)
    syncpr.add_argument("--output-dir", default="experiments/uncertainty/fafa_syncpr_proxy")
    syncpr.set_defaults(func=analyze_fafa_syncpr)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
