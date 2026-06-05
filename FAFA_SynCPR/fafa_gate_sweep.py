#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from fafa_hard_negative_reranker import (
    HardNegativeReranker,
    evaluate_scores,
    pair_features_all,
    pair_features_candidates,
)


def word_count(text):
    return len(re.findall(r"[A-Za-z0-9]+", text))


def per_query(scores, qids, gids):
    order = torch.argsort(scores, dim=1, descending=True)
    rel = gids[order].eq(qids.view(-1, 1))
    top1 = rel[:, 0].cpu()
    ranks = torch.argmax(rel.int(), dim=1) + 1
    ranks[~rel.any(dim=1)] = scores.size(1) + 1
    aps = []
    for row in rel:
        pos = torch.nonzero(row, as_tuple=False).flatten()
        if pos.numel() == 0:
            aps.append(torch.tensor(0.0))
        else:
            precision = torch.arange(1, pos.numel() + 1, dtype=torch.float32) / (pos.float() + 1.0)
            aps.append(precision.mean())
    return top1, ranks.cpu(), torch.stack(aps).cpu()


@torch.no_grad()
def compute_topk_payload(args):
    payload_path = Path(args.payload_cache)
    if payload_path.exists() and not args.force:
        print(f"Loading cached gate payload from {payload_path}")
        return torch.load(payload_path, map_location="cpu", weights_only=False)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = HardNegativeReranker(hidden=ckpt["args"].get("hidden", 128)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cache = torch.load(args.itcpr_cache, map_location="cpu")
    qfeats = cache["qfeats"]
    gfeats = cache["gfeats"]
    q_gpu = qfeats.to(device)

    base_scores = torch.empty((qfeats.size(0), gfeats.size(0)), dtype=torch.float32)
    for start in tqdm(range(0, gfeats.size(0), args.score_batch_size), desc="base scores"):
        end = min(start + args.score_batch_size, gfeats.size(0))
        base, _ = pair_features_all(q_gpu, gfeats[start:end].to(device), fda_k=args.fda_k)
        base_scores[:, start:end] = (model.base_scale * base).cpu()

    top_vals, top_idx = torch.topk(base_scores, k=args.topk, dim=1)
    delta_top = torch.empty_like(top_vals)
    for start in tqdm(range(0, qfeats.size(0), args.query_batch_size), desc="top-k deltas"):
        end = min(start + args.query_batch_size, qfeats.size(0))
        candidates = gfeats[top_idx[start:end]].to(device)
        _, feats = pair_features_candidates(qfeats[start:end].to(device), candidates, fda_k=args.fda_k)
        delta = model.net(feats).squeeze(-1)
        delta_top[start:end] = (model.delta_scale * delta).cpu()

    payload = {
        "cache": cache,
        "base_scores": base_scores,
        "top_vals": top_vals,
        "top_idx": top_idx,
        "delta_top": delta_top,
        "topk": args.topk,
        "delta_mult": args.delta_mult,
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, payload_path)
    print(f"Saved gate payload to {payload_path}")
    return payload


def evaluate_gate(payload, gate, delta_mult):
    cache = payload["cache"]
    base_scores = payload["base_scores"]
    top_vals = payload["top_vals"]
    top_idx = payload["top_idx"]
    delta_top = payload["delta_top"]
    qids = cache["qids"]
    gids = cache["gids"]

    rows = torch.nonzero(gate, as_tuple=False).flatten()
    scores = base_scores.clone()
    scores[rows.unsqueeze(1), top_idx[rows]] = top_vals[rows] + delta_mult * delta_top[rows]
    metrics = evaluate_scores(scores, qids, gids)
    return scores, rows, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/fafa_hard_negative_reranker/fafa_hard_negative_reranker.pt")
    parser.add_argument("--itcpr-cache", default="experiments/fafa_pretrained/itcpr_fafa_features.pt")
    parser.add_argument("--query-json", default="../data/query.json")
    parser.add_argument("--output", default="experiments/fafa_hard_negative_reranker/itcpr_multi_gate_sweep.json")
    parser.add_argument("--payload-cache", default="experiments/fafa_hard_negative_reranker/gate_payload_top1000.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fda-k", type=int, default=6)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--delta-mult", type=float, default=-0.015)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = compute_topk_payload(args)
    cache = payload["cache"]
    base_scores = payload["base_scores"]
    top_vals = payload["top_vals"]
    qids = cache["qids"]
    gids = cache["gids"]

    query_annos = json.loads(Path(args.query_json).read_text(encoding="utf-8"))
    caption_words = torch.tensor([word_count(anno.get("caption", "")) for anno in query_annos], dtype=torch.long)

    margin12 = top_vals[:, 0] - top_vals[:, 1]
    gap10 = top_vals[:, 0] - top_vals[:, :10].mean(dim=1)
    prob10 = torch.softmax(top_vals[:, :10], dim=1)
    entropy10 = -(prob10 * prob10.clamp_min(1e-12).log()).sum(dim=1)

    base_metrics = evaluate_scores(base_scores, qids, gids)
    base_top1, base_rank, base_ap = per_query(base_scores, qids, gids)

    results = []

    def add_result(name, gate, params):
        scores, rows, metrics = evaluate_gate(payload, gate, args.delta_mult)
        top1, rank, ap = per_query(scores, qids, gids)
        ap_delta = ap - base_ap
        item = {
            "gate": name,
            "params": params,
            "applied_queries": int(rows.numel()),
            "metrics": metrics,
            "top1_improved": int((~base_top1 & top1).sum()),
            "top1_regressed": int((base_top1 & ~top1).sum()),
            "rank_improved": int((rank < base_rank).sum()),
            "rank_regressed": int((rank > base_rank).sum()),
            "ap_positive": int((ap_delta > 0).sum()),
            "ap_negative": int((ap_delta < 0).sum()),
            "mean_ap_delta": float(ap_delta.mean()),
        }
        results.append(item)
        print(
            f"{name} {params} n={item['applied_queries']} "
            f"R1={metrics['R1']:.3f} mAP={metrics['mAP']:.3f} "
            f"top1 +{item['top1_improved']}/-{item['top1_regressed']} "
            f"AP +{item['ap_positive']}/-{item['ap_negative']}"
        )

    margin_qs = [0.10, 0.20, 0.30, 0.40, 0.50]
    gap_qs = [0.10, 0.20, 0.30, 0.40, 0.50]
    entropy_qs = [0.50, 0.60, 0.70, 0.80, 0.90]
    min_words_list = [5, 6, 7, 8]

    for min_words in min_words_list:
        word_gate = caption_words >= min_words
        for q in gap_qs:
            gate = word_gate & (gap10 <= torch.quantile(gap10, q))
            add_result("gap10_caption", gate, {"gap10_quantile": q, "min_caption_words": min_words})

    for min_words in min_words_list:
        word_gate = caption_words >= min_words
        for mq in margin_qs:
            margin_gate = margin12 <= torch.quantile(margin12, mq)
            for gq in [0.30, 0.50, 0.70]:
                gate = word_gate & margin_gate & (gap10 <= torch.quantile(gap10, gq))
                add_result(
                    "margin_gap10_caption",
                    gate,
                    {"margin_quantile": mq, "gap10_quantile": gq, "min_caption_words": min_words},
                )

    for min_words in min_words_list:
        word_gate = caption_words >= min_words
        for mq in margin_qs:
            margin_gate = margin12 <= torch.quantile(margin12, mq)
            for eq in entropy_qs:
                gate = word_gate & margin_gate & (entropy10 >= torch.quantile(entropy10, eq))
                add_result(
                    "margin_entropy_caption",
                    gate,
                    {"margin_quantile": mq, "entropy10_quantile": eq, "min_caption_words": min_words},
                )

    def sort_key(item):
        return (
            item["top1_regressed"] == 0,
            item["metrics"]["R1"],
            item["metrics"]["mAP"],
            -item["ap_negative"],
        )

    best = sorted(results, key=sort_key, reverse=True)[:20]
    out = {
        "baseline": base_metrics,
        "topk": args.topk,
        "delta_mult": args.delta_mult,
        "feature_summary": {
            "margin12": {"median": float(margin12.median()), "min": float(margin12.min()), "max": float(margin12.max())},
            "gap10": {"median": float(gap10.median()), "min": float(gap10.min()), "max": float(gap10.max())},
            "entropy10": {"median": float(entropy10.median()), "min": float(entropy10.min()), "max": float(entropy10.max())},
        },
        "best_zero_top1_regression": best,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved multi-gate sweep to {out_path}")
    print("Best zero-top1-regression candidates:")
    for item in best[:10]:
        print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()
