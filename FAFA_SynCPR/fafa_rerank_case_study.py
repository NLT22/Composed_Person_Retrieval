#!/usr/bin/env python
import argparse
import html
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from fafa_hard_negative_reranker import (
    HardNegativeReranker,
    evaluate_scores,
    pair_features_all,
    pair_features_candidates,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def image_path(root, anno):
    return str((Path(root) / anno["file_path"]).resolve())


@torch.no_grad()
def compute_scores(args):
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

    rerank_scores = base_scores.clone()
    margin = top_vals[:, 0] - top_vals[:, 1]
    if args.margin_quantile is None:
        rows = torch.arange(qfeats.size(0))
        margin_threshold = None
    else:
        margin_threshold = float(torch.quantile(margin, args.margin_quantile).item())
        rows = torch.nonzero(margin <= margin_threshold, as_tuple=False).flatten()
    if args.min_caption_words > 0:
        query_annos = load_json(args.query_json)
        word_counts = torch.tensor(
            [len(re.findall(r"[A-Za-z0-9]+", anno.get("caption", ""))) for anno in query_annos],
            dtype=torch.long,
        )
        keep = torch.zeros(qfeats.size(0), dtype=torch.bool)
        keep[rows] = True
        keep &= word_counts >= args.min_caption_words
        rows = torch.nonzero(keep, as_tuple=False).flatten()
    rerank_scores[rows.unsqueeze(1), top_idx[rows]] = top_vals[rows] + args.delta_mult * delta_top[rows]

    return cache, base_scores, rerank_scores, {
        "topk": args.topk,
        "delta_mult": args.delta_mult,
        "margin_quantile": args.margin_quantile,
        "margin_threshold": margin_threshold,
        "min_caption_words": args.min_caption_words,
        "applied_queries": int(rows.numel()),
    }


def per_query(scores, qids, gids):
    order = torch.argsort(scores, dim=1, descending=True)
    sorted_gids = gids[order]
    rel = sorted_gids.eq(qids.view(-1, 1))
    top1 = rel[:, 0]
    ranks = torch.argmax(rel.int(), dim=1) + 1
    ranks[~rel.any(dim=1)] = scores.size(1) + 1
    aps = []
    for row in rel:
        pos = torch.nonzero(row, as_tuple=False).flatten()
        if pos.numel() == 0:
            aps.append(torch.tensor(0.0))
            continue
        precision = torch.arange(1, pos.numel() + 1, dtype=torch.float32) / (pos.float() + 1.0)
        aps.append(precision.mean())
    return {
        "order": order,
        "top1": top1.cpu(),
        "rank": ranks.cpu(),
        "ap": torch.stack(aps).cpu(),
    }


def make_case(i, group, base, rerank, cache, query_annos, gallery_annos, data_root):
    qids = cache["qids"]
    gids = cache["gids"]
    qid = int(qids[i])
    gt_positions = torch.nonzero(gids.eq(qid), as_tuple=False).flatten()
    gt_idx = int(gt_positions[0]) if gt_positions.numel() else None
    base_idx = int(base["order"][i, 0])
    rerank_idx = int(rerank["order"][i, 0])
    anno = query_annos[i]

    def gallery_payload(idx):
        if idx is None:
            return None
        ganno = gallery_annos[idx]
        return {
            "gallery_index": idx,
            "instance_id": int(gids[idx]),
            "person_id": int(ganno["person_id"]),
            "path": image_path(data_root, ganno),
        }

    return {
        "group": group,
        "query_index": int(i),
        "query_id": qid,
        "query_person_id": int(anno["person_id"]),
        "caption": anno["caption"],
        "reference_path": image_path(data_root, anno),
        "base_rank": int(base["rank"][i]),
        "rerank_rank": int(rerank["rank"][i]),
        "rank_delta": int(rerank["rank"][i] - base["rank"][i]),
        "base_ap": float(base["ap"][i]),
        "rerank_ap": float(rerank["ap"][i]),
        "ap_delta": float(rerank["ap"][i] - base["ap"][i]),
        "base_top1": gallery_payload(base_idx),
        "rerank_top1": gallery_payload(rerank_idx),
        "gt_first": gallery_payload(gt_idx),
    }


def select_cases(base, rerank, limit):
    top1_improved = torch.nonzero(~base["top1"] & rerank["top1"], as_tuple=False).flatten()
    top1_regressed = torch.nonzero(base["top1"] & ~rerank["top1"], as_tuple=False).flatten()
    ap_delta = rerank["ap"] - base["ap"]
    ap_improved = torch.topk(ap_delta, k=min(limit, ap_delta.numel())).indices
    ap_regressed = torch.topk(-ap_delta, k=min(limit, ap_delta.numel())).indices
    return [
        ("top1_improved", top1_improved[:limit]),
        ("top1_regressed", top1_regressed[:limit]),
        ("largest_ap_improvements", ap_improved),
        ("largest_ap_regressions", ap_regressed),
    ]


def write_html(path, summary, cases):
    def esc(value):
        return html.escape(str(value))

    cards = []
    for case in cases:
        imgs = []
        for title, payload in [
            ("Reference", {"path": case["reference_path"], "instance_id": case["query_id"], "person_id": case["query_person_id"]}),
            ("GT", case["gt_first"]),
            ("Baseline top1", case["base_top1"]),
            ("Rerank top1", case["rerank_top1"]),
        ]:
            if payload is None:
                continue
            imgs.append(
                f"""
                <figure>
                  <img src="{esc(payload['path'])}" loading="lazy">
                  <figcaption>{esc(title)}<br>iid={esc(payload['instance_id'])}, pid={esc(payload['person_id'])}</figcaption>
                </figure>
                """
            )
        cards.append(
            f"""
            <section class="case">
              <h2>{esc(case['group'])} | qidx={case['query_index']} | qid={case['query_id']}</h2>
              <p class="caption">{esc(case['caption'])}</p>
              <p>rank: {case['base_rank']} -> {case['rerank_rank']} | AP: {case['base_ap']:.3f} -> {case['rerank_ap']:.3f} | delta={case['ap_delta']:.3f}</p>
              <div class="grid">{''.join(imgs)}</div>
            </section>
            """
        )

    metrics = summary["metrics"]
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>FAFA SynCPR Rerank Case Study</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    .summary, .case {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; max-height: 260px; object-fit: contain; background: #f5f5f5; border: 1px solid #ddd; }}
    figcaption {{ font-size: 13px; line-height: 1.35; margin-top: 6px; }}
    h1 {{ margin-top: 0; }}
    h2 {{ font-size: 18px; margin-bottom: 8px; }}
    .caption {{ font-weight: 600; }}
    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FAFA + SynCPR Rerank Case Study</h1>
  <section class="summary">
    <p>Config: <code>{esc(summary['config'])}</code></p>
    <p>Baseline: R1={metrics['baseline']['R1']:.3f}, R5={metrics['baseline']['R5']:.3f}, R10={metrics['baseline']['R10']:.3f}, mAP={metrics['baseline']['mAP']:.3f}</p>
    <p>Rerank: R1={metrics['rerank']['R1']:.3f}, R5={metrics['rerank']['R5']:.3f}, R10={metrics['rerank']['R10']:.3f}, mAP={metrics['rerank']['mAP']:.3f}</p>
    <p>Top1 improved={summary['top1']['improved_count']}, regressed={summary['top1']['regressed_count']}</p>
  </section>
  {''.join(cards)}
</body>
</html>
"""
    Path(path).write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/fafa_hard_negative_reranker/fafa_hard_negative_reranker.pt")
    parser.add_argument("--itcpr-cache", default="experiments/fafa_pretrained/itcpr_fafa_features.pt")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--query-json", default="../data/query.json")
    parser.add_argument("--gallery-json", default="../data/gallery.json")
    parser.add_argument("--output-dir", default="experiments/fafa_hard_negative_reranker/case_study")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fda-k", type=int, default=6)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--delta-mult", type=float, default=-0.015)
    parser.add_argument("--margin-quantile", type=float, default=0.30)
    parser.add_argument("--min-caption-words", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache, base_scores, rerank_scores, config = compute_scores(args)
    qids = cache["qids"]
    gids = cache["gids"]
    base_metrics = evaluate_scores(base_scores, qids, gids)
    rerank_metrics = evaluate_scores(rerank_scores, qids, gids)
    base = per_query(base_scores, qids, gids)
    rerank = per_query(rerank_scores, qids, gids)

    query_annos = load_json(args.query_json)
    gallery_annos = load_json(args.gallery_json)
    cases = []
    for group, indices in select_cases(base, rerank, args.limit):
        for i in indices.tolist():
            cases.append(make_case(i, group, base, rerank, cache, query_annos, gallery_annos, args.data_root))

    top1_improved = torch.nonzero(~base["top1"] & rerank["top1"], as_tuple=False).flatten()
    top1_regressed = torch.nonzero(base["top1"] & ~rerank["top1"], as_tuple=False).flatten()
    ap_delta = rerank["ap"] - base["ap"]
    summary = {
        "config": config,
        "metrics": {"baseline": base_metrics, "rerank": rerank_metrics},
        "top1": {
            "improved_count": int(top1_improved.numel()),
            "regressed_count": int(top1_regressed.numel()),
            "unchanged_correct": int((base["top1"] & rerank["top1"]).sum()),
            "unchanged_wrong": int((~base["top1"] & ~rerank["top1"]).sum()),
        },
        "rank": {
            "improved_count": int((rerank["rank"] < base["rank"]).sum()),
            "regressed_count": int((rerank["rank"] > base["rank"]).sum()),
            "unchanged_count": int((rerank["rank"] == base["rank"]).sum()),
        },
        "ap_delta": {
            "mean": float(ap_delta.mean()),
            "median": float(ap_delta.median()),
            "positive_count": int((ap_delta > 0).sum()),
            "negative_count": int((ap_delta < 0).sum()),
            "zero_count": int((ap_delta == 0).sum()),
        },
    }
    payload = {"summary": summary, "cases": cases}
    (out_dir / "case_study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_html(out_dir / "case_study.html", summary, cases)
    print(json.dumps(summary, indent=2))
    print(f"Saved case study to {out_dir}")


if __name__ == "__main__":
    sys.path.insert(0, "src")
    main()
