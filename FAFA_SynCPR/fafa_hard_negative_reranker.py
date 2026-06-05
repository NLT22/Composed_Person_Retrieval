#!/usr/bin/env python
import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys

sys.path.insert(0, "src")

from validate_blip import rank


SHARD_RE = re.compile(r"shard_(\d+)_(\d+)\.pt")


def load_shard(path):
    obj = torch.load(path, map_location="cpu")
    return obj["indices"], obj["qfeats"], obj["target_tokens"]


def shard_start(path):
    match = SHARD_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Bad shard name: {path.name}")
    return int(match.group(1))


def fda_pair_scores(qfeats, target_tokens, fda_k=6):
    token_scores = torch.einsum("bd,gld->bgl", qfeats.float(), target_tokens.float())
    return torch.topk(token_scores, k=fda_k, dim=-1).values.mean(dim=-1)


def pair_features_all(qfeats, target_tokens, fda_k=6):
    scores = torch.einsum("bd,gld->bgl", qfeats.float(), target_tokens.float())
    topk = torch.topk(scores, k=fda_k, dim=-1).values
    base = topk.mean(dim=-1)
    feats = torch.stack(
        [
            base,
            topk.std(dim=-1, unbiased=False),
            scores.mean(dim=-1),
            scores.std(dim=-1, unbiased=False),
            scores.max(dim=-1).values,
            scores.max(dim=-1).values - base,
        ],
        dim=-1,
    )
    return base, feats


def pair_features_candidates(qfeats, candidates, fda_k=6):
    scores = torch.einsum("bd,bcld->bcl", qfeats.float(), candidates.float())
    topk = torch.topk(scores, k=fda_k, dim=-1).values
    base = topk.mean(dim=-1)
    feats = torch.stack(
        [
            base,
            topk.std(dim=-1, unbiased=False),
            scores.mean(dim=-1),
            scores.std(dim=-1, unbiased=False),
            scores.max(dim=-1).values,
            scores.max(dim=-1).values - base,
        ],
        dim=-1,
    )
    return base, feats


class HardNegativeReranker(nn.Module):
    def __init__(self, stat_dim=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.base_scale = nn.Parameter(torch.tensor(20.0))
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def forward_candidates(self, qfeats, candidates, fda_k=6):
        base, feats = pair_features_candidates(qfeats, candidates, fda_k=fda_k)
        delta = self.net(feats).squeeze(-1)
        return self.base_scale * base + self.delta_scale * delta

    def forward_all(self, qfeats, target_tokens, fda_k=6):
        base, feats = pair_features_all(qfeats, target_tokens, fda_k=fda_k)
        delta = self.net(feats).squeeze(-1)
        return self.base_scale * base + self.delta_scale * delta


@torch.no_grad()
def mine(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    shards = sorted(Path(args.syncpr_features).glob("shard_*.pt"))
    if args.max_shards:
        shards = shards[: args.max_shards]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for shard in shards:
        out_path = out_dir / shard.name
        if out_path.exists() and not args.force:
            manifest.append({"shard": shard.name, "path": str(out_path), "skipped": True})
            continue

        indices, qfeats, target_tokens = load_shard(shard)
        n = qfeats.size(0)
        target_gpu = target_tokens.to(device)
        hard_local = torch.empty((n, args.topk), dtype=torch.long)
        hard_scores = torch.empty((n, args.topk), dtype=torch.float16)

        for start in tqdm(range(0, n, args.query_batch_size), desc=f"mine {shard.name}"):
            end = min(start + args.query_batch_size, n)
            q_gpu = qfeats[start:end].to(device)
            scores = fda_pair_scores(q_gpu, target_gpu, args.fda_k)
            diag = torch.arange(start, end, device=device)
            scores[torch.arange(end - start, device=device), diag] = -1e4
            vals, locs = torch.topk(scores, k=args.topk, dim=1)
            hard_local[start:end] = locs.cpu()
            hard_scores[start:end] = vals.cpu().half()

        payload = {
            "source_shard": shard.name,
            "indices": indices,
            "hard_local": hard_local,
            "hard_scores": hard_scores,
            "topk": args.topk,
            "fda_k": args.fda_k,
        }
        torch.save(payload, out_path)
        manifest.append({"shard": shard.name, "path": str(out_path), "samples": n})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved mined hard negatives to {out_dir}")


def iter_hard_batches(feature_dir, mined_dir, batch_size, negatives, shuffle=True):
    feature_shards = {p.name: p for p in Path(feature_dir).glob("shard_*.pt")}
    mined = sorted(Path(mined_dir).glob("shard_*.pt"))
    if shuffle:
        random.shuffle(mined)

    for mined_path in mined:
        feature_path = feature_shards[mined_path.name]
        _, qfeats, target_tokens = load_shard(feature_path)
        hard = torch.load(mined_path, map_location="cpu")["hard_local"]
        n = qfeats.size(0)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for start in range(0, n, batch_size):
            rows = order[start : start + batch_size]
            if rows.numel() < 2:
                continue
            neg_cols = hard[rows, :negatives]
            pos = target_tokens[rows].unsqueeze(1)
            neg = target_tokens[neg_cols]
            candidates = torch.cat([pos, neg], dim=1)
            yield qfeats[rows], candidates


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = HardNegativeReranker(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    step = 0
    iterator = iter_hard_batches(args.syncpr_features, args.mined_dir, args.batch_size, args.negatives)
    progress = tqdm(iterator, total=args.max_steps, desc="hard-rerank")
    for qfeats, candidates in progress:
        qfeats = qfeats.to(device, non_blocking=True)
        candidates = candidates.to(device, non_blocking=True)
        labels = torch.zeros(qfeats.size(0), dtype=torch.long, device=device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model.forward_candidates(qfeats, candidates, fda_k=args.fda_k)
            loss = F.cross_entropy(logits, labels) + args.delta_reg * model.delta_scale.pow(2)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        step += 1
        if step % args.log_every == 0:
            with torch.no_grad():
                r1 = (logits.argmax(dim=1) == labels).float().mean()
                item = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "batch_r1": float(r1.detach().cpu()),
                    "base_scale": float(model.base_scale.detach().cpu()),
                    "delta_scale": float(model.delta_scale.detach().cpu()),
                }
                history.append(item)
                progress.set_postfix(
                    loss=f"{item['loss']:.4f}",
                    r1=f"{item['batch_r1']:.3f}",
                    base=f"{item['base_scale']:.2f}",
                    delta=f"{item['delta_scale']:.3f}",
                )
        if args.max_steps and step >= args.max_steps:
            break

    torch.save(
        {
            "model": model.state_dict(),
            "args": {k: v for k, v in vars(args).items() if k != "func"},
            "history": history,
            "step": step,
        },
        out_dir / "fafa_hard_negative_reranker.pt",
    )
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {out_dir / 'fafa_hard_negative_reranker.pt'}")


def evaluate_scores(scores, qids, gids):
    cmc, mAP, mINP, _ = rank(similarity=scores, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    cmc = cmc.numpy()
    return {
        "R1": float(cmc[0]),
        "R5": float(cmc[4]),
        "R10": float(cmc[9]),
        "mAP": float(mAP.numpy()),
        "mINP": float(mINP.numpy()),
    }


@torch.no_grad()
def eval_itcpr(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = HardNegativeReranker(hidden=ckpt["args"].get("hidden", 128)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cache = torch.load(args.itcpr_cache, map_location="cpu")
    qfeats = cache["qfeats"].to(device)
    gfeats = cache["gfeats"]
    scores = torch.empty((qfeats.size(0), gfeats.size(0)), dtype=torch.float32)
    for start in tqdm(range(0, gfeats.size(0), args.score_batch_size), desc="eval scores"):
        end = min(start + args.score_batch_size, gfeats.size(0))
        block = model.forward_all(qfeats, gfeats[start:end].to(device), fda_k=args.fda_k)
        scores[:, start:end] = block.cpu()

    metrics = evaluate_scores(scores, cache["qids"], cache["gids"])
    out = {
        "checkpoint": str(args.checkpoint),
        "metrics": metrics,
        "base_scale": float(model.base_scale.detach().cpu()),
        "delta_scale": float(model.delta_scale.detach().cpu()),
    }
    print(
        f"R1={metrics['R1']:.3f} R5={metrics['R5']:.3f} "
        f"R10={metrics['R10']:.3f} mAP={metrics['mAP']:.3f} mINP={metrics['mINP']:.3f}"
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved eval results to {out_path}")


@torch.no_grad()
def topk_rerank_itcpr(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = HardNegativeReranker(hidden=ckpt["args"].get("hidden", 128)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cache = torch.load(args.itcpr_cache, map_location="cpu")
    qfeats = cache["qfeats"]
    gfeats = cache["gfeats"]
    q_gpu = qfeats.to(device)
    caption_word_counts = None
    if args.min_caption_words > 0:
        query_annos = json.loads(Path(args.query_json).read_text(encoding="utf-8"))
        if len(query_annos) != qfeats.size(0):
            raise ValueError(f"Query JSON has {len(query_annos)} rows, expected {qfeats.size(0)}")
        caption_word_counts = torch.tensor(
            [len(re.findall(r"[A-Za-z0-9]+", anno.get("caption", ""))) for anno in query_annos],
            dtype=torch.long,
        )

    base_scores = torch.empty((qfeats.size(0), gfeats.size(0)), dtype=torch.float32)
    print("Computing baseline FDA score matrix.")
    for start in tqdm(range(0, gfeats.size(0), args.score_batch_size), desc="base scores"):
        end = min(start + args.score_batch_size, gfeats.size(0))
        base, _ = pair_features_all(q_gpu, gfeats[start:end].to(device), fda_k=args.fda_k)
        base_scores[:, start:end] = (model.base_scale * base).cpu()

    baseline = evaluate_scores(base_scores, cache["qids"], cache["gids"])
    max_k = max(args.topk)
    top_vals, top_idx = torch.topk(base_scores, k=max_k, dim=1)
    margin = top_vals[:, 0] - top_vals[:, 1]

    delta_top = torch.empty_like(top_vals)
    print(f"Computing learned correction for baseline top-{max_k}.")
    for start in tqdm(range(0, qfeats.size(0), args.query_batch_size), desc="topk deltas"):
        end = min(start + args.query_batch_size, qfeats.size(0))
        q_block = qfeats[start:end].to(device)
        idx_block = top_idx[start:end]
        candidates = gfeats[idx_block].to(device)
        _, feats = pair_features_candidates(q_block, candidates, fda_k=args.fda_k)
        delta = model.net(feats).squeeze(-1)
        delta_top[start:end] = (model.delta_scale * delta).cpu()

    results = []
    margin_quantiles = args.margin_quantile or [None]
    for quantile in margin_quantiles:
        if quantile is None:
            row_indices = None
            threshold = None
            applied_queries = qfeats.size(0)
        else:
            threshold = float(torch.quantile(margin, quantile).item())
            row_indices = torch.nonzero(margin <= threshold, as_tuple=False).flatten()
            applied_queries = int(row_indices.numel())
        if caption_word_counts is not None:
            caption_rows = torch.nonzero(caption_word_counts >= args.min_caption_words, as_tuple=False).flatten()
            if row_indices is None:
                row_indices = caption_rows
            else:
                keep = torch.zeros(qfeats.size(0), dtype=torch.bool)
                keep[row_indices] = True
                keep &= caption_word_counts >= args.min_caption_words
                row_indices = torch.nonzero(keep, as_tuple=False).flatten()
            applied_queries = int(row_indices.numel())

        for k in args.topk:
            idx_k = top_idx[:, :k]
            base_k = top_vals[:, :k]
            delta_k = delta_top[:, :k]
            for mult in args.delta_mult:
                scores = base_scores.clone()
                if row_indices is None:
                    scores.scatter_(1, idx_k, base_k + mult * delta_k)
                else:
                    rows = row_indices
                    scores[rows.unsqueeze(1), idx_k[rows]] = base_k[rows] + mult * delta_k[rows]
                metrics = evaluate_scores(scores, cache["qids"], cache["gids"])
                item = {
                    "topk": k,
                    "delta_mult": mult,
                    "margin_quantile": quantile,
                    "margin_threshold": threshold,
                    "min_caption_words": args.min_caption_words,
                    "applied_queries": applied_queries,
                    "metrics": metrics,
                }
                results.append(item)
                gate = "" if quantile is None else f" q={quantile:g} n={applied_queries}"
                print(
                    f"topk={k}{gate} mult={mult:g} R1={metrics['R1']:.3f} "
                    f"R5={metrics['R5']:.3f} R10={metrics['R10']:.3f} mAP={metrics['mAP']:.3f}"
                )

    out = {
        "checkpoint": str(args.checkpoint),
        "itcpr_cache": str(args.itcpr_cache),
        "baseline": baseline,
        "margin_summary": {
            "min": float(margin.min()),
            "median": float(margin.median()),
            "max": float(margin.max()),
        },
        "min_caption_words": args.min_caption_words,
        "base_scale": float(model.base_scale.detach().cpu()),
        "delta_scale": float(model.delta_scale.detach().cpu()),
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved top-k rerank sweep to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    mine_p = sub.add_parser("mine")
    mine_p.add_argument("--syncpr-features", default="experiments/fafa_pretrained/syncpr_features")
    mine_p.add_argument("--out-dir", default="experiments/fafa_hard_negatives")
    mine_p.add_argument("--device", default="cuda")
    mine_p.add_argument("--max-shards", type=int)
    mine_p.add_argument("--topk", type=int, default=32)
    mine_p.add_argument("--fda-k", type=int, default=6)
    mine_p.add_argument("--query-batch-size", type=int, default=128)
    mine_p.add_argument("--force", action="store_true")
    mine_p.set_defaults(func=mine)

    train_p = sub.add_parser("train")
    train_p.add_argument("--syncpr-features", default="experiments/fafa_pretrained/syncpr_features")
    train_p.add_argument("--mined-dir", default="experiments/fafa_hard_negatives")
    train_p.add_argument("--out-dir", default="experiments/fafa_hard_negative_reranker")
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--batch-size", type=int, default=256)
    train_p.add_argument("--negatives", type=int, default=16)
    train_p.add_argument("--hidden", type=int, default=128)
    train_p.add_argument("--max-steps", type=int, default=1000)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--grad-clip", type=float, default=1.0)
    train_p.add_argument("--delta-reg", type=float, default=1e-3)
    train_p.add_argument("--fda-k", type=int, default=6)
    train_p.add_argument("--log-every", type=int, default=20)
    train_p.set_defaults(func=train)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", default="experiments/fafa_hard_negative_reranker/fafa_hard_negative_reranker.pt")
    eval_p.add_argument("--itcpr-cache", default="experiments/fafa_pretrained/itcpr_fafa_features.pt")
    eval_p.add_argument("--output", default="experiments/fafa_hard_negative_reranker/itcpr_eval.json")
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--fda-k", type=int, default=6)
    eval_p.add_argument("--score-batch-size", type=int, default=256)
    eval_p.set_defaults(func=eval_itcpr)

    topk_p = sub.add_parser("topk-rerank")
    topk_p.add_argument("--checkpoint", default="experiments/fafa_hard_negative_reranker/fafa_hard_negative_reranker.pt")
    topk_p.add_argument("--itcpr-cache", default="experiments/fafa_pretrained/itcpr_fafa_features.pt")
    topk_p.add_argument("--output", default="experiments/fafa_hard_negative_reranker/itcpr_topk_rerank.json")
    topk_p.add_argument("--device", default="cuda")
    topk_p.add_argument("--fda-k", type=int, default=6)
    topk_p.add_argument("--score-batch-size", type=int, default=256)
    topk_p.add_argument("--query-batch-size", type=int, default=32)
    topk_p.add_argument("--topk", nargs="+", type=int, default=[10, 50, 100, 500, 1000])
    topk_p.add_argument("--query-json", default="../data/query.json")
    topk_p.add_argument("--min-caption-words", type=int, default=0)
    topk_p.add_argument(
        "--delta-mult",
        nargs="+",
        type=float,
        default=[-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1],
    )
    topk_p.add_argument(
        "--margin-quantile",
        nargs="+",
        type=float,
        help="Optionally apply reranking only to queries whose baseline top1-top2 margin is below these quantiles.",
    )
    topk_p.set_defaults(func=topk_rerank_itcpr)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
