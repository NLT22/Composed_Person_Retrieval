#!/usr/bin/env python
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys

sys.path.insert(0, "src")

from validate_blip import rank


def soft_clamp_tanh(x, min_val=-10.0, max_val=-6.0):
    center = (max_val + min_val) * 0.5
    half_range = (max_val - min_val) * 0.5
    return center + half_range * torch.tanh((x - center) / half_range)


def pairwise_prob_distance(query, target):
    q_mu, q_var = query["mean"], query["var"]
    t_mu, t_var = target["mean"], target["var"]

    mu_dist = torch.cdist(q_mu, t_mu, p=2).pow(2)
    q_std = torch.exp(q_var * 0.5)
    t_std = torch.exp(t_var * 0.5)
    sigma_dist = torch.cdist(q_std, t_std, p=2).pow(2)
    cross = 2 * q_mu.size(-1) * (q_std.mean(dim=-1, keepdim=True) * t_std.mean(dim=-1).unsqueeze(0))
    return mu_dist + sigma_dist + cross


class FAFACoPEHead(nn.Module):
    def __init__(self, dim=256, hidden=512, log_var_min=-10.0, log_var_max=-6.0):
        super().__init__()
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        self.query_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.target_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.target_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.query_var = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, dim),
        )
        self.target_var = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.shift = nn.Parameter(torch.tensor(0.0))

    def encode_query(self, qfeats):
        residual = qfeats.float()
        mean = F.normalize(residual + self.query_proj(residual), dim=-1)
        var = soft_clamp_tanh(self.query_var(residual), self.log_var_min, self.log_var_max)
        return {"mean": mean, "var": var}

    def encode_target(self, target_tokens):
        tokens = target_tokens.float()
        weights = torch.softmax(self.target_score(tokens).squeeze(-1), dim=-1)
        pooled = torch.einsum("bl,bld->bd", weights, tokens)
        mean = F.normalize(pooled + self.target_proj(pooled), dim=-1)
        stats = torch.cat([tokens.mean(dim=1), tokens.std(dim=1, unbiased=False)], dim=-1)
        var = soft_clamp_tanh(self.target_var(stats), self.log_var_min, self.log_var_max)
        return {"mean": mean, "var": var}

    def logits(self, qfeats, target_tokens):
        query = self.encode_query(qfeats)
        target = self.encode_target(target_tokens)
        dist = pairwise_prob_distance(query, target)
        scale = F.softplus(self.logit_scale) + 1e-6
        return self.shift - scale * dist


def load_shard(path):
    obj = torch.load(path, map_location="cpu")
    return obj["qfeats"], obj["target_tokens"]


def iter_batches(shards, batch_size, steps_per_shard=None, shuffle=True):
    order = list(shards)
    if shuffle:
        random.shuffle(order)

    for shard in order:
        qfeats, target_tokens = load_shard(shard)
        n = qfeats.size(0)
        indices = torch.randperm(n) if shuffle else torch.arange(n)
        max_steps = math.ceil(n / batch_size) if steps_per_shard is None else steps_per_shard
        for step in range(max_steps):
            start = (step * batch_size) % n
            batch_idx = indices[start : start + batch_size]
            if batch_idx.numel() < batch_size and n >= batch_size:
                extra = indices[: batch_size - batch_idx.numel()]
                batch_idx = torch.cat([batch_idx, extra], dim=0)
            yield qfeats[batch_idx], target_tokens[batch_idx]


def contrastive_loss(logits):
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    shards = sorted(Path(args.syncpr_features).glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {args.syncpr_features}")

    model = FAFACoPEHead(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0

    for epoch in range(args.epochs):
        progress = tqdm(
            iter_batches(shards, args.batch_size, shuffle=True),
            total=args.max_steps if args.max_steps else None,
            desc=f"epoch {epoch + 1}",
        )
        for qfeats, target_tokens in progress:
            qfeats = qfeats.to(device, non_blocking=True)
            target_tokens = target_tokens.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model.logits(qfeats, target_tokens)
                loss = contrastive_loss(logits)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % args.log_every == 0:
                with torch.no_grad():
                    acc = (logits.argmax(dim=1) == torch.arange(logits.size(0), device=device)).float().mean()
                    scale = F.softplus(model.logit_scale).item()
                item = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach().cpu()),
                    "batch_r1": float(acc.detach().cpu()),
                    "scale": scale,
                }
                history.append(item)
                progress.set_postfix(loss=f"{item['loss']:.4f}", r1=f"{item['batch_r1']:.3f}", scale=f"{scale:.2f}")

            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    ckpt = {
        "model": model.state_dict(),
        "args": {k: v for k, v in vars(args).items() if k != "func"},
        "history": history,
        "step": global_step,
    }
    torch.save(ckpt, out_dir / "fafa_cope_head.pt")
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {out_dir / 'fafa_cope_head.pt'}")


def compute_fda_scores(qfeats, gfeats, device, fda_k=6, batch_size=256):
    qfeats = qfeats.float()
    gfeats_t = gfeats.permute(0, 2, 1).contiguous().float()
    scores = torch.empty((qfeats.size(0), gfeats.size(0)), dtype=torch.float32)
    for qs in tqdm(range(0, qfeats.size(0), batch_size), desc="FDA scores"):
        qe = min(qs + batch_size, qfeats.size(0))
        qb = qfeats[qs:qe].to(device).unsqueeze(1).unsqueeze(1)
        for gs in range(0, gfeats_t.size(0), batch_size):
            ge = min(gs + batch_size, gfeats_t.size(0))
            gb = gfeats_t[gs:ge].to(device)
            token_scores = torch.matmul(qb, gb).squeeze(2)
            scores[qs:qe, gs:ge] = torch.topk(token_scores, k=fda_k, dim=-1).values.mean(-1).cpu()
    return scores


@torch.no_grad()
def compute_prob_scores(model, qfeats, gfeats, device, batch_size=256):
    q_parts = []
    for start in range(0, qfeats.size(0), batch_size):
        enc = model.encode_query(qfeats[start : start + batch_size].to(device))
        q_parts.append({k: v.cpu() for k, v in enc.items()})
    qdist = {k: torch.cat([x[k] for x in q_parts], dim=0) for k in q_parts[0]}

    t_parts = []
    for start in tqdm(range(0, gfeats.size(0), batch_size), desc="target distributions"):
        enc = model.encode_target(gfeats[start : start + batch_size].to(device))
        t_parts.append({k: v.cpu() for k, v in enc.items()})
    tdist = {k: torch.cat([x[k] for x in t_parts], dim=0) for k in t_parts[0]}

    scores = torch.empty((qfeats.size(0), gfeats.size(0)), dtype=torch.float32)
    scale = F.softplus(model.logit_scale).detach().cpu() + 1e-6
    shift = model.shift.detach().cpu()
    qdist_gpu = {k: v.to(device) for k, v in qdist.items()}
    for start in tqdm(range(0, gfeats.size(0), batch_size), desc="prob scores"):
        end = min(start + batch_size, gfeats.size(0))
        tdist_gpu = {k: v[start:end].to(device) for k, v in tdist.items()}
        dist = pairwise_prob_distance(qdist_gpu, tdist_gpu)
        scores[:, start:end] = (shift.to(device) - scale.to(device) * dist).cpu()
    return scores


def zscore_rows(x):
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


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


def eval_itcpr(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cache = torch.load(args.itcpr_cache, map_location="cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FAFACoPEHead(hidden=ckpt["args"].get("hidden", 512)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    qfeats = cache["qfeats"]
    gfeats = cache["gfeats"]
    qids = cache["qids"]
    gids = cache["gids"]

    prob_scores = compute_prob_scores(model, qfeats, gfeats, device, args.score_batch_size)
    fda_scores = compute_fda_scores(qfeats, gfeats, device, args.fda_k, args.score_batch_size)

    results = []
    prob_metrics = evaluate_scores(prob_scores, qids, gids)
    prob_metrics["alpha"] = 1.0
    prob_metrics["score"] = "prob_only"
    results.append(prob_metrics)

    fda_metrics = evaluate_scores(fda_scores, qids, gids)
    fda_metrics["alpha"] = 0.0
    fda_metrics["score"] = "fda_only"
    results.append(fda_metrics)

    fda_z = zscore_rows(fda_scores)
    prob_z = zscore_rows(prob_scores)
    for alpha in args.alphas:
        mixed = (1 - alpha) * fda_z + alpha * prob_z
        metrics = evaluate_scores(mixed, qids, gids)
        metrics["alpha"] = alpha
        metrics["score"] = "zscore_mix"
        results.append(metrics)
        print(
            f"alpha={alpha:.3g} R1={metrics['R1']:.3f} R5={metrics['R5']:.3f} "
            f"R10={metrics['R10']:.3f} mAP={metrics['mAP']:.3f}"
        )

    out = {
        "checkpoint": str(args.checkpoint),
        "itcpr_cache": str(args.itcpr_cache),
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved eval results to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--syncpr-features", default="experiments/fafa_pretrained/syncpr_features")
    train_p.add_argument("--out-dir", default="experiments/fafa_cope_head")
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--batch-size", type=int, default=512)
    train_p.add_argument("--hidden", type=int, default=512)
    train_p.add_argument("--epochs", type=int, default=1)
    train_p.add_argument("--max-steps", type=int)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--grad-clip", type=float, default=1.0)
    train_p.add_argument("--log-every", type=int, default=20)
    train_p.set_defaults(func=train)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", default="experiments/fafa_cope_head/fafa_cope_head.pt")
    eval_p.add_argument("--itcpr-cache", default="experiments/fafa_pretrained/itcpr_fafa_features.pt")
    eval_p.add_argument("--output", default="experiments/fafa_cope_head/itcpr_eval.json")
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--score-batch-size", type=int, default=256)
    eval_p.add_argument("--fda-k", type=int, default=6)
    eval_p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    eval_p.set_defaults(func=eval_itcpr)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
