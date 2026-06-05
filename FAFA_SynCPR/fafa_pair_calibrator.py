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


def load_shard(path):
    obj = torch.load(path, map_location="cpu")
    return obj["qfeats"], obj["target_tokens"]


def iter_batches(shards, batch_size, shuffle=True):
    order = list(shards)
    if shuffle:
        random.shuffle(order)
    for shard in order:
        qfeats, target_tokens = load_shard(shard)
        n = qfeats.size(0)
        idx = torch.randperm(n) if shuffle else torch.arange(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            if batch_idx.numel() < 2:
                continue
            yield qfeats[batch_idx], target_tokens[batch_idx]


def pair_stats(qfeats, target_tokens, fda_k=6):
    q = qfeats.float()
    t = target_tokens.float()
    scores = torch.einsum("bd,gld->bgl", q, t)
    topk = torch.topk(scores, k=fda_k, dim=-1).values
    topk_mean = topk.mean(dim=-1)
    topk_std = topk.std(dim=-1, unbiased=False)
    token_mean = scores.mean(dim=-1)
    token_std = scores.std(dim=-1, unbiased=False)
    token_max = scores.max(dim=-1).values
    margin = token_max - topk_mean
    stats = torch.stack([topk_mean, topk_std, token_mean, token_std, token_max, margin], dim=-1)
    return topk_mean, stats


class PairCalibrator(nn.Module):
    def __init__(self, stat_dim=6, hidden=64):
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

    def forward(self, qfeats, target_tokens, fda_k=6):
        base, stats = pair_stats(qfeats, target_tokens, fda_k=fda_k)
        delta = self.net(stats).squeeze(-1)
        return self.base_scale * base + self.delta_scale * delta


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    shards = sorted(Path(args.syncpr_features).glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {args.syncpr_features}")

    model = PairCalibrator(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    step = 0
    progress = tqdm(iter_batches(shards, args.batch_size), total=args.max_steps, desc="pair-calib")
    for qfeats, target_tokens in progress:
        qfeats = qfeats.to(device, non_blocking=True)
        target_tokens = target_tokens.to(device, non_blocking=True)
        labels = torch.arange(qfeats.size(0), device=device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(qfeats, target_tokens, fda_k=args.fda_k)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            delta_reg = model.delta_scale.pow(2) * args.delta_reg
            loss = loss + delta_reg

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
        out_dir / "fafa_pair_calibrator.pt",
    )
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {out_dir / 'fafa_pair_calibrator.pt'}")


@torch.no_grad()
def score_matrix(model, qfeats, gfeats, device, fda_k=6, batch_size=256):
    scores = torch.empty((qfeats.size(0), gfeats.size(0)), dtype=torch.float32)
    q_gpu = qfeats.to(device)
    for start in tqdm(range(0, gfeats.size(0), batch_size), desc="pair scores"):
        end = min(start + batch_size, gfeats.size(0))
        block = model(q_gpu, gfeats[start:end].to(device), fda_k=fda_k)
        scores[:, start:end] = block.cpu()
    return scores


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
    model = PairCalibrator(hidden=ckpt["args"].get("hidden", 64)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scores = score_matrix(model, cache["qfeats"], cache["gfeats"], device, args.fda_k, args.score_batch_size)
    metrics = evaluate_scores(scores, cache["qids"], cache["gids"])
    out = {
        "checkpoint": str(args.checkpoint),
        "itcpr_cache": str(args.itcpr_cache),
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


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--syncpr-features", default="experiments/fafa_pretrained/syncpr_features")
    train_p.add_argument("--out-dir", default="experiments/fafa_pair_calibrator")
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--batch-size", type=int, default=512)
    train_p.add_argument("--hidden", type=int, default=64)
    train_p.add_argument("--max-steps", type=int, default=1000)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--grad-clip", type=float, default=1.0)
    train_p.add_argument("--delta-reg", type=float, default=1e-3)
    train_p.add_argument("--fda-k", type=int, default=6)
    train_p.add_argument("--log-every", type=int, default=20)
    train_p.set_defaults(func=train)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", default="experiments/fafa_pair_calibrator/fafa_pair_calibrator.pt")
    eval_p.add_argument("--itcpr-cache", default="experiments/fafa_pretrained/itcpr_fafa_features.pt")
    eval_p.add_argument("--output", default="experiments/fafa_pair_calibrator/itcpr_eval.json")
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--fda-k", type=int, default=6)
    eval_p.add_argument("--score-batch-size", type=int, default=256)
    eval_p.set_defaults(func=eval_itcpr)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
