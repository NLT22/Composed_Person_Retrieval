#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "src")

from data_utils import GalleryDataset, ITCPRDataset, QueryDataset, squarepad_transform_test
from lavis.models import load_model_and_preprocess
from utils import collate_fn
from validate_blip import rank


def load_fafa(checkpoint_path, device):
    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_fafa_cpr",
        model_type="pretrain",
        model_path=str(checkpoint_path),
        is_eval=True,
        device=device,
    )
    model.eval()
    model.fda_k = 6
    model.use_soft = True
    return model, txt_processors


def build_itcpr(root):
    dataset = ITCPRDataset(root=root)
    preprocess = squarepad_transform_test(224)
    query = dataset.query
    gallery = dataset.gallery
    query_set = QueryDataset(
        query["instance_ids"],
        query["img_paths"],
        query["captions"],
        preprocess,
    )
    gallery_set = GalleryDataset(
        gallery["instance_ids"],
        gallery["img_paths"],
        preprocess,
    )
    return query_set, gallery_set


@torch.no_grad()
def extract_features(model, txt_processors, query_set, gallery_set, device, cache_path):
    gallery_loader = DataLoader(
        gallery_set, batch_size=64, num_workers=2, pin_memory=True, collate_fn=collate_fn
    )
    query_loader = DataLoader(
        query_set, batch_size=64, num_workers=2, pin_memory=True, collate_fn=collate_fn
    )

    gids, gallery_features = [], []
    print("Extracting gallery token features.")
    for iid, images in tqdm(gallery_loader):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            feats, _ = model.extract_target_features(images.half() if device.type == "cuda" else images)
        gids.append(iid.view(-1).cpu())
        gallery_features.append(feats.detach().cpu().half())

    qids, query_features = [], []
    print("Extracting query features.")
    for iid, images, captions in tqdm(query_loader):
        images = images.to(device, non_blocking=True)
        captions = np.array(captions).T.flatten().tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            feats = model.extract_features(
                {"image": images.half() if device.type == "cuda" else images, "text_input": captions}
            ).multimodal_embeds
        qids.append(iid.view(-1).cpu())
        query_features.append(feats.detach().cpu().half())

    cache = {
        "qids": torch.cat(qids, 0),
        "gids": torch.cat(gids, 0),
        "qfeats": torch.cat(query_features, 0),
        "gfeats": torch.cat(gallery_features, 0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved feature cache to {cache_path}")
    return cache


def compute_score_and_penalty(cache, device, fda_k=6, batch_size=256):
    qfeats = cache["qfeats"]
    gfeats = cache["gfeats"]
    num_q = qfeats.size(0)
    num_g = gfeats.size(0)

    scores = torch.empty((num_q, num_g), dtype=torch.float32)
    penalties = torch.empty((num_q, num_g), dtype=torch.float32)

    gfeats_t = gfeats.permute(0, 2, 1).contiguous()
    print("Computing baseline FDA scores and token-disagreement penalties.")
    for qs in tqdm(range(0, num_q, batch_size)):
        qe = min(qs + batch_size, num_q)
        qb = qfeats[qs:qe].to(device).float().unsqueeze(1).unsqueeze(1)
        for gs in range(0, num_g, batch_size):
            ge = min(gs + batch_size, num_g)
            gb = gfeats_t[gs:ge].to(device).float()
            token_scores = torch.matmul(qb, gb).squeeze(2)
            topk_scores, _ = torch.topk(token_scores, k=fda_k, dim=-1)
            block_scores = topk_scores.mean(-1)
            block_penalties = topk_scores.std(-1, unbiased=False)
            scores[qs:qe, gs:ge] = block_scores.cpu()
            penalties[qs:qe, gs:ge] = block_penalties.cpu()
    return scores, penalties


def evaluate(similarity, qids, gids):
    cmc, mAP, mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    cmc = cmc.numpy()
    return {
        "R1": float(cmc[0]),
        "R5": float(cmc[4]),
        "R10": float(cmc[9]),
        "mAP": float(mAP.numpy()),
        "mINP": float(mINP.numpy()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--itcpr-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-path", default="experiments/fafa_pretrained/itcpr_fafa_features.pt")
    parser.add_argument("--results-path", default="experiments/fafa_pretrained/rerank_heuristic_results.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cache_path = Path(args.cache_path)
    if cache_path.exists() and not args.force_extract:
        print(f"Loading feature cache from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
    else:
        model, txt_processors = load_fafa(args.checkpoint, device)
        query_set, gallery_set = build_itcpr(args.itcpr_root)
        cache = extract_features(model, txt_processors, query_set, gallery_set, device, cache_path)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores, penalties = compute_score_and_penalty(cache, device)
    qids = cache["qids"]
    gids = cache["gids"]

    results = []
    for lam in args.lambdas:
        sim = scores - lam * penalties
        metrics = evaluate(sim, qids, gids)
        metrics["lambda"] = lam
        results.append(metrics)
        print(
            f"lambda={lam:.3g} R1={metrics['R1']:.3f} R5={metrics['R5']:.3f} "
            f"R10={metrics['R10']:.3f} mAP={metrics['mAP']:.3f} mINP={metrics['mINP']:.3f}"
        )

    out = {
        "method": "FAFA FDA baseline plus token-disagreement penalty",
        "score": "mean(topk_token_scores) - lambda * std(topk_token_scores)",
        "results": results,
    }
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved rerank results to {results_path}")


if __name__ == "__main__":
    main()
