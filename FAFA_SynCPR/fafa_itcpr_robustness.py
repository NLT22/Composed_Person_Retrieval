#!/usr/bin/env python
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
import skimage as ski
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image
from io import BytesIO

sys.path.insert(0, "src")

from data_utils import QueryDataset, read_image, squarepad_transform_test
from fafa_cache_and_rerank import load_fafa
from fafa_hard_negative_reranker import (
    HardNegativeReranker,
    evaluate_scores,
    pair_features_all,
    pair_features_candidates,
)
from utils import collate_fn


WORD_RE = re.compile(r"[A-Za-z0-9]+")
QWERTY_NEIGHBORS = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wsdr",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujko",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}
CORRUPTION_SEED_OFFSET = {
    "clean": 0,
    "character_filter": 101,
    "RemoveChar_filter": 211,
    "remove_space_filter": 307,
    "qwerty_filter": 401,
    "repetition_filter": 503,
}
IMAGE_CORRUPTION_SEED_OFFSET = {
    "clean": 0,
    "gaussian_noise_filter": 601,
    "motion_blur_filter": 701,
    "brightness_filter": 809,
    "contrast_filter": 907,
    "pixelate_filter": 1009,
    "jpeg_compression": 1103,
}


def _char_positions(text, include_space=False):
    return [
        i
        for i, ch in enumerate(text)
        if (ch.isspace() and include_space) or (not ch.isspace() and ch.isprintable())
    ]


def _edit_count(text, severity):
    return max(1, round(len(text) * {1: 0.02, 2: 0.04, 3: 0.08, 4: 0.12, 5: 0.16}[severity]))


def corrupt_caption(text, corruption, severity, rng):
    if corruption == "clean":
        return text

    chars = list(text)
    edits = _edit_count(text, severity)

    if corruption == "character_filter":
        for _ in range(edits):
            positions = _char_positions("".join(chars))
            if len(positions) < 2:
                break
            i = rng.choice(positions[:-1])
            j = i + 1
            chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)

    if corruption == "RemoveChar_filter":
        for _ in range(edits):
            positions = _char_positions("".join(chars))
            if not positions:
                break
            del chars[rng.choice(positions)]
        return "".join(chars)

    if corruption == "remove_space_filter":
        for _ in range(edits):
            positions = [i for i, ch in enumerate(chars) if ch.isspace()]
            if not positions:
                break
            del chars[rng.choice(positions)]
        return "".join(chars)

    if corruption == "qwerty_filter":
        for _ in range(edits):
            positions = [i for i, ch in enumerate(chars) if ch.lower() in QWERTY_NEIGHBORS]
            if not positions:
                break
            i = rng.choice(positions)
            repl = rng.choice(QWERTY_NEIGHBORS[chars[i].lower()])
            chars[i] = repl.upper() if chars[i].isupper() else repl
        return "".join(chars)

    if corruption == "repetition_filter":
        words = text.split()
        if not words:
            return text
        for _ in range(max(1, severity)):
            i = rng.randrange(len(words))
            words.insert(i, words[i])
        return " ".join(words)

    raise ValueError(f"Unknown corruption: {corruption}")


def corrupt_image(image, corruption, severity, rng):
    if corruption == "clean":
        return image

    x = np.array(image.convert("RGB"))
    np_rng = np.random.default_rng(rng.randrange(2**32))

    if corruption == "gaussian_noise_filter":
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        y = np.clip(x / 255.0 + np_rng.normal(size=x.shape, scale=c), 0, 1) * 255
        return Image.fromarray(y.astype(np.uint8))

    if corruption == "motion_blur_filter":
        size = [5, 7, 9, 11, 13][severity - 1]
        angle = rng.uniform(-45, 45)
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[(size - 1) // 2, :] = 1.0
        kernel = cv2.warpAffine(
            kernel,
            cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0),
            (size, size),
        )
        kernel = kernel / np.sum(kernel)
        y = cv2.filter2D(x, -1, kernel)
        return Image.fromarray(y.astype(np.uint8))

    if corruption == "brightness_filter":
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        y = x / 255.0
        y = ski.color.rgb2hsv(y)
        y[:, :, 2] = np.clip(y[:, :, 2] + c, 0, 1)
        y = np.clip(ski.color.hsv2rgb(y), 0, 1) * 255
        return Image.fromarray(y.astype(np.uint8))

    if corruption == "contrast_filter":
        c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
        y = x / 255.0
        mean = np.mean(y, axis=(0, 1), keepdims=True)
        y = np.clip((y - mean) * c + mean, 0, 1) * 255
        return Image.fromarray(y.astype(np.uint8))

    if corruption == "pixelate_filter":
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        h, w = x.shape[:2]
        small = image.resize((max(1, int(w * c)), max(1, int(h * c))), Image.Resampling.BOX)
        return small.resize((w, h), Image.Resampling.BOX)

    if corruption == "jpeg_compression":
        c = [25, 18, 15, 10, 7][severity - 1]
        output = BytesIO()
        image.save(output, "JPEG", quality=c)
        output.seek(0)
        return Image.open(output).convert("RGB")

    raise ValueError(f"Unknown image corruption: {corruption}")


class RobustQueryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        instance_ids,
        img_paths,
        captions,
        preprocess,
        image_corruption="clean",
        image_severity=0,
        seed=42,
    ):
        self.instance_ids = instance_ids
        self.img_paths = img_paths
        self.captions = captions
        self.preprocess = preprocess
        self.image_corruption = image_corruption
        self.image_severity = image_severity
        self.seed = seed

    def __len__(self):
        return len(self.instance_ids)

    def __getitem__(self, index):
        image = read_image(self.img_paths[index])
        if self.image_corruption != "clean":
            rng = random.Random(
                self.seed
                + index * 1009
                + self.image_severity * 9176
                + IMAGE_CORRUPTION_SEED_OFFSET[self.image_corruption]
            )
            image = corrupt_image(image, self.image_corruption, self.image_severity, rng)
        image = self.preprocess(image)
        return self.instance_ids[index], image, self.captions[index]


def build_corrupted_query_set(root, corruption, severity, seed, max_queries=None):
    root = Path(root)
    query_annos = json.loads((root / "query.json").read_text(encoding="utf-8"))
    preprocess = squarepad_transform_test(224)
    instance_ids = [int(anno["instance_id"]) for anno in query_annos]
    img_paths = [str(root / anno["file_path"]) for anno in query_annos]
    original = [anno["caption"] for anno in query_annos]
    captions = []
    for idx, caption in enumerate(original):
        rng = random.Random(seed + idx * 1009 + severity * 9176 + CORRUPTION_SEED_OFFSET[corruption])
        captions.append(corrupt_caption(caption, corruption, severity, rng))

    query_set = QueryDataset(
        instance_ids,
        img_paths,
        captions,
        preprocess,
    )
    if max_queries is not None:
        query_set = Subset(query_set, range(max_queries))
    return query_set, captions[:max_queries], original[:max_queries]


def build_image_corrupted_query_set(root, corruption, severity, seed, max_queries=None):
    root = Path(root)
    query_annos = json.loads((root / "query.json").read_text(encoding="utf-8"))
    preprocess = squarepad_transform_test(224)
    instance_ids = [int(anno["instance_id"]) for anno in query_annos]
    img_paths = [str(root / anno["file_path"]) for anno in query_annos]
    captions = [anno["caption"] for anno in query_annos]
    query_set = RobustQueryDataset(
        instance_ids,
        img_paths,
        captions,
        preprocess,
        image_corruption=corruption,
        image_severity=severity,
        seed=seed,
    )
    if max_queries is not None:
        query_set = Subset(query_set, range(max_queries))
    return query_set, captions[:max_queries]


@torch.no_grad()
def extract_query_features(model, txt_processors, query_set, device, batch_size, num_workers):
    loader = DataLoader(
        query_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    qids, qfeats = [], []
    for iid, images, captions in tqdm(loader, desc="query features"):
        images = images.to(device, non_blocking=True)
        captions = np.array(captions).T.flatten().tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            feats = model.extract_features(
                {"image": images.half() if device.type == "cuda" else images, "text_input": captions}
            ).multimodal_embeds
        qids.append(iid.view(-1).cpu())
        qfeats.append(feats.detach().cpu().half())
    return torch.cat(qids, 0), torch.cat(qfeats, 0)


@torch.no_grad()
def score_baseline(qfeats, gfeats, device, fda_k, score_batch_size, base_scale=1.0):
    scores = torch.empty((qfeats.size(0), gfeats.size(0)), dtype=torch.float32)
    q_gpu = qfeats.to(device)
    for start in tqdm(range(0, gfeats.size(0), score_batch_size), desc="base scores"):
        end = min(start + score_batch_size, gfeats.size(0))
        base, _ = pair_features_all(q_gpu, gfeats[start:end].to(device), fda_k=fda_k)
        scores[:, start:end] = (base_scale * base).cpu()
    return scores


@torch.no_grad()
def apply_gated_rerank(
    base_scores,
    qfeats,
    gfeats,
    reranker,
    captions,
    device,
    fda_k,
    topk,
    delta_mult,
    margin_quantile,
    min_caption_words,
    query_batch_size,
):
    top_vals, top_idx = torch.topk(base_scores, k=topk, dim=1)
    margin = top_vals[:, 0] - top_vals[:, 1]
    threshold = float(torch.quantile(margin, margin_quantile).item())
    word_counts = torch.tensor([len(WORD_RE.findall(caption)) for caption in captions], dtype=torch.long)
    rows = torch.nonzero((margin <= threshold) & (word_counts >= min_caption_words), as_tuple=False).flatten()
    if rows.numel() == 0:
        return base_scores.clone(), {
            "applied_queries": 0,
            "margin_threshold": threshold,
            "margin_quantile": margin_quantile,
            "min_caption_words": min_caption_words,
        }

    delta_top = torch.empty_like(top_vals)
    for start in tqdm(range(0, qfeats.size(0), query_batch_size), desc="topk deltas"):
        end = min(start + query_batch_size, qfeats.size(0))
        q_block = qfeats[start:end].to(device)
        idx_block = top_idx[start:end]
        candidates = gfeats[idx_block].to(device)
        _, feats = pair_features_candidates(q_block, candidates, fda_k=fda_k)
        delta = reranker.net(feats).squeeze(-1)
        delta_top[start:end] = (reranker.delta_scale * delta).cpu()

    scores = base_scores.clone()
    scores[rows.unsqueeze(1), top_idx[rows]] = top_vals[rows] + delta_mult * delta_top[rows]
    return scores, {
        "applied_queries": int(rows.numel()),
        "margin_threshold": threshold,
        "margin_quantile": margin_quantile,
        "min_caption_words": min_caption_words,
        "topk": topk,
        "delta_mult": delta_mult,
    }


def load_reranker(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = HardNegativeReranker(hidden=ckpt["args"].get("hidden", 128)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def write_report(path, payload):
    domain = payload.get("domain", "text")
    lines = [
        f"# ITCPR-C {domain.title()} Robustness Pilot",
        "",
        "This pilot adapts the robustness benchmark idea to composed person retrieval.",
        f"It keeps the gallery image features clean and corrupts only query {domain}, so it is cheap enough to run without retraining FAFA or CoPE.",
        "",
        "## Research Framing",
        "",
        "- Deep-research framing: test whether robustness benchmarking gives a cleaner contribution than forcing a direct FAFA-CoPE model fusion.",
        "- Academic-pipeline framing: keep the experiment reproducible with fixed corruptions, fixed seed, fixed checkpoint, JSON output, and a markdown report.",
        "- Karpathy-guidelines framing: run the smallest meaningful benchmark first before adding training-heavy components.",
        "",
        "## Setup",
        "",
        f"- ITCPR root: `{payload['itcpr_root']}`",
        f"- FAFA checkpoint: `{payload['checkpoint']}`",
        f"- Gallery cache: `{payload['itcpr_cache']}`",
        f"- Queries evaluated: `{payload['num_queries']}`",
        f"- Gallery size: `{payload['num_gallery']}`",
        "",
        "## Results",
        "",
        "| corruption | severity | method | R1 | R5 | R10 | mAP | RR@1 | RR@mAP | applied |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    clean = payload["clean_baseline"]
    for item in payload["results"]:
        metrics = item["metrics"]
        rr1 = metrics["R1"] / clean["R1"] if clean["R1"] else 0.0
        rrmap = metrics["mAP"] / clean["mAP"] if clean["mAP"] else 0.0
        lines.append(
            f"| {item['corruption']} | {item['severity']} | {item['method']} | "
            f"{metrics['R1']:.3f} | {metrics['R5']:.3f} | {metrics['R10']:.3f} | {metrics['mAP']:.3f} | "
            f"{rr1:.3f} | {rrmap:.3f} | {item.get('applied_queries', '')} |"
        )

    clean_r1 = clean["R1"]
    clean_map = clean["mAP"]
    fafa_rows = [
        item
        for item in payload["results"]
        if item["method"] == "FAFA" and item["corruption"] != "clean"
    ]
    worst_r1 = min(fafa_rows, key=lambda item: item["metrics"]["R1"])
    best_r1 = max(fafa_rows, key=lambda item: item["metrics"]["R1"])
    rerank_pairs = []
    by_key = {
        (item["corruption"], item["severity"], item["method"]): item
        for item in payload["results"]
    }
    for key, rerank in by_key.items():
        corruption, severity, method = key
        if method != "FAFA+SynCPR-gated-rerank":
            continue
        base = by_key[(corruption, severity, "FAFA")]
        rerank_pairs.append(
            (
                corruption,
                severity,
                rerank["metrics"]["R1"] - base["metrics"]["R1"],
                rerank["metrics"]["mAP"] - base["metrics"]["mAP"],
            )
        )
    best_rerank = max(rerank_pairs, key=lambda x: x[2])
    worst_rerank = min(rerank_pairs, key=lambda x: x[2])
    lines.extend(
        [
            "",
            "## Key Takeaways",
            "",
            f"- Clean FAFA is R1={clean_r1:.3f}, mAP={clean_map:.3f}.",
            f"- Worst tested corruption is `{worst_r1['corruption']}` severity {worst_r1['severity']}: "
            f"R1 drops by {worst_r1['metrics']['R1'] - clean_r1:.3f}, mAP drops by {worst_r1['metrics']['mAP'] - clean_map:.3f}.",
            f"- Mildest tested corruption is `{best_r1['corruption']}` severity {best_r1['severity']}: "
            f"R1 drops by {best_r1['metrics']['R1'] - clean_r1:.3f}, mAP drops by {best_r1['metrics']['mAP'] - clean_map:.3f}.",
            f"- Best gated-rerank movement is `{best_rerank[0]}` severity {best_rerank[1]}: "
            f"R1 {best_rerank[2]:+.3f}, mAP {best_rerank[3]:+.3f}.",
            f"- Worst gated-rerank movement is `{worst_rerank[0]}` severity {worst_rerank[1]}: "
            f"R1 {worst_rerank[2]:+.3f}, mAP {worst_rerank[3]:+.3f}.",
            f"- Current SynCPR reranker is not a robust correction module under {domain} corruption; it is better treated as an ambiguity diagnostic until trained or gated on corrupted-query data.",
            "",
            "## Interpretation",
            "",
            "- RR is relative robustness against the clean FAFA baseline on the same query subset.",
            "- Per-corruption robustness is more informative than a single aggregate score; the benchmark exposes which composed-query input channel is brittle.",
            "- The stronger paper direction is now ITCPR-C: a robustness benchmark for composed person retrieval, with FAFA as a pretrained CPR model and CoPE/robustness benchmark ideas as methodology.",
            "- A model-combination direction still exists, but it should be driven by measured failure modes rather than assumed fixes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--itcpr-root", default="../data")
    parser.add_argument("--checkpoint", default="tuned_recall_at1_step.pt")
    parser.add_argument("--itcpr-cache", default="experiments/fafa_pretrained/itcpr_fafa_features.pt")
    parser.add_argument("--reranker-checkpoint", default="experiments/fafa_hard_negative_reranker/fafa_hard_negative_reranker.pt")
    parser.add_argument("--output-dir", default="experiments/itcpr_c_text_robustness")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fda-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--benchmark-mode", choices=["text", "image"], default="text")
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=["clean", "character_filter", "RemoveChar_filter", "qwerty_filter", "remove_space_filter", "repetition_filter"],
    )
    parser.add_argument(
        "--image-corruptions",
        nargs="+",
        default=["clean", "gaussian_noise_filter", "motion_blur_filter", "brightness_filter", "jpeg_compression"],
    )
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--image-severities", nargs="+", type=int, default=[3])
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--delta-mult", type=float, default=-0.015)
    parser.add_argument("--margin-quantile", type=float, default=0.30)
    parser.add_argument("--min-caption-words", type=int, default=6)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(args.itcpr_cache, map_location="cpu")
    gfeats = cache["gfeats"]
    gids = cache["gids"]
    if args.max_queries is not None:
        clean_qids = cache["qids"][: args.max_queries]
        clean_qfeats = cache["qfeats"][: args.max_queries]
    else:
        clean_qids = cache["qids"]
        clean_qfeats = cache["qfeats"]

    reranker = None if args.skip_rerank else load_reranker(args.reranker_checkpoint, device)
    base_scale = float(reranker.base_scale.detach().cpu()) if reranker is not None else 1.0

    print("Scoring clean cached FAFA features.")
    clean_scores = score_baseline(clean_qfeats, gfeats, device, args.fda_k, args.score_batch_size, base_scale)
    clean_baseline = evaluate_scores(clean_scores, clean_qids, gids)
    results = [
        {"corruption": "clean", "severity": 0, "method": "FAFA", "metrics": clean_baseline},
    ]

    model, txt_processors = load_fafa(args.checkpoint, device)
    model.eval()

    if args.benchmark_mode == "text":
        conditions = [
            (corruption, severity)
            for corruption in args.corruptions
            if corruption != "clean"
            for severity in args.severities
        ]
    else:
        conditions = [
            (corruption, severity)
            for corruption in args.image_corruptions
            if corruption != "clean"
            for severity in args.image_severities
        ]
    for corruption, severity in conditions:
            print(f"Evaluating {args.benchmark_mode} {corruption} severity={severity}.")
            if args.benchmark_mode == "text":
                query_set, corrupted_captions, original_captions = build_corrupted_query_set(
                    args.itcpr_root,
                    corruption,
                    severity,
                    args.seed,
                    args.max_queries,
                )
                sample_captions = [
                    {"original": o, "corrupted": c}
                    for o, c in list(zip(original_captions, corrupted_captions))[:5]
                ]
            else:
                query_set, corrupted_captions = build_image_corrupted_query_set(
                    args.itcpr_root,
                    corruption,
                    severity,
                    args.seed,
                    args.max_queries,
                )
                sample_captions = []
            qids, qfeats = extract_query_features(
                model,
                txt_processors,
                query_set,
                device,
                args.batch_size,
                args.num_workers,
            )
            base_scores = score_baseline(qfeats, gfeats, device, args.fda_k, args.score_batch_size, base_scale)
            metrics = evaluate_scores(base_scores, qids, gids)
            results.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "method": "FAFA",
                    "metrics": metrics,
                    "sample_captions": sample_captions,
                }
            )
            print(
                f"{corruption} sev={severity} FAFA R1={metrics['R1']:.3f} "
                f"R5={metrics['R5']:.3f} R10={metrics['R10']:.3f} mAP={metrics['mAP']:.3f}"
            )

            if reranker is not None:
                rerank_scores, gate = apply_gated_rerank(
                    base_scores,
                    qfeats,
                    gfeats,
                    reranker,
                    corrupted_captions,
                    device,
                    args.fda_k,
                    args.topk,
                    args.delta_mult,
                    args.margin_quantile,
                    args.min_caption_words,
                    args.query_batch_size,
                )
                rerank_metrics = evaluate_scores(rerank_scores, qids, gids)
                results.append(
                    {
                        "corruption": corruption,
                        "severity": severity,
                        "method": "FAFA+SynCPR-gated-rerank",
                        "metrics": rerank_metrics,
                        **gate,
                    }
                )
                print(
                    f"{corruption} sev={severity} rerank R1={rerank_metrics['R1']:.3f} "
                    f"R5={rerank_metrics['R5']:.3f} R10={rerank_metrics['R10']:.3f} "
                    f"mAP={rerank_metrics['mAP']:.3f} applied={gate['applied_queries']}"
                )

    payload = {
        "task": f"ITCPR-C {args.benchmark_mode} robustness pilot",
        "domain": args.benchmark_mode,
        "itcpr_root": args.itcpr_root,
        "checkpoint": args.checkpoint,
        "itcpr_cache": args.itcpr_cache,
        "reranker_checkpoint": None if args.skip_rerank else args.reranker_checkpoint,
        "num_queries": int(clean_qids.numel()),
        "num_gallery": int(gids.numel()),
        "clean_baseline": clean_baseline,
        "results": results,
        "args": vars(args),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir / "REPORT.md", payload)
    print(f"Saved results to {out_dir / 'results.json'}")
    print(f"Saved report to {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
