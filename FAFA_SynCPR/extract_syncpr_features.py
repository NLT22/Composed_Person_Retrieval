#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, "src")

from data_utils import squarepad_transform_test
from lavis.models import load_model_and_preprocess


def collate_syncpr(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    indices, cpr_ids, refs, tgts, captions = zip(*batch)
    return (
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(cpr_ids, dtype=torch.long),
        torch.stack(refs, dim=0),
        torch.stack(tgts, dim=0),
        list(captions),
    )


class SynCPRShardDataset(Dataset):
    def __init__(self, root, annotations, start, end, preprocess, skip_missing=False):
        self.root = Path(root)
        self.annotations = annotations
        self.start = start
        self.end = end
        self.preprocess = preprocess
        self.skip_missing = skip_missing

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, local_index):
        index = self.start + local_index
        item = self.annotations[index]
        ref_path = self.root / item["reference_image_path"]
        tgt_path = self.root / item["target_image_path"]

        try:
            ref_img = self.preprocess(Image.open(ref_path).convert("RGB"))
            tgt_img = self.preprocess(Image.open(tgt_path).convert("RGB"))
        except Exception as exc:
            if not self.skip_missing:
                raise
            print(f"Skipping index={index}: {exc}")
            return None

        return index, int(item["cpr_id"]), ref_img, tgt_img, item["edit_caption"]


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


def shard_path(out_dir, start, end):
    return out_dir / f"shard_{start:07d}_{end - 1:07d}.pt"


@torch.no_grad()
def extract_shard(model, txt_processors, dataset, args, device, out_path):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_syncpr,
    )

    all_indices = []
    all_cpr_ids = []
    all_qfeats = []
    all_target_tokens = []
    missing_or_failed = 0

    for batch in tqdm(loader, desc=out_path.name):
        if batch is None:
            missing_or_failed += args.batch_size
            continue

        indices, cpr_ids, ref_imgs, tgt_imgs, captions = batch
        ref_imgs = ref_imgs.to(device, non_blocking=True)
        tgt_imgs = tgt_imgs.to(device, non_blocking=True)
        processed_captions = [txt_processors["eval"](caption) for caption in captions]

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            target_tokens, _ = model.extract_target_features(
                tgt_imgs.half() if device.type == "cuda" else tgt_imgs
            )
            qfeats = model.extract_features(
                {
                    "image": ref_imgs.half() if device.type == "cuda" else ref_imgs,
                    "text_input": processed_captions,
                }
            ).multimodal_embeds

        all_indices.append(indices.cpu())
        all_cpr_ids.append(cpr_ids.cpu())
        all_qfeats.append(qfeats.detach().cpu().half())
        all_target_tokens.append(target_tokens.detach().cpu().half())

    payload = {
        "indices": torch.cat(all_indices, dim=0) if all_indices else torch.empty(0, dtype=torch.long),
        "cpr_ids": torch.cat(all_cpr_ids, dim=0) if all_cpr_ids else torch.empty(0, dtype=torch.long),
        "qfeats": torch.cat(all_qfeats, dim=0) if all_qfeats else torch.empty(0, 256, dtype=torch.float16),
        "target_tokens": torch.cat(all_target_tokens, dim=0)
        if all_target_tokens
        else torch.empty(0, 32, 256, dtype=torch.float16),
        "source": {
            "syncpr_root": str(args.syncpr_root),
            "checkpoint": str(args.checkpoint),
            "start": dataset.start,
            "end": dataset.end,
            "batch_size": args.batch_size,
            "missing_or_failed_estimate": missing_or_failed,
        },
    }
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)
    print(f"Saved {out_path} with {payload['indices'].numel()} samples")


def validate_first_sample(root, annotations):
    if not annotations:
        raise RuntimeError("SynCPR annotations are empty")
    first = annotations[0]
    missing = []
    for key in ("reference_image_path", "target_image_path"):
        path = Path(root) / first[key]
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("SynCPR images are not extracted yet. Missing: " + ", ".join(missing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syncpr-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        default="experiments/fafa_pretrained/syncpr_features",
        type=Path,
    )
    parser.add_argument("--json-name", default="SynCPR.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=25000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing/corrupt image pairs instead of failing the shard.",
    )
    args = parser.parse_args()

    annotations_path = args.syncpr_root / args.json_name
    print(f"Loading annotations from {annotations_path}")
    with annotations_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    validate_first_sample(args.syncpr_root, annotations)

    total = len(annotations)
    start_index = max(0, args.start_index)
    end_index = total if args.max_samples is None else min(total, start_index + args.max_samples)
    print(f"SynCPR samples available={total}; extracting range=[{start_index}, {end_index})")

    if args.verify_only:
        print("Verified annotation JSON and first image pair.")
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, txt_processors = load_fafa(args.checkpoint, device)
    preprocess = squarepad_transform_test(224)

    for shard_start in range(start_index, end_index, args.shard_size):
        shard_end = min(shard_start + args.shard_size, end_index)
        out_path = shard_path(args.out_dir, shard_start, shard_end)
        if out_path.exists() and not args.force:
            print(f"Skipping existing shard {out_path}")
            continue

        dataset = SynCPRShardDataset(
            root=args.syncpr_root,
            annotations=annotations,
            start=shard_start,
            end=shard_end,
            preprocess=preprocess,
            skip_missing=args.skip_missing,
        )
        extract_shard(model, txt_processors, dataset, args, device, out_path)

    manifest = {
        "syncpr_root": str(args.syncpr_root),
        "checkpoint": str(args.checkpoint),
        "json_name": args.json_name,
        "num_annotations": total,
        "range": [start_index, end_index],
        "shard_size": args.shard_size,
        "feature_shapes": {
            "qfeats": ["N", 256],
            "target_tokens": ["N", 32, 256],
        },
        "dtype": "float16",
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
