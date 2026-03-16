"""
Build a DeepFashion-style .pt file from an image directory.

Creates CLIP embeddings and writes:
{
  "train": {"embeddings": Tensor[N,D], "item_ids": list, "keywords": list, "paths": list},
  "val":   {...},
  "test":  {...}
}

By default, item_ids are the filename (as requested). This yields no multi-view positives.
Use --item-id-mode parsed_id if you want the same item identity across views.
"""

import argparse
import os
import random
import re
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
import clip


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(root: str) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]
    return paths


GENERIC_TOKENS = {
    "densepose", "front", "back", "side", "full", "flat", "additional",
    "png", "jpg", "jpeg", "webp", "id",
}


def tokenize_keywords(filename: str) -> List[str]:
    # Example: WOMEN-Dresses-id_00005051-02_1_front_densepose.png
    tokens = re.findall(r"[a-z0-9]+", filename.lower())
    tokens = [t for t in tokens if t not in GENERIC_TOKENS]
    return tokens


def extract_item_id(filename: str, mode: str) -> str:
    if mode == "filename":
        return filename
    if mode == "stem":
        return Path(filename).stem
    if mode == "parsed_id":
        # Try to use the "id_00005051-02" portion as identity
        match = re.search(r"id_\d+(?:-\d+)?", filename)
        if match:
            return match.group(0)
        return Path(filename).stem
    raise ValueError(f"Unknown item-id mode: {mode}")


def split_by_item_ids(
    item_ids: List[str],
    split: Tuple[float, float, float],
) -> Tuple[List[int], List[int], List[int]]:
    item_to_indices = {}
    for i, item in enumerate(item_ids):
        item_to_indices.setdefault(item, []).append(i)
    items = list(item_to_indices.keys())
    random.shuffle(items)

    n_items = len(items)
    n_train = int(n_items * split[0])
    n_val = int(n_items * split[1])
    train_items = set(items[:n_train])
    val_items = set(items[n_train:n_train + n_val])
    test_items = set(items[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for item, indices in item_to_indices.items():
        if item in train_items:
            train_idx.extend(indices)
        elif item in val_items:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)
    return train_idx, val_idx, test_idx


def embed_images(paths: List[Path], device: torch.device, batch_size: int) -> torch.Tensor:
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = torch.stack(images).to(device)
            emb = model.encode_image(batch)
            emb = emb.float().cpu()
            all_embeddings.append(emb)

    return torch.cat(all_embeddings, dim=0)


def build_split(
    paths: List[Path],
    embeddings: torch.Tensor,
    item_id_mode: str,
) -> dict:
    filenames = [p.name for p in paths]
    item_ids = [extract_item_id(fn, item_id_mode) for fn in filenames]
    keywords = [tokenize_keywords(fn) for fn in filenames]
    return {
        "embeddings": embeddings,
        "item_ids": item_ids,
        "keywords": keywords,
        "paths": [str(p) for p in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DeepFashion .pt embeddings from image directory.")
    parser.add_argument("--image-dir", required=True, help="Path to image directory")
    parser.add_argument("--out", default="deepfashion.pt", help="Output .pt file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--item-id-mode", choices=["filename", "stem", "parsed_id"], default="filename")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Scanning images in: {args.image_dir}")
    paths = list_images(args.image_dir)
    if args.max_samples:
        paths = paths[: args.max_samples]
    if not paths:
        raise ValueError("No images found.")

    print(f"Found {len(paths)} images")
    print(f"Device: {device}")
    print(f"Item ID mode: {args.item_id_mode}")

    embeddings = embed_images(paths, device=device, batch_size=args.batch_size)
    filenames = [p.name for p in paths]
    item_ids = [extract_item_id(fn, args.item_id_mode) for fn in filenames]

    train_idx, val_idx, test_idx = split_by_item_ids(item_ids, tuple(args.split))

    def subset(indices: List[int]) -> dict:
        sub_paths = [paths[i] for i in indices]
        sub_emb = embeddings[indices]
        return build_split(sub_paths, sub_emb, args.item_id_mode)

    data = {
        "train": subset(train_idx),
        "val": subset(val_idx),
        "test": subset(test_idx),
    }

    torch.save(data, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
