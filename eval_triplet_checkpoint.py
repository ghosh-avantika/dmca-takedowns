"""
Load a fine-tuned triplet checkpoint and run inference/evaluation utilities.

Examples:
  python eval_triplet_checkpoint.py --checkpoint finetuned_triplet_resnet.pt --image img1.jpg
  python eval_triplet_checkpoint.py --checkpoint finetuned_triplet_resnet.pt --image img1.jpg --image2 img2.jpg
  python eval_triplet_checkpoint.py --checkpoint finetuned_triplet_resnet.pt --image-dir ./images --out embeddings.pt
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def find_first_file(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def remap_path(raw_path: str, src_prefix: str | None, dst_prefix: str | None) -> Path:
    p = Path(raw_path).expanduser()
    if src_prefix and dst_prefix:
        src = str(Path(src_prefix).expanduser())
        dst = str(Path(dst_prefix).expanduser())
        raw = str(p)
        if raw.startswith(src):
            p = Path(dst + raw[len(src):])
    return p


def build_model(backbone: str, embed_dim: int) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    model.fc = nn.Linear(model.fc.in_features, embed_dim)
    return model


def build_metric_model(backbone: str, embed_dim: int, num_classes: int) -> nn.Module:
    if backbone == "resnet18":
        base = models.resnet18(weights=None)
    elif backbone == "resnet50":
        base = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    in_features = base.fc.in_features
    base.fc = nn.Identity()

    class MetricEvalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base
            self.embed_head = nn.Linear(in_features, embed_dim)
            self.classifier = nn.Linear(in_features, num_classes)

        def forward(self, x):
            feats = self.backbone(x)
            return self.embed_head(feats)

    return MetricEvalModel()


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), str(path)


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt["backbone"]
    embed_dim = ckpt["embed_dim"]
    state = ckpt["model_state"]

    is_metric_model = any(k.startswith("backbone.") for k in state.keys())
    if is_metric_model:
        num_classes = 1
        if "classifier.weight" in state:
            num_classes = int(state["classifier.weight"].shape[0])
        model = build_metric_model(backbone, int(embed_dim), num_classes)
        model.load_state_dict(state, strict=False)
    else:
        model = build_model(backbone, int(embed_dim))
        model.load_state_dict(state)

    model.to(device).eval()
    return ckpt, model


def parse_item_id(path_str: str, mode: str) -> str:
    name = Path(path_str).name.lower()
    if mode == "base_id":
        match = re.search(r"id_\d+", name, flags=re.IGNORECASE)
        return match.group(0) if match else Path(path_str).stem.lower()
    if mode == "parsed_id":
        match = re.search(r"id_\d+(?:-\d+)?", name, flags=re.IGNORECASE)
        return match.group(0) if match else Path(path_str).stem.lower()
    if mode == "filename":
        return Path(path_str).stem.lower()
    raise ValueError(f"Unknown item-id mode: {mode}")


@torch.no_grad()
def embed_one(model: nn.Module, image_path: Path, tfm, device: torch.device) -> torch.Tensor:
    x = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    z = model(x)
    if isinstance(z, tuple):
        z = z[0]
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    return z.squeeze(0).cpu()


@torch.no_grad()
def embed_many(
    model: nn.Module,
    image_paths: List[Path],
    tfm,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    def run_loader(worker_count: int):
        ds = ImagePathDataset(image_paths, tfm)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=worker_count)
        all_embeddings = []
        all_paths = []

        for images, paths in dl:
            images = images.to(device)
            z = model(images)
            if isinstance(z, tuple):
                z = z[0]
            z = torch.nn.functional.normalize(z, p=2, dim=1).cpu()
            all_embeddings.append(z)
            all_paths.extend(paths)
        return all_paths, torch.cat(all_embeddings, dim=0)

    try:
        return run_loader(num_workers)
    except RuntimeError as exc:
        if num_workers > 0 and ("torch_shm_manager" in str(exc) or "Operation not permitted" in str(exc)):
            print("Falling back to --num-workers 0 due to shared-memory restrictions in this environment.")
            return run_loader(0)
        raise


def evaluate_retrieval(
    embeddings: torch.Tensor,
    item_ids: List[str],
    topk: List[int],
) -> Dict[str, float]:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {tuple(embeddings.shape)}")
    n = embeddings.shape[0]
    if n != len(item_ids):
        raise ValueError("Embeddings and item_ids must have the same length.")
    if n < 2:
        raise ValueError("Need at least 2 images to run retrieval evaluation.")

    item_to_indices: Dict[str, List[int]] = {}
    for idx, item in enumerate(item_ids):
        item_to_indices.setdefault(item, []).append(idx)

    valid_queries = [i for i, item in enumerate(item_ids) if len(item_to_indices[item]) > 1]
    if not valid_queries:
        raise ValueError("No valid queries with at least one positive sample in this split.")

    topk = sorted(set(topk))
    if any(k <= 0 for k in topk):
        raise ValueError("--topk values must be positive integers.")
    max_k = min(max(topk), n - 1)

    similarities = embeddings @ embeddings.T
    hits = {k: 0 for k in topk}
    mrr_sum = 0.0
    ap_sum = 0.0

    for q_idx in valid_queries:
        target_item = item_ids[q_idx]
        row = similarities[q_idx].clone()
        row[q_idx] = -1e9
        ranked = torch.argsort(row, descending=True).tolist()

        for k in topk:
            k_eff = min(k, n - 1)
            if any(item_ids[j] == target_item for j in ranked[:k_eff]):
                hits[k] += 1

        first_rank = None
        for rank, j in enumerate(ranked, start=1):
            if item_ids[j] == target_item:
                first_rank = rank
                break
        if first_rank is not None:
            mrr_sum += 1.0 / first_rank

        num_relevant = len(item_to_indices[target_item]) - 1
        found = 0
        precision_sum = 0.0
        for rank, j in enumerate(ranked, start=1):
            if item_ids[j] == target_item:
                found += 1
                precision_sum += found / rank
                if found == num_relevant:
                    break
        if num_relevant > 0:
            ap_sum += precision_sum / num_relevant

    num_queries = len(valid_queries)
    metrics: Dict[str, float] = {
        "num_images": float(n),
        "num_queries": float(num_queries),
        "MRR": mrr_sum / num_queries,
        "mAP": ap_sum / num_queries,
    }
    for k in topk:
        metrics[f"Recall@{k}"] = hits[k] / num_queries
    metrics["max_k_used"] = float(max_k)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run inference using a triplet-tuned ResNet checkpoint.")
    parser.add_argument("--checkpoint", help="Path to finetuned_triplet_resnet.pt")
    parser.add_argument("--image", help="Path to one image for embedding or pair similarity")
    parser.add_argument("--image2", help="Second image path for similarity scoring")
    parser.add_argument("--image-dir", help="Directory to batch-embed and save vectors")
    parser.add_argument("--deepfashion-pt", help="Path to deepfashion.pt for retrieval eval mode")
    parser.add_argument(
        "--eval-split",
        choices=["train", "val", "test", "all"],
        default="val",
        help="Split to evaluate when --deepfashion-pt is provided",
    )
    parser.add_argument(
        "--item-id-source",
        choices=["auto", "dataset", "parse"],
        default="auto",
        help="Use item IDs from dataset when available, otherwise parse from filename",
    )
    parser.add_argument(
        "--item-id-mode",
        choices=["auto", "base_id", "parsed_id", "filename"],
        default="auto",
        help="Filename parsing mode for item IDs when parsing is used; 'auto' uses checkpoint mode if present",
    )
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10], help="Recall@K values")
    parser.add_argument("--out", default="triplet_embeddings.pt", help="Output .pt path for --image-dir mode")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--path-prefix-from",
        help="Optional source path prefix to rewrite paths stored in deepfashion.pt (e.g. /Users/name/Downloads/densepose).",
    )
    parser.add_argument(
        "--path-prefix-to",
        help="Optional destination prefix for rewritten paths (e.g. /kaggle/input/densepose).",
    )
    # In notebook runtimes, extra args like "-f <kernel.json>" may be injected.
    # parse_known_args lets us ignore those safely.
    args, _unknown = parser.parse_known_args(argv)

    def require_existing_path(raw_path: str, label: str) -> Path:
        p = Path(raw_path).expanduser()
        if "/path/to/" in raw_path or str(p).startswith("/path/to/"):
            parser.error(
                f"{label} looks like a placeholder path: {raw_path}\n"
                f"Use a real file path, for example: --image ./data/sample.jpg"
            )
        if not p.exists():
            parser.error(
                f"{label} not found: {p.resolve()}\n"
                f"Use an existing path."
            )
        return p

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Kaggle-friendly fallback: auto-discover checkpoint if not provided.
    if not args.checkpoint:
        candidates = [
            Path("/kaggle/working/finetuned_triplet_resnet_v2_best.pt"),
            Path("/kaggle/working/finetuned_triplet_resnet_v2.pt"),
            Path("/kaggle/working/finetuned_triplet_resnet_best.pt"),
            Path("/kaggle/working/finetuned_triplet_resnet.pt"),
        ]
        checkpoint_guess = next((p for p in candidates if p.exists()), None)
        if checkpoint_guess is None:
            checkpoint_guess = find_first_file(Path("/kaggle/working"), "finetuned_triplet*.pt")
        if checkpoint_guess is None:
            parser.error(
                "Checkpoint not provided and no default checkpoint found in /kaggle/working.\n"
                "Pass --checkpoint /kaggle/working/<your_model>.pt"
            )
        args.checkpoint = str(checkpoint_guess)
        print(f"Auto-detected checkpoint: {args.checkpoint}")

    if args.deepfashion_pt is None and not args.image and not args.image_dir:
        # Auto-detect deepfashion.pt only when no explicit inference target was provided.
        deepfashion_guess = find_first_file(Path("/kaggle/input"), "deepfashion.pt")
        if deepfashion_guess is not None:
            args.deepfashion_pt = str(deepfashion_guess)
            print(f"Auto-detected deepfashion.pt: {args.deepfashion_pt}")

    checkpoint_path = require_existing_path(args.checkpoint, "Checkpoint")

    ckpt, model = load_checkpoint(checkpoint_path, device)
    tfm = build_transform()
    resolved_item_id_mode = ckpt.get("item_id_mode", "base_id") if args.item_id_mode == "auto" else args.item_id_mode

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Backbone: {ckpt['backbone']} | Embedding dim: {ckpt['embed_dim']} | Device: {device}")

    if args.image:
        image_path = require_existing_path(args.image, "Image")
        if not image_path.is_file():
            parser.error(f"Image must be a file path, got: {image_path}")
        emb = embed_one(model, image_path, tfm, device)
        print(f"Embedded image: {image_path}")
        print(f"Embedding shape: {tuple(emb.shape)}")
        print(f"Embedding norm: {emb.norm().item():.6f}")
        print(f"First 8 values: {emb[:8].tolist()}")

    if args.image and args.image2:
        image2_path = require_existing_path(args.image2, "Image2")
        if not image2_path.is_file():
            parser.error(f"Image2 must be a file path, got: {image2_path}")
        emb1 = embed_one(model, image_path, tfm, device)
        emb2 = embed_one(model, image2_path, tfm, device)
        sim = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1
        ).item()
        print(f"Cosine similarity: {sim:.6f}")

    if args.image_dir:
        image_dir = require_existing_path(args.image_dir, "Directory")
        if not image_dir.is_dir():
            parser.error(f"Directory must be a directory path, got: {image_dir}")

        image_paths = list_images(image_dir)
        if not image_paths:
            raise ValueError(f"No supported images found in: {image_dir}")

        print(f"Embedding {len(image_paths)} images from: {image_dir}")
        paths, embeddings = embed_many(
            model=model,
            image_paths=image_paths,
            tfm=tfm,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        out_path = Path(args.out)
        torch.save(
            {
                "checkpoint": str(checkpoint_path),
                "backbone": ckpt["backbone"],
                "embed_dim": ckpt["embed_dim"],
                "paths": paths,
                "embeddings": embeddings,
            },
            out_path,
        )
        print(f"Saved embeddings: {out_path} | shape={tuple(embeddings.shape)}")

    if args.deepfashion_pt:
        deepfashion_path = require_existing_path(args.deepfashion_pt, "DeepFashion PT")
        if not deepfashion_path.is_file():
            parser.error(f"DeepFashion PT must be a file path, got: {deepfashion_path}")

        data = torch.load(deepfashion_path, map_location="cpu")
        splits = ["train", "val", "test"] if args.eval_split == "all" else [args.eval_split]

        for split_name in splits:
            if split_name not in data:
                parser.error(f"Split '{split_name}' not found in dataset: {deepfashion_path}")
            split_obj = data[split_name]
            if "paths" not in split_obj:
                parser.error(f"Split '{split_name}' is missing 'paths' in dataset.")

            image_paths = [
                remap_path(str(p), args.path_prefix_from, args.path_prefix_to)
                for p in split_obj["paths"]
            ]
            missing = [str(p) for p in image_paths if not p.exists()]
            if missing:
                parser.error(
                    f"Split '{split_name}' has missing image files (showing first 3): "
                    f"{missing[:3]}"
                )

            if args.item_id_source == "dataset":
                if "item_ids" not in split_obj:
                    parser.error(f"Split '{split_name}' missing 'item_ids' while --item-id-source=dataset.")
                item_ids = [str(x) for x in split_obj["item_ids"]]
            elif args.item_id_source == "parse":
                item_ids = [parse_item_id(str(p), resolved_item_id_mode) for p in image_paths]
            else:
                if "item_ids" in split_obj and len(split_obj["item_ids"]) == len(image_paths):
                    item_ids = [str(x) for x in split_obj["item_ids"]]
                else:
                    item_ids = [parse_item_id(str(p), resolved_item_id_mode) for p in image_paths]

            print(f"\nEvaluating split: {split_name} | images={len(image_paths)}")
            if args.item_id_source == "parse" or (
                args.item_id_source == "auto" and ("item_ids" not in split_obj or len(split_obj["item_ids"]) != len(image_paths))
            ):
                print(f"Item ID mode (parsing): {resolved_item_id_mode}")
            _, embeddings = embed_many(
                model=model,
                image_paths=image_paths,
                tfm=tfm,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            metrics = evaluate_retrieval(embeddings=embeddings, item_ids=item_ids, topk=args.topk)
            print(f"split: {split_name}")
            print(f"num_images: {int(metrics['num_images'])}")
            print(f"num_queries: {int(metrics['num_queries'])}")
            for k in sorted(set(args.topk)):
                print(f"Recall@{k}: {metrics[f'Recall@{k}']:.6f}")
            print(f"MRR: {metrics['MRR']:.6f}")
            print(f"mAP: {metrics['mAP']:.6f}")

    if not args.image and not args.image_dir and not args.deepfashion_pt:
        print("No inference target provided. Use --image, --image-dir, or --deepfashion-pt.")
        print("Kaggle example:")
        print(
            "  !python eval_triplet_checkpoint.py "
            "--checkpoint /kaggle/working/finetuned_triplet_resnet_v2_best.pt "
            "--deepfashion-pt /kaggle/input/<dataset>/deepfashion.pt --eval-split all --topk 1 5 10"
        )


if __name__ == "__main__":
    # Default behavior remains CLI-friendly; works in Kaggle with `!python ...`.
    # Also safe in notebook `%run eval_triplet_checkpoint.py -- <args>`.
    main(sys.argv[1:])
