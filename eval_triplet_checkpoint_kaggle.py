# Kaggle cell: triplet checkpoint eval (no argparse needed)

import re
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# =========================
# CONFIG (edit these)
# =========================
CHECKPOINT = "/kaggle/working/finetuned_triplet_resnet_v2_best.pt"  # e.g. "/kaggle/working/finetuned_triplet_resnet_v2_best.pt"
DEEPFASHION_PT = "/kaggle/input/datasets/milly2019/deepfashion-split-data-metadata/deepfashion.pt"  # e.g. "/kaggle/input/.../deepfashion.pt"
KAGGLE_IMAGE_ROOT = "/kaggle/input/datasets/milly2019/mlp-metric-learning-densepose-training-dataset/densepose"  # e.g. "/kaggle/input/.../densepose"

EVAL_SPLIT = "all"  # "train", "val", "test", "all"
TOPK = [1, 5, 10]
BATCH_SIZE = 64
NUM_WORKERS = 2
DEVICE = "auto"  # "auto", "cpu", "cuda"
ITEM_ID_SOURCE = "parse"  # forced for protocol match: "parse"
ITEM_ID_MODE = "base_id"  # forced for protocol match: "base_id"
SKIP_TRAIN_WHEN_ALL = True  # recommended for speed; evaluate val/test when EVAL_SPLIT="all"
OUT_METRICS_JSON = "/kaggle/working/triplet_eval_metrics.json"

# Optional single-image tests:
IMAGE = None
IMAGE2 = None
IMAGE_DIR = None
OUT_EMBEDDINGS = "/kaggle/working/triplet_embeddings.pt"


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def find_first_file(root: Path, filename: str) -> Optional[Path]:
    if not root.exists():
        return None
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def find_first_dir_contains(root: Path, needle: str) -> Optional[Path]:
    if not root.exists():
        return None
    needle_l = needle.lower()
    for p in root.rglob("*"):
        if p.is_dir() and needle_l in p.name.lower():
            return p
    return None


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
    state = ckpt["model_state"]
    backbone = ckpt["backbone"]
    embed_dim = int(ckpt["embed_dim"])

    is_metric_model = any(k.startswith("backbone.") for k in state.keys())
    if is_metric_model:
        num_classes = 1
        if "classifier.weight" in state:
            num_classes = int(state["classifier.weight"].shape[0])
        model = build_metric_model(backbone, embed_dim, num_classes)
        model.load_state_dict(state, strict=False)
    else:
        model = build_model(backbone, embed_dim)
        model.load_state_dict(state)

    model.to(device).eval()
    return ckpt, model


def parse_item_id(path_str: str, mode: str) -> str:
    name = Path(path_str).name.lower()
    if mode == "base_id":
        m = re.search(r"id_\d+", name, flags=re.IGNORECASE)
        return m.group(0) if m else Path(path_str).stem.lower()
    if mode == "parsed_id":
        m = re.search(r"id_\d+(?:-\d+)?", name, flags=re.IGNORECASE)
        return m.group(0) if m else Path(path_str).stem.lower()
    if mode == "filename":
        return Path(path_str).stem.lower()
    raise ValueError(f"Unknown item-id mode: {mode}")


def remap_to_kaggle_path(raw_path: str, kaggle_image_root: Optional[Path]) -> Path:
    p = Path(raw_path).expanduser()
    if p.exists():
        return p
    if kaggle_image_root is None:
        return p
    return kaggle_image_root / p.name


@torch.no_grad()
def embed_one(model: nn.Module, image_path: Path, tfm, device: torch.device) -> torch.Tensor:
    x = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    z = model(x)
    if isinstance(z, tuple):
        z = z[0]
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    return z.squeeze(0).cpu()


@torch.no_grad()
def embed_many(model, image_paths, tfm, device, batch_size, num_workers):
    def run_loader(worker_count: int):
        ds = ImagePathDataset(image_paths, tfm)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=worker_count)
        all_embeddings = []
        all_paths = []
        total_batches = len(dl)
        for step, (images, paths) in enumerate(dl, start=1):
            images = images.to(device)
            z = model(images)
            if isinstance(z, tuple):
                z = z[0]
            z = torch.nn.functional.normalize(z, p=2, dim=1).cpu()
            all_embeddings.append(z)
            all_paths.extend(paths)
            if step == 1 or step % 50 == 0 or step == total_batches:
                print(
                    f"Embedding progress: batch {step}/{total_batches} "
                    f"({len(all_paths)}/{len(image_paths)} images)",
                    flush=True,
                )
        return all_paths, torch.cat(all_embeddings, dim=0)

    try:
        return run_loader(num_workers)
    except RuntimeError as exc:
        if num_workers > 0 and ("torch_shm_manager" in str(exc) or "Operation not permitted" in str(exc)):
            print("Falling back to num_workers=0 due to shared-memory restrictions.")
            return run_loader(0)
        raise


def evaluate_retrieval(embeddings: torch.Tensor, item_ids: List[str], topk: List[int]) -> Dict[str, float]:
    n = embeddings.shape[0]
    if n != len(item_ids):
        raise ValueError("Embeddings and item_ids must have same length.")
    if n < 2:
        raise ValueError("Need at least 2 images.")

    item_to_indices: Dict[str, List[int]] = {}
    for i, item in enumerate(item_ids):
        item_to_indices.setdefault(item, []).append(i)

    valid_queries = [i for i, item in enumerate(item_ids) if len(item_to_indices[item]) > 1]
    if not valid_queries:
        raise ValueError("No valid queries (no items with >1 image).")

    topk = sorted(set(topk))
    sims = embeddings @ embeddings.T
    hits = {k: 0 for k in topk}
    mrr_sum = 0.0
    ap_sum = 0.0

    total_queries = len(valid_queries)
    for q_num, q_idx in enumerate(valid_queries, start=1):
        target = item_ids[q_idx]
        row = sims[q_idx].clone()
        row[q_idx] = -1e9
        ranked = torch.argsort(row, descending=True).tolist()

        for k in topk:
            k_eff = min(k, n - 1)
            if any(item_ids[j] == target for j in ranked[:k_eff]):
                hits[k] += 1

        first_rank = None
        for rank, j in enumerate(ranked, start=1):
            if item_ids[j] == target:
                first_rank = rank
                break
        if first_rank is not None:
            mrr_sum += 1.0 / first_rank

        num_rel = len(item_to_indices[target]) - 1
        found = 0
        precision_sum = 0.0
        for rank, j in enumerate(ranked, start=1):
            if item_ids[j] == target:
                found += 1
                precision_sum += found / rank
                if found == num_rel:
                    break
        if num_rel > 0:
            ap_sum += precision_sum / num_rel
        if q_num == 1 or q_num % 500 == 0 or q_num == total_queries:
            print(f"Retrieval progress: query {q_num}/{total_queries}", flush=True)

    nq = total_queries
    out = {
        "num_images": float(n),
        "num_queries": float(nq),
        "MRR": mrr_sum / nq,
        "mAP": ap_sum / nq,
    }
    for k in topk:
        out[f"Recall@{k}"] = hits[k] / nq
    return out


# =========================
# Run
# =========================
device = torch.device("cuda" if (DEVICE == "auto" and torch.cuda.is_available()) else DEVICE if DEVICE != "auto" else "cpu")

if CHECKPOINT is None:
    for c in [
        "/kaggle/working/finetuned_triplet_resnet_v2_best.pt",
        "/kaggle/working/finetuned_triplet_resnet_v2.pt",
        "/kaggle/working/finetuned_triplet_resnet_best.pt",
        "/kaggle/working/finetuned_triplet_resnet.pt",
    ]:
        if Path(c).exists():
            CHECKPOINT = c
            break
if CHECKPOINT is None:
    raise FileNotFoundError("No checkpoint found. Set CHECKPOINT at top.")

if DEEPFASHION_PT is None and IMAGE is None and IMAGE_DIR is None:
    guess = find_first_file(Path("/kaggle/input"), "deepfashion.pt")
    if guess:
        DEEPFASHION_PT = str(guess)

if KAGGLE_IMAGE_ROOT is None:
    guess_dir = find_first_dir_contains(Path("/kaggle/input"), "densepose")
    if guess_dir:
        KAGGLE_IMAGE_ROOT = str(guess_dir)

checkpoint_path = Path(CHECKPOINT)
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

ckpt, model = load_checkpoint(checkpoint_path, device)
tfm = build_transform()
resolved_item_id_mode = ITEM_ID_MODE

print(f"Loaded checkpoint: {checkpoint_path}")
print(f"Backbone: {ckpt['backbone']} | Embedding dim: {ckpt['embed_dim']} | Device: {device}")
print(f"Item ID protocol: source={ITEM_ID_SOURCE}, mode={resolved_item_id_mode}")

if IMAGE:
    image_path = Path(IMAGE).expanduser()
    emb = embed_one(model, image_path, tfm, device)
    print(f"Embedded image: {image_path}")
    print(f"Embedding shape: {tuple(emb.shape)}")
    print(f"Embedding norm: {emb.norm().item():.6f}")
    print(f"First 8 values: {emb[:8].tolist()}")

if IMAGE and IMAGE2:
    emb1 = embed_one(model, Path(IMAGE).expanduser(), tfm, device)
    emb2 = embed_one(model, Path(IMAGE2).expanduser(), tfm, device)
    sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()
    print(f"Cosine similarity: {sim:.6f}")

if IMAGE_DIR:
    image_dir = Path(IMAGE_DIR).expanduser()
    image_paths = list_images(image_dir)
    print(f"Embedding {len(image_paths)} images from: {image_dir}")
    paths, embeddings = embed_many(model, image_paths, tfm, device, BATCH_SIZE, NUM_WORKERS)
    torch.save(
        {
            "checkpoint": str(checkpoint_path),
            "backbone": ckpt["backbone"],
            "embed_dim": ckpt["embed_dim"],
            "paths": paths,
            "embeddings": embeddings,
        },
        OUT_EMBEDDINGS,
    )
    print(f"Saved embeddings: {OUT_EMBEDDINGS} | shape={tuple(embeddings.shape)}")

if DEEPFASHION_PT:
    deepfashion_path = Path(DEEPFASHION_PT).expanduser()
    if not deepfashion_path.exists():
        raise FileNotFoundError(f"deepfashion.pt not found: {deepfashion_path}")

    kaggle_root = Path(KAGGLE_IMAGE_ROOT) if KAGGLE_IMAGE_ROOT else None
    if kaggle_root:
        print(f"Using image root: {kaggle_root}")

    data = torch.load(deepfashion_path, map_location="cpu")
    if EVAL_SPLIT == "all":
        splits = ["val", "test"] if SKIP_TRAIN_WHEN_ALL else ["train", "val", "test"]
    else:
        splits = [EVAL_SPLIT]

    all_metrics: Dict[str, Dict[str, float]] = {}

    for split_name in splits:
        split_obj = data[split_name]
        raw_paths = split_obj["paths"]
        image_paths = [remap_to_kaggle_path(str(p), kaggle_root) for p in raw_paths]

        missing = [str(p) for p in image_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{split_name}: missing files after remap (first 3): {missing[:3]}\n"
                f"Set KAGGLE_IMAGE_ROOT correctly."
            )

        # Forced protocol: always parse IDs from filename using base_id.
        item_ids = [parse_item_id(str(p), resolved_item_id_mode) for p in image_paths]

        print(f"\nEvaluating split: {split_name} | images={len(image_paths)}")
        _, embeddings = embed_many(model, image_paths, tfm, device, BATCH_SIZE, NUM_WORKERS)
        metrics = evaluate_retrieval(embeddings, item_ids, TOPK)

        print(f"split: {split_name}")
        print(f"num_images: {int(metrics['num_images'])}")
        print(f"num_queries: {int(metrics['num_queries'])}")
        for k in sorted(set(TOPK)):
            print(f"Recall@{k}: {metrics[f'Recall@{k}']:.6f}")
        print(f"MRR: {metrics['MRR']:.6f}")
        print(f"mAP: {metrics['mAP']:.6f}")
        all_metrics[split_name] = {k: float(v) for k, v in metrics.items()}

    with open(OUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "deepfashion_pt": str(deepfashion_path),
                "kaggle_image_root": str(kaggle_root) if kaggle_root else None,
                "eval_split": EVAL_SPLIT,
                "actual_splits": splits,
                "item_id_source": ITEM_ID_SOURCE,
                "item_id_mode": resolved_item_id_mode,
                "topk": TOPK,
                "metrics": all_metrics,
            },
            f,
            indent=2,
        )
    print(f"\nSaved metrics JSON: {OUT_METRICS_JSON}")
else:
    if not IMAGE and not IMAGE_DIR:
        print("Nothing to run. Set DEEPFASHION_PT or IMAGE/IMAGE_DIR in CONFIG.")
