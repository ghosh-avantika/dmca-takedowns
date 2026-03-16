"""
Notebook-friendly triplet training script for DeepFashion-style images.

Usage in one Kaggle cell:
1) Set IMAGE_DIR and optionally MANIFEST_CSV.
2) Run this file content in a single cell.

Manifest mode (recommended):
- CSV with columns: image_path,item_id
- Optional split column: split (train/val/test)
"""

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models, transforms


# =========================
# Config (edit these)
# =========================
IMAGE_DIR = "/kaggle/input/REPLACE_WITH_REAL_FOLDER"
MANIFEST_CSV = None  # e.g. "/kaggle/input/your-dataset/train_manifest.csv"
OUT_MODEL = "/kaggle/working/finetuned_triplet_resnet_v3.pt"

BACKBONE = "resnet18"  # "resnet18" or "resnet50"
PRETRAINED = True
EMBED_DIM = 128
EPOCHS = 20
SEED = 42
ITEM_ID_MODE = "base_id"  # used only when MANIFEST_CSV is None

# Optimization
BASE_LR = 1e-5            # backbone LR
HEAD_LR = 1e-4            # embedding / classifier heads LR
WEIGHT_DECAY = 1e-4
MARGIN_START = 0.08
MARGIN_END = 0.16
TRIPLET_WEIGHT = 1.0
CE_WEIGHT = 0.5

# Loader / sampler
NUM_WORKERS = 2
P_PER_BATCH = 16          # number of item IDs per batch
K_PER_ITEM = 4            # images sampled per item ID
STEPS_PER_EPOCH = 500
VAL_FRACTION = 0.1        # used only when split column is absent
EVAL_BATCH_SIZE = 128

# Unfreeze schedule
UNFREEZE_LAYER3_EPOCH = 5
UNFREEZE_ALL_EPOCH = 9

# Early stop / selection
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4
EVAL_FROZEN_BASELINE = True
BEST_METRIC = "recall@1"  # "mAP" or "recall@1"


@dataclass
class SampleRecord:
    path: Path
    item_id: str
    split: Optional[str] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_item_id(filename: str, mode: str) -> str:
    name = filename.lower()
    if mode == "base_id":
        match = re.search(r"id_\d+", name, flags=re.IGNORECASE)
        return match.group(0) if match else Path(filename).stem
    if mode == "parsed_id":
        match = re.search(r"id_\d+(?:-\d+)?", name, flags=re.IGNORECASE)
        return match.group(0) if match else Path(filename).stem
    if mode == "filename":
        return Path(filename).stem
    raise ValueError(f"Unknown item-id mode: {mode}")


def list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def resolve_manifest_path(raw: str, image_root: Path) -> Optional[Path]:
    p = Path(raw).expanduser()
    if p.exists():
        return p
    candidate = image_root / raw
    if candidate.exists():
        return candidate
    return None


def load_manifest_records(manifest_csv: Path, image_root: Path) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    missing = 0
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Manifest CSV has no header.")
        required = {"image_path", "item_id"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError("Manifest must contain columns: image_path,item_id")

        for row in reader:
            raw_path = (row.get("image_path") or "").strip()
            item_id = (row.get("item_id") or "").strip()
            split = (row.get("split") or "").strip().lower() or None
            if not raw_path or not item_id:
                continue
            resolved = resolve_manifest_path(raw_path, image_root)
            if resolved is None:
                missing += 1
                continue
            records.append(SampleRecord(path=resolved, item_id=item_id, split=split))

    if missing > 0:
        print(f"Manifest warning: skipped {missing} missing image path(s).")
    if not records:
        raise ValueError("No valid records loaded from manifest.")
    return records


def build_records_from_files(paths: List[Path], item_id_mode: str) -> List[SampleRecord]:
    return [SampleRecord(path=p, item_id=parse_item_id(p.name, item_id_mode)) for p in paths]


def split_records(
    records: List[SampleRecord],
    val_fraction: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    # If split labels are provided in manifest, use them.
    has_split = any(r.split is not None for r in records)
    if has_split:
        train = [r for r in records if r.split in {"train", "tr"}]
        val = [r for r in records if r.split in {"val", "valid", "validation", "dev"}]
        if train and val:
            return train, val
        print("Manifest split column found but train/val are incomplete; falling back to item-level split.")

    item_to_records: Dict[str, List[SampleRecord]] = {}
    for r in records:
        item_to_records.setdefault(r.item_id, []).append(r)

    item_ids = list(item_to_records.keys())
    rng = random.Random(seed)
    rng.shuffle(item_ids)

    if len(item_ids) <= 1 or val_fraction <= 0:
        return records, []

    val_items = max(1, int(round(len(item_ids) * val_fraction)))
    val_items = min(val_items, len(item_ids) - 1)
    val_item_ids = set(item_ids[:val_items])

    train_records: List[SampleRecord] = []
    val_records: List[SampleRecord] = []
    for item_id, group in item_to_records.items():
        if item_id in val_item_ids:
            val_records.extend(group)
        else:
            train_records.extend(group)
    return train_records, val_records


def filter_train_records_for_triplet(records: List[SampleRecord]) -> List[SampleRecord]:
    counts: Dict[str, int] = {}
    for r in records:
        counts[r.item_id] = counts.get(r.item_id, 0) + 1
    return [r for r in records if counts[r.item_id] >= 2]


class ImageWithLabelDataset(Dataset):
    def __init__(self, records: List[SampleRecord], transform):
        self.records = records
        self.transform = transform

        self.item_id_strings = [r.item_id for r in records]
        unique_ids = sorted(set(self.item_id_strings))
        self.item_to_label = {item_id: i for i, item_id in enumerate(unique_ids)}
        self.labels = [self.item_to_label[item_id] for item_id in self.item_id_strings]

        self.label_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self.labels):
            self.label_to_indices.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.records[idx].path).convert("RGB"))
        return image, self.labels[idx]


class PKBatchSampler(Sampler[List[int]]):
    """Samples batches with P labels and K images per label."""

    def __init__(
        self,
        label_to_indices: Dict[int, List[int]],
        p_per_batch: int,
        k_per_item: int,
        steps_per_epoch: int,
    ):
        self.label_to_indices = {k: v for k, v in label_to_indices.items() if len(v) >= 2}
        if len(self.label_to_indices) < p_per_batch:
            raise ValueError(
                f"Need at least {p_per_batch} labels with >=2 images; found {len(self.label_to_indices)}."
            )
        self.labels = list(self.label_to_indices.keys())
        self.p_per_batch = p_per_batch
        self.k_per_item = k_per_item
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.steps_per_epoch):
            chosen_labels = random.sample(self.labels, self.p_per_batch)
            batch_indices: List[int] = []
            for label in chosen_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.k_per_item:
                    picked = random.sample(indices, self.k_per_item)
                else:
                    picked = random.choices(indices, k=self.k_per_item)
                batch_indices.extend(picked)
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.steps_per_epoch


class MetricModel(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, embed_dim: int, num_classes: int):
        super().__init__()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
        else:
            raise ValueError("Unsupported backbone")

        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone_name = backbone
        self.backbone = base
        self.embed_head = nn.Linear(in_features, embed_dim)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        embeds = self.embed_head(feats)
        logits = self.classifier(feats)
        return embeds, logits


def set_trainable_stage(model: MetricModel, stage: int) -> None:
    """
    stage 0: train layer4 + heads
    stage 1: train layer3 + layer4 + heads
    stage 2: train full backbone + heads
    """
    for p in model.backbone.parameters():
        p.requires_grad = False

    if stage >= 2:
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        if stage >= 1:
            for p in model.backbone.layer3.parameters():
                p.requires_grad = True
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True

    for p in model.embed_head.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def build_optimizer(model: MetricModel) -> optim.Optimizer:
    groups = [
        {"params": list(model.backbone.parameters()), "lr": BASE_LR},
        {"params": list(model.embed_head.parameters()), "lr": HEAD_LR},
        {"params": list(model.classifier.parameters()), "lr": HEAD_LR},
    ]
    return optim.AdamW(groups, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=1)


def batch_hard_semihard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    sim = embeddings @ embeddings.T
    dist = 1 - sim
    losses = []
    n = embeddings.shape[0]

    for i in range(n):
        label_i = labels[i]
        pos_mask = labels == label_i
        pos_mask[i] = False
        neg_mask = labels != label_i

        if not torch.any(pos_mask) or not torch.any(neg_mask):
            continue

        d_ap = dist[i][pos_mask].max()
        neg_dists = dist[i][neg_mask]
        semi_hard = neg_dists[(neg_dists > d_ap) & (neg_dists < d_ap + margin)]

        if semi_hard.numel() > 0:
            d_an = semi_hard.min()
        else:
            d_an = neg_dists.min()

        losses.append(torch.relu(d_ap - d_an + margin))

    if not losses:
        return torch.zeros((), device=embeddings.device, requires_grad=True)
    return torch.stack(losses).mean()


def make_loader(dataset: Dataset, batch_sampler, num_workers: int) -> DataLoader:
    try:
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    except RuntimeError as exc:
        if num_workers > 0 and ("torch_shm_manager" in str(exc) or "Operation not permitted" in str(exc)):
            print("Falling back to num_workers=0 due to shared-memory restrictions.")
            return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
        raise


def compute_retrieval_metrics(
    embeddings: torch.Tensor, labels: torch.Tensor, ks: Tuple[int, int] = (1, 5)
) -> Dict[str, float]:
    sim = embeddings @ embeddings.T
    n = embeddings.shape[0]
    sim.fill_diagonal_(-float("inf"))

    max_k = min(max(ks), max(1, n - 1))
    sorted_idx = torch.argsort(sim, dim=1, descending=True)
    topk_idx = sorted_idx[:, :max_k]

    recalls = {k: 0.0 for k in ks}
    ap_values: List[float] = []
    valid_queries = 0

    for i in range(n):
        positives = labels == labels[i]
        positives[i] = False
        if int(positives.sum().item()) == 0:
            continue
        valid_queries += 1

        for k in ks:
            k_eff = min(k, max_k)
            hit = positives[topk_idx[i, :k_eff]].any()
            recalls[k] += float(hit.item())

        ranked = sorted_idx[i]
        rel = positives[ranked].float()
        if rel.sum() > 0:
            cumsum_rel = torch.cumsum(rel, dim=0)
            ranks = torch.arange(1, rel.shape[0] + 1, device=rel.device, dtype=torch.float32)
            precision_at_r = cumsum_rel / ranks
            ap = (precision_at_r * rel).sum() / rel.sum()
            ap_values.append(float(ap.item()))

    if valid_queries == 0:
        return {"valid_queries": 0.0, "recall@1": 0.0, "recall@5": 0.0, "mAP": 0.0}

    metrics = {"valid_queries": float(valid_queries)}
    for k in ks:
        metrics[f"recall@{k}"] = recalls[k] / valid_queries
    metrics["mAP"] = float(np.mean(ap_values)) if ap_values else 0.0
    return metrics


def evaluate_retrieval(model: MetricModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_z: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            z, _ = model(images)
            z = l2_normalize(z).cpu()
            all_z.append(z)
            all_labels.append(labels.cpu())

    if not all_z:
        return {"valid_queries": 0.0, "recall@1": 0.0, "recall@5": 0.0, "mAP": 0.0}
    embeddings = torch.cat(all_z, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return compute_retrieval_metrics(embeddings, labels, ks=(1, 5))


def margin_for_epoch(epoch: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return end
    t = (epoch - 1) / (total_epochs - 1)
    return start + t * (end - start)


def run_training() -> None:
    image_root = Path(IMAGE_DIR).expanduser()
    if "REPLACE_WITH_REAL_FOLDER" in IMAGE_DIR or str(image_root).endswith("REPLACE_WITH_REAL_FOLDER"):
        raise ValueError("Set IMAGE_DIR to a real path before running.")
    if not image_root.exists():
        raise FileNotFoundError(f"IMAGE_DIR not found: {image_root}")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if MANIFEST_CSV is not None:
        manifest_path = Path(MANIFEST_CSV).expanduser()
        if not manifest_path.exists():
            raise FileNotFoundError(f"MANIFEST_CSV not found: {manifest_path}")
        records = load_manifest_records(manifest_path, image_root)
        print(f"Loaded manifest records: {len(records)} from {manifest_path}")
    else:
        paths = list_images(image_root)
        if not paths:
            raise ValueError(f"No images found in: {image_root}")
        print(f"Found images: {len(paths)}")
        records = build_records_from_files(paths, ITEM_ID_MODE)

    train_records, val_records = split_records(records, VAL_FRACTION, SEED)
    train_records = filter_train_records_for_triplet(train_records)
    print(f"Split: train images={len(train_records)} | val images={len(val_records)}")

    train_counts: Dict[str, int] = {}
    for r in train_records:
        train_counts[r.item_id] = train_counts.get(r.item_id, 0) + 1
    train_multi_view = sum(1 for c in train_counts.values() if c > 1)
    print(f"Train unique items: {len(train_counts)} | train multi-view items: {train_multi_view}")

    if train_multi_view == 0:
        raise ValueError("No items with multiple images to form positives.")
    if len(train_counts) < 50:
        print("Warning: very few identities in training set; retrieval quality will likely be poor.")

    batch_size = P_PER_BATCH * K_PER_ITEM
    print(
        f"Batch strategy: P={P_PER_BATCH}, K={K_PER_ITEM}, "
        f"batch_size={batch_size}, steps/epoch={STEPS_PER_EPOCH}"
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.015),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageWithLabelDataset(train_records, train_transform)
    batch_sampler = PKBatchSampler(
        label_to_indices=train_ds.label_to_indices,
        p_per_batch=P_PER_BATCH,
        k_per_item=K_PER_ITEM,
        steps_per_epoch=STEPS_PER_EPOCH,
    )
    train_loader = make_loader(train_ds, batch_sampler, NUM_WORKERS)

    val_loader = None
    if val_records:
        val_ds = ImageWithLabelDataset(val_records, eval_transform)
        val_loader = DataLoader(
            val_ds,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    num_classes = len(train_ds.item_to_label)
    model = MetricModel(BACKBONE, PRETRAINED, EMBED_DIM, num_classes)
    current_stage = 0
    set_trainable_stage(model, current_stage)
    model.to(device)

    out_model = Path(OUT_MODEL).expanduser()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    best_model = out_model.with_name(f"{out_model.stem}_best{out_model.suffix}")

    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS), eta_min=HEAD_LR * 0.05
    )
    ce_loss = nn.CrossEntropyLoss()

    if BEST_METRIC not in {"mAP", "recall@1"}:
        raise ValueError("BEST_METRIC must be 'mAP' or 'recall@1'.")

    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    stopped_early = False
    baseline_metrics = None

    if val_loader is not None and EVAL_FROZEN_BASELINE:
        baseline_metrics = evaluate_retrieval(model, val_loader, device)
        print(
            "Frozen ImageNet baseline | "
            f"queries={int(baseline_metrics['valid_queries'])} | "
            f"R@1={baseline_metrics['recall@1']:.4f} | "
            f"R@5={baseline_metrics['recall@5']:.4f} | "
            f"mAP={baseline_metrics['mAP']:.4f}"
        )
        model.train()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        target_stage = 0
        if epoch >= UNFREEZE_ALL_EPOCH:
            target_stage = 2
        elif epoch >= UNFREEZE_LAYER3_EPOCH:
            target_stage = 1
        if target_stage != current_stage:
            current_stage = target_stage
            set_trainable_stage(model, current_stage)
            print(f"Stage update at epoch {epoch}: training stage={current_stage}")

        total_loss = 0.0
        total_triplet = 0.0
        total_ce = 0.0

        num_batches = len(train_loader)
        margin_now = margin_for_epoch(epoch, EPOCHS, MARGIN_START, MARGIN_END)

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            z, logits = model(images)
            z = l2_normalize(z)

            t_loss = batch_hard_semihard_triplet_loss(z, labels, margin_now)
            c_loss = ce_loss(logits, labels)
            loss = TRIPLET_WEIGHT * t_loss + CE_WEIGHT * c_loss

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_triplet += float(t_loss.item())
            total_ce += float(c_loss.item())

            avg_loss = total_loss / step
            avg_t = total_triplet / step
            avg_c = total_ce / step
            print(
                f"\rEpoch {epoch:03d}/{EPOCHS:03d} | "
                f"Batch {step:04d}/{num_batches:04d} | "
                f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | "
                f"T: {avg_t:.4f} | CE: {avg_c:.4f}",
                end="",
                flush=True,
            )

        print()
        train_loss = total_loss / max(1, num_batches)
        print(
            f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | "
            f"LR(backbone/head): {BASE_LR:.2e}/{HEAD_LR:.2e} | Margin: {margin_now:.4f}"
        )

        if val_loader is not None:
            metrics = evaluate_retrieval(model, val_loader, device)
            print(
                "Validation retrieval | "
                f"queries={int(metrics['valid_queries'])} | "
                f"R@1={metrics['recall@1']:.4f} | "
                f"R@5={metrics['recall@5']:.4f} | "
                f"mAP={metrics['mAP']:.4f}"
            )

            score = metrics[BEST_METRIC]
            if score > best_score + EARLY_STOPPING_MIN_DELTA:
                best_score = score
                best_epoch = epoch
                epochs_without_improve = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "backbone": BACKBONE,
                        "embed_dim": EMBED_DIM,
                        "item_id_mode": ITEM_ID_MODE,
                        "image_dir": str(image_root),
                        "manifest_csv": str(MANIFEST_CSV) if MANIFEST_CSV else None,
                        "epoch": epoch,
                        "val_metrics": metrics,
                        "best_metric": BEST_METRIC,
                        "best_metric_value": best_score,
                    },
                    best_model,
                )
                print(
                    f"New best model saved at epoch {epoch}: "
                    f"{BEST_METRIC}={score:.4f} -> {best_model}"
                )
            else:
                epochs_without_improve += 1
                print(
                    f"No {BEST_METRIC} improvement for "
                    f"{epochs_without_improve}/{EARLY_STOPPING_PATIENCE} epoch(s)."
                )
            model.train()

        scheduler.step()

        if val_loader is not None and epochs_without_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best {BEST_METRIC}={best_score:.4f} at epoch {best_epoch}."
            )
            stopped_early = True
            break

    if val_loader is not None and best_epoch > 0:
        print(f"Best checkpoint: {best_model} (epoch {best_epoch}, {BEST_METRIC}={best_score:.4f})")
        if baseline_metrics is not None:
            baseline_score = baseline_metrics[BEST_METRIC]
            delta_score = best_score - baseline_score
            print(
                "Best-vs-frozen baseline | "
                f"Frozen {BEST_METRIC}={baseline_score:.4f} -> "
                f"Best {BEST_METRIC}={best_score:.4f} | Delta={delta_score:+.4f}"
            )

    if val_loader is None:
        print("Validation split is empty; skipped best-checkpoint tracking and early stopping.")

    if stopped_early:
        print(f"Training ended early based on validation {BEST_METRIC} plateau.")

    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": BACKBONE,
            "embed_dim": EMBED_DIM,
            "item_id_mode": ITEM_ID_MODE,
            "image_dir": str(image_root),
            "manifest_csv": str(MANIFEST_CSV) if MANIFEST_CSV else None,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_metric": BEST_METRIC,
            "best_metric_value": best_score if best_epoch > 0 else None,
        },
        out_model,
    )
    print(f"Saved last-epoch model: {out_model}")


if __name__ == "__main__":
    run_training()
