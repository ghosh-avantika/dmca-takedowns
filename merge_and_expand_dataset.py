"""
Merge negative cases with existing dataset and expand to 500 total samples.
This script:
1. Loads existing dataset (236 samples)
2. Adds negative "original" cases (43 samples)
3. Generates CLIP embeddings for new cases
4. Applies data augmentation to reach 500 samples
5. Creates new train/val/test splits
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import clip
from PIL import Image

print("="*80)
print("MERGE AND EXPAND DATASET TO 500 SAMPLES")
print("="*80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 1. LOAD EXISTING DATA
# ============================================================================
print("\n" + "="*80)
print("1. LOADING EXISTING DATA")
print("="*80)

# Load existing dataset
existing_data = torch.load('dataset_splits_WITH_PSEUDO_LABELS.pt')

# Combine all existing embeddings and labels
all_embeddings = []
all_labels = []

for split in ['train', 'val', 'test']:
    all_embeddings.append(existing_data[split]['embeddings'])
    all_labels.append(existing_data[split]['labels'])

existing_embeddings = torch.cat(all_embeddings, dim=0)
existing_labels = torch.cat(all_labels, dim=0)

print(f"  Existing samples: {len(existing_embeddings)}")
print(f"  Existing knockoff: {(existing_labels == 0).sum().item()}")
print(f"  Existing similar: {(existing_labels == 1).sum().item()}")

# ============================================================================
# 2. LOAD AND PROCESS NEGATIVE CASES
# ============================================================================
print("\n" + "="*80)
print("2. PROCESSING NEGATIVE (ORIGINAL) CASES")
print("="*80)

# Load negative cases
negative_df = pd.read_csv('negative_cases_originals.csv')
print(f"  Loaded {len(negative_df)} negative cases from CSV")

# Load CLIP model
print("  Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"  Using device: {device}")

# Generate embeddings for negative cases
print("  Generating CLIP embeddings for negative cases...")

negative_embeddings = []
for idx, row in negative_df.iterrows():
    # Create description text for CLIP (combine designer, item type, and design elements)
    description = f"{row['original_designer_name']} {row['original_item_type']}: {row['original_design_elements']}"

    # Tokenize and encode
    text = clip.tokenize([description], truncate=True).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    negative_embeddings.append(text_features.cpu())

negative_embeddings = torch.cat(negative_embeddings, dim=0)

# For negative/original cases, we'll add a new label: 2 = "original"
# But to keep it binary (knockoff vs not-knockoff), we'll label originals as "similar" (1)
# Since they are NOT knockoffs
negative_labels = torch.ones(len(negative_embeddings), dtype=torch.long)  # Label 1 = similar/not-knockoff

print(f"  Generated {len(negative_embeddings)} embeddings")
print(f"  Embedding shape: {negative_embeddings.shape}")

# ============================================================================
# 3. COMBINE DATASETS
# ============================================================================
print("\n" + "="*80)
print("3. COMBINING DATASETS")
print("="*80)

combined_embeddings = torch.cat([existing_embeddings, negative_embeddings], dim=0)
combined_labels = torch.cat([existing_labels, negative_labels], dim=0)

print(f"  Combined samples: {len(combined_embeddings)}")
print(f"  Knockoff (0): {(combined_labels == 0).sum().item()}")
print(f"  Similar/Original (1): {(combined_labels == 1).sum().item()}")

# ============================================================================
# 4. DATA AUGMENTATION TO REACH 500 SAMPLES
# ============================================================================
print("\n" + "="*80)
print("4. DATA AUGMENTATION TO 500 SAMPLES")
print("="*80)

current_count = len(combined_embeddings)
target_count = 500
samples_needed = target_count - current_count

print(f"  Current samples: {current_count}")
print(f"  Target samples: {target_count}")
print(f"  Samples to generate: {samples_needed}")

# Augmentation strategies
np.random.seed(42)
augmented_embeddings = []
augmented_labels = []

# Get class counts
knockoff_indices = torch.where(combined_labels == 0)[0].numpy()
similar_indices = torch.where(combined_labels == 1)[0].numpy()

# Balance augmentation between classes
knockoff_count = len(knockoff_indices)
similar_count = len(similar_indices)

# Calculate how many to augment per class (roughly balanced)
knockoff_augment = int(samples_needed * knockoff_count / current_count)
similar_augment = samples_needed - knockoff_augment

print(f"\n  Augmentation plan:")
print(f"    Knockoff samples to generate: {knockoff_augment}")
print(f"    Similar samples to generate: {similar_augment}")

def augment_embedding(embedding, noise_scale=0.02):
    """Add small Gaussian noise to embedding"""
    noise = torch.randn_like(embedding) * noise_scale
    augmented = embedding + noise
    # Re-normalize
    augmented = augmented / augmented.norm(dim=-1, keepdim=True)
    return augmented

def mixup_embeddings(emb1, emb2, alpha=0.2):
    """Mixup two embeddings with small alpha"""
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure we stay close to one class
    mixed = lam * emb1 + (1 - lam) * emb2
    # Re-normalize
    mixed = mixed / mixed.norm(dim=-1, keepdim=True)
    return mixed

def interpolate_embeddings(emb1, emb2, t=None):
    """SMOTE-like interpolation between two embeddings"""
    if t is None:
        t = np.random.uniform(0.1, 0.4)  # Small interpolation
    interpolated = emb1 + t * (emb2 - emb1)
    # Re-normalize
    interpolated = interpolated / interpolated.norm(dim=-1, keepdim=True)
    return interpolated

# Augment knockoff class
print("\n  Generating knockoff augmentations...")
for i in range(knockoff_augment):
    # Choose augmentation strategy
    strategy = np.random.choice(['noise', 'mixup', 'interpolate'], p=[0.5, 0.25, 0.25])

    if strategy == 'noise':
        idx = np.random.choice(knockoff_indices)
        aug_emb = augment_embedding(combined_embeddings[idx:idx+1])
    elif strategy == 'mixup':
        idx1, idx2 = np.random.choice(knockoff_indices, size=2, replace=False)
        aug_emb = mixup_embeddings(combined_embeddings[idx1:idx1+1], combined_embeddings[idx2:idx2+1])
    else:  # interpolate
        idx1, idx2 = np.random.choice(knockoff_indices, size=2, replace=False)
        aug_emb = interpolate_embeddings(combined_embeddings[idx1:idx1+1], combined_embeddings[idx2:idx2+1])

    augmented_embeddings.append(aug_emb)
    augmented_labels.append(0)

# Augment similar class
print("  Generating similar augmentations...")
for i in range(similar_augment):
    # Choose augmentation strategy
    strategy = np.random.choice(['noise', 'mixup', 'interpolate'], p=[0.5, 0.25, 0.25])

    if strategy == 'noise':
        idx = np.random.choice(similar_indices)
        aug_emb = augment_embedding(combined_embeddings[idx:idx+1])
    elif strategy == 'mixup':
        idx1, idx2 = np.random.choice(similar_indices, size=2, replace=False)
        aug_emb = mixup_embeddings(combined_embeddings[idx1:idx1+1], combined_embeddings[idx2:idx2+1])
    else:  # interpolate
        idx1, idx2 = np.random.choice(similar_indices, size=2, replace=False)
        aug_emb = interpolate_embeddings(combined_embeddings[idx1:idx1+1], combined_embeddings[idx2:idx2+1])

    augmented_embeddings.append(aug_emb)
    augmented_labels.append(1)

# Combine with augmented data
if augmented_embeddings:
    augmented_embeddings = torch.cat(augmented_embeddings, dim=0)
    augmented_labels = torch.tensor(augmented_labels, dtype=torch.long)

    final_embeddings = torch.cat([combined_embeddings, augmented_embeddings], dim=0)
    final_labels = torch.cat([combined_labels, augmented_labels], dim=0)
else:
    final_embeddings = combined_embeddings
    final_labels = combined_labels

print(f"\n  Final dataset size: {len(final_embeddings)}")
print(f"  Knockoff (0): {(final_labels == 0).sum().item()}")
print(f"  Similar (1): {(final_labels == 1).sum().item()}")

# ============================================================================
# 5. CREATE NEW TRAIN/VAL/TEST SPLITS
# ============================================================================
print("\n" + "="*80)
print("5. CREATING TRAIN/VAL/TEST SPLITS")
print("="*80)

from sklearn.model_selection import train_test_split

# Convert to numpy for sklearn
X = final_embeddings.numpy()
y = final_labels.numpy()

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Training:   {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Testing:    {len(X_test)} samples")
print(f"  Total:      {len(X_train) + len(X_val) + len(X_test)} samples")

print(f"\n  Class distribution:")
print(f"    Train - knockoff: {(y_train == 0).sum()}, similar: {(y_train == 1).sum()}")
print(f"    Val   - knockoff: {(y_val == 0).sum()}, similar: {(y_val == 1).sum()}")
print(f"    Test  - knockoff: {(y_test == 0).sum()}, similar: {(y_test == 1).sum()}")

# ============================================================================
# 6. SAVE EXPANDED DATASET
# ============================================================================
print("\n" + "="*80)
print("6. SAVING EXPANDED DATASET")
print("="*80)

label_to_idx = {'knockoff': 0, 'similar': 1}
idx_to_label = {0: 'knockoff', 1: 'similar'}

expanded_dataset = {
    'train': {
        'embeddings': torch.from_numpy(X_train).float(),
        'labels': torch.from_numpy(y_train).long()
    },
    'val': {
        'embeddings': torch.from_numpy(X_val).float(),
        'labels': torch.from_numpy(y_val).long()
    },
    'test': {
        'embeddings': torch.from_numpy(X_test).float(),
        'labels': torch.from_numpy(y_test).long()
    },
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'metadata': {
        'total_samples': len(final_embeddings),
        'original_samples': current_count,
        'augmented_samples': samples_needed,
        'negative_cases_added': len(negative_df),
        'creation_date': datetime.now().isoformat()
    }
}

torch.save(expanded_dataset, 'dataset_500_samples.pt')
print(f"  Saved to: dataset_500_samples.pt")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DATASET EXPANSION COMPLETE")
print("="*80)
print(f"""
Dataset Summary:
  Original samples:     {len(existing_embeddings)}
  Negative cases added: {len(negative_df)}
  Augmented samples:    {samples_needed}
  Final total:          {len(final_embeddings)}

Split Distribution:
  Training:   {len(X_train)} samples ({len(X_train)/len(final_embeddings)*100:.1f}%)
  Validation: {len(X_val)} samples ({len(X_val)/len(final_embeddings)*100:.1f}%)
  Testing:    {len(X_test)} samples ({len(X_test)/len(final_embeddings)*100:.1f}%)

Class Balance:
  Knockoff: {(final_labels == 0).sum().item()} ({(final_labels == 0).sum().item()/len(final_labels)*100:.1f}%)
  Similar:  {(final_labels == 1).sum().item()} ({(final_labels == 1).sum().item()/len(final_labels)*100:.1f}%)

Files:
  - dataset_500_samples.pt (new expanded dataset)

Next Steps:
  1. Retrain MLP classifier: python3 train_mlp_classifier_500.py
  2. Run baseline experiment: python3 baseline_experiment_500.py
""")