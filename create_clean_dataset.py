"""
Create a clean dataset split where:
- Training: real samples + augmented samples
- Validation: ONLY real samples (no augmentation leakage)
- Test: ONLY real samples (no augmentation leakage)

This ensures test accuracy reflects true generalization.
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

print("="*80)
print("CREATE CLEAN DATASET (Real-Only Val/Test)")
print("="*80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 1. LOAD ORIGINAL REAL DATA
# ============================================================================
print("\n" + "="*80)
print("1. LOADING ORIGINAL REAL DATA")
print("="*80)

# Load the original dataset with pseudo-labels (contains only real samples)
original_data = torch.load('dataset_splits_WITH_PSEUDO_LABELS.pt')

# Combine all real samples
all_real_embeddings = []
all_real_labels = []

for split in ['train', 'val', 'test']:
    all_real_embeddings.append(original_data[split]['embeddings'])
    all_real_labels.append(original_data[split]['labels'])

real_embeddings = torch.cat(all_real_embeddings, dim=0)
real_labels = torch.cat(all_real_labels, dim=0)

print(f"  Total real samples: {len(real_embeddings)}")
print(f"  Real knockoff: {(real_labels == 0).sum().item()}")
print(f"  Real similar: {(real_labels == 1).sum().item()}")

# Also load negative cases if they exist
try:
    import pandas as pd
    import clip

    negative_df = pd.read_csv('negative_cases_originals.csv')
    print(f"\n  Loading {len(negative_df)} negative cases...")

    device = "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    negative_embeddings = []
    for idx, row in negative_df.iterrows():
        description = f"{row['original_designer_name']} {row['original_item_type']}: {row['original_design_elements']}"
        text = clip.tokenize([description], truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        negative_embeddings.append(text_features.cpu())

    negative_embeddings = torch.cat(negative_embeddings, dim=0)
    negative_labels = torch.ones(len(negative_embeddings), dtype=torch.long)  # similar = 1

    # Combine with real data
    real_embeddings = torch.cat([real_embeddings, negative_embeddings], dim=0)
    real_labels = torch.cat([real_labels, negative_labels], dim=0)

    print(f"  Added {len(negative_df)} negative cases")
    print(f"  Total real samples now: {len(real_embeddings)}")
except Exception as e:
    print(f"  Note: Could not load negative cases ({e})")

print(f"\n  Final real data:")
print(f"    Total: {len(real_embeddings)}")
print(f"    Knockoff (0): {(real_labels == 0).sum().item()}")
print(f"    Similar (1): {(real_labels == 1).sum().item()}")

# ============================================================================
# 2. SPLIT REAL DATA (Val/Test are ONLY real)
# ============================================================================
print("\n" + "="*80)
print("2. SPLITTING REAL DATA")
print("="*80)

X = real_embeddings.numpy()
y = real_labels.numpy()

# First split: 70% train, 30% temp (for val+test)
X_train_real, X_temp, y_train_real, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Second split: 50/50 of temp -> 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Real train samples: {len(X_train_real)}")
print(f"  Real val samples:   {len(X_val)}")
print(f"  Real test samples:  {len(X_test)}")

# ============================================================================
# 3. AUGMENT TRAINING DATA ONLY
# ============================================================================
print("\n" + "="*80)
print("3. AUGMENTING TRAINING DATA ONLY")
print("="*80)

np.random.seed(42)

def augment_embedding(embedding, noise_scale=0.02):
    """Add small Gaussian noise to embedding"""
    noise = np.random.randn(*embedding.shape) * noise_scale
    augmented = embedding + noise
    augmented = augmented / np.linalg.norm(augmented)
    return augmented

def mixup_embeddings(emb1, emb2, alpha=0.2):
    """Mixup two embeddings"""
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    mixed = lam * emb1 + (1 - lam) * emb2
    mixed = mixed / np.linalg.norm(mixed)
    return mixed

def interpolate_embeddings(emb1, emb2):
    """SMOTE-like interpolation"""
    t = np.random.uniform(0.1, 0.4)
    interpolated = emb1 + t * (emb2 - emb1)
    interpolated = interpolated / np.linalg.norm(interpolated)
    return interpolated

# Get indices by class
knockoff_indices = np.where(y_train_real == 0)[0]
similar_indices = np.where(y_train_real == 1)[0]

# Target: augment to ~350 training samples (to have meaningful augmentation)
current_train = len(X_train_real)
target_train = 400
samples_to_generate = target_train - current_train

print(f"  Current training samples: {current_train}")
print(f"  Target training samples: {target_train}")
print(f"  Samples to augment: {samples_to_generate}")

# Balance augmentation between classes
knockoff_aug = int(samples_to_generate * len(knockoff_indices) / current_train)
similar_aug = samples_to_generate - knockoff_aug

augmented_X = []
augmented_y = []

# Augment knockoff class
print(f"\n  Generating {knockoff_aug} knockoff augmentations...")
for i in range(knockoff_aug):
    strategy = np.random.choice(['noise', 'mixup', 'interpolate'], p=[0.5, 0.25, 0.25])
    if strategy == 'noise':
        idx = np.random.choice(knockoff_indices)
        aug = augment_embedding(X_train_real[idx])
    elif strategy == 'mixup':
        idx1, idx2 = np.random.choice(knockoff_indices, size=2, replace=False)
        aug = mixup_embeddings(X_train_real[idx1], X_train_real[idx2])
    else:
        idx1, idx2 = np.random.choice(knockoff_indices, size=2, replace=False)
        aug = interpolate_embeddings(X_train_real[idx1], X_train_real[idx2])
    augmented_X.append(aug)
    augmented_y.append(0)

# Augment similar class
print(f"  Generating {similar_aug} similar augmentations...")
for i in range(similar_aug):
    strategy = np.random.choice(['noise', 'mixup', 'interpolate'], p=[0.5, 0.25, 0.25])
    if strategy == 'noise':
        idx = np.random.choice(similar_indices)
        aug = augment_embedding(X_train_real[idx])
    elif strategy == 'mixup':
        idx1, idx2 = np.random.choice(similar_indices, size=2, replace=False)
        aug = mixup_embeddings(X_train_real[idx1], X_train_real[idx2])
    else:
        idx1, idx2 = np.random.choice(similar_indices, size=2, replace=False)
        aug = interpolate_embeddings(X_train_real[idx1], X_train_real[idx2])
    augmented_X.append(aug)
    augmented_y.append(1)

# Combine real + augmented for training
augmented_X = np.array(augmented_X)
augmented_y = np.array(augmented_y)

X_train_final = np.vstack([X_train_real, augmented_X])
y_train_final = np.concatenate([y_train_real, augmented_y])

print(f"\n  Final training set: {len(X_train_final)} samples")
print(f"    Real: {len(X_train_real)}")
print(f"    Augmented: {len(augmented_X)}")

# ============================================================================
# 4. SAVE CLEAN DATASET
# ============================================================================
print("\n" + "="*80)
print("4. SAVING CLEAN DATASET")
print("="*80)

label_to_idx = {'knockoff': 0, 'similar': 1}
idx_to_label = {0: 'knockoff', 1: 'similar'}

clean_dataset = {
    'train': {
        'embeddings': torch.from_numpy(X_train_final).float(),
        'labels': torch.from_numpy(y_train_final).long()
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
        'total_real_samples': len(real_embeddings),
        'train_real': len(X_train_real),
        'train_augmented': len(augmented_X),
        'train_total': len(X_train_final),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'val_test_are_real_only': True,
        'creation_date': datetime.now().isoformat()
    }
}

torch.save(clean_dataset, 'dataset_clean_real_test.pt')
print(f"  Saved to: dataset_clean_real_test.pt")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("CLEAN DATASET CREATED")
print("="*80)
print(f"""
Dataset Summary:

  Training Set: {len(X_train_final)} samples
    - Real samples: {len(X_train_real)} ({len(X_train_real)/len(X_train_final)*100:.1f}%)
    - Augmented: {len(augmented_X)} ({len(augmented_X)/len(X_train_final)*100:.1f}%)
    - Knockoff: {(y_train_final == 0).sum()}
    - Similar: {(y_train_final == 1).sum()}

  Validation Set: {len(X_val)} samples (100% REAL)
    - Knockoff: {(y_val == 0).sum()}
    - Similar: {(y_val == 1).sum()}

  Test Set: {len(X_test)} samples (100% REAL)
    - Knockoff: {(y_test == 0).sum()}
    - Similar: {(y_test == 1).sum()}

Key Property:
  - Val and Test sets contain ONLY real samples
  - No augmentation leakage between train and test
  - Test accuracy will reflect true generalization

Next Steps:
  1. Update baseline_experiment.py to use 'dataset_clean_real_test.pt'
  2. Re-run: python3 baseline_experiment.py
""")