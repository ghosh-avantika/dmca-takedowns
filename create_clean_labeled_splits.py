"""
Create train/val/test splits using similarity-based pseudo-labeling
Auto-labels scraped cases based on similarity to gold standard cases
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("="*80)
print("CREATING CLEAN LABELED DATASET")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")

data = torch.load('stabilized_text_embeddings.pt')
embeddings = data['embeddings']
scraped_count = data['scraped_count']
gold_count = data['gold_count']

# Load CSVs
scraped_df = pd.read_csv('scraped_cases_text_only_CLEAN.csv')
gold_df = pd.read_csv('gold-standard-cases-CORRECTED.csv')
gold_df = gold_df.dropna(how='all')

print(f"   Total embeddings: {len(embeddings)}")
print(f"   - Scraped: {scraped_count}")
print(f"   - Gold: {gold_count}")

# ============================================================================
# 2. EXTRACT LABELS (GOLD STANDARD ONLY)
# ============================================================================
print("\n2. Extracting labels from gold standard...")

gold_labels = []
gold_texts = []

for idx in range(min(gold_count, len(gold_df))):
    row = gold_df.iloc[idx]
    label = None
    for col in ['infringement_label', 'label']:
        if col in row and pd.notna(row[col]):
            label = str(row[col]).strip().lower()
            break

    text = ''
    for col in ['original_design_elements', 'elements']:
        if col in row and pd.notna(row[col]):
            text = str(row[col])
            break

    # Only keep if has valid label
    if label in ['knockoff', 'similar']:
        gold_labels.append(label)
        gold_texts.append(text)

print(f"   Found {len(gold_labels)} properly labeled cases:")
print(f"   - knockoff: {gold_labels.count('knockoff')}")
print(f"   - similar: {gold_labels.count('similar')}")

# Create label mappings for binary classification
label_to_idx = {'knockoff': 0, 'similar': 1}
idx_to_label = {0: 'knockoff', 1: 'similar'}

# Get corresponding embeddings (last gold_count from embeddings)
gold_start = scraped_count
gold_embeddings = embeddings[gold_start:gold_start+len(gold_labels)]

print(f"\n   Gold embeddings shape: {gold_embeddings.shape}")

# ============================================================================
# 4. SIMILARITY-BASED PSEUDO-LABELING
# ============================================================================
print("\n" + "="*80)
print("SIMILARITY-BASED PSEUDO-LABELING")
print("="*80)

print("\nUsing embeddings similarity to label 'unknown' cases...")

# Compute similarity between all embeddings and gold standard
all_embeddings = embeddings[:scraped_count + len(gold_labels)]  # Trim to actual data
similarity_matrix = (all_embeddings @ all_embeddings.T).numpy()

# For each unknown case (scraped), find most similar labeled case
pseudo_labels = []
pseudo_confidences = []

for i in range(scraped_count):
    # Get similarities to all gold standard cases
    sims_to_gold = similarity_matrix[i, scraped_count:scraped_count+len(gold_labels)]

    # Find most similar gold case
    most_similar_idx = np.argmax(sims_to_gold)
    max_similarity = sims_to_gold[most_similar_idx]

    # Assign label based on similarity threshold
    if max_similarity > 0.75:  # High similarity threshold
        pseudo_label = gold_labels[most_similar_idx]
        pseudo_labels.append(pseudo_label)
        pseudo_confidences.append(max_similarity)
    else:
        pseudo_labels.append(None)  # Too uncertain
        pseudo_confidences.append(max_similarity)

# Count pseudo-labeled cases
valid_pseudo = [l for l in pseudo_labels if l is not None]
print(f"\n Pseudo-labeled {len(valid_pseudo)} / {scraped_count} scraped cases:")
print(f"  - knockoff: {valid_pseudo.count('knockoff')}")
print(f"  - similar: {valid_pseudo.count('similar')}")
print(f"  - too uncertain: {pseudo_labels.count(None)}")

# Combine with gold standard
combined_labels = pseudo_labels + gold_labels
combined_texts = ['scraped_case'] * scraped_count + gold_texts
combined_embeddings = all_embeddings

# Filter out None labels
valid_indices = [i for i, label in enumerate(combined_labels) if label is not None]

if len(valid_indices) > 20:
    filtered_embeddings = combined_embeddings[valid_indices]
    filtered_labels = [combined_labels[i] for i in valid_indices]

    print(f"\nCombined dataset: {len(filtered_labels)} samples")
    print(f"  - knockoff: {filtered_labels.count('knockoff')}")
    print(f"  - similar: {filtered_labels.count('similar')}")

    # Convert and split
    X_combined = filtered_embeddings.numpy()
    y_combined = np.array([label_to_idx[label] for label in filtered_labels])

    X_train_b, X_temp_b, y_train_b, y_temp_b = train_test_split(
        X_combined, y_combined,
        test_size=0.30,
        random_state=42,
        stratify=y_combined
    )

    X_val_b, X_test_b, y_val_b, y_test_b = train_test_split(
        X_temp_b, y_temp_b,
        test_size=0.50,
        random_state=42
    )

    print(f"\nSplit sizes (with pseudo-labels):")
    print(f"  Training:   {len(X_train_b)} samples")
    print(f"  Validation: {len(X_val_b)} samples")
    print(f"  Testing:    {len(X_test_b)} samples")

    # Save pseudo-labeled dataset
    pseudo_dataset = {
        'train': {
            'embeddings': torch.from_numpy(X_train_b),
            'labels': torch.from_numpy(y_train_b),
            'size': len(X_train_b)
        },
        'val': {
            'embeddings': torch.from_numpy(X_val_b),
            'labels': torch.from_numpy(y_val_b),
            'size': len(X_val_b)
        },
        'test': {
            'embeddings': torch.from_numpy(X_test_b),
            'labels': torch.from_numpy(y_test_b),
            'size': len(X_test_b)
        },
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'num_classes': 2,
        'feature_dim': 512,
        'note': 'Includes pseudo-labeled scraped data (similarity > 0.75)'
    }

    torch.save(pseudo_dataset, 'dataset_splits_WITH_PSEUDO_LABELS.pt')
    print(f"\n✅ Saved pseudo-labeled dataset to: dataset_splits_WITH_PSEUDO_LABELS.pt")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n📁 Generated File:")
print("   dataset_splits_WITH_PSEUDO_LABELS.pt")
print(f"      - Gold standard + auto-labeled scraped data")
print(f"      - {len(valid_pseudo) + len(gold_labels)} samples total")
print(f"      - {len(valid_pseudo)} pseudo-labeled (similarity > 0.75)")
print(f"      - {len(gold_labels)} gold standard labeled")
print("      - Larger dataset with inferred labels for better training")

print("\n🎯 Dataset uses BINARY CLASSIFICATION:")
print("   - Class 0: knockoff")
print("   - Class 1: similar")
print("   - NO 'unknown' category!")

print("\n💡 Pseudo-labeling approach:")
print("   - Each unlabeled case assigned label of most similar gold case")
print("   - Only cases with similarity > 0.75 are labeled")
print(f"   - {pseudo_labels.count(None)} cases too uncertain (excluded)")
