"""
Split the dataset into training, validation, and testing sets
Uses stratified splitting to maintain label distribution
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("DATA SPLITTING: TRAIN / VALIDATION / TEST")
print("="*80)

# ============================================================================
# 1. LOAD EMBEDDINGS AND LABELS
# ============================================================================
print("\n1. Loading data...")

# Load embeddings
data = torch.load('stabilized_text_embeddings.pt')
embeddings = data['embeddings']  # [186, 512]
scraped_count = data['scraped_count']  # 136
gold_count = data['gold_count']  # 50

print(f"   Total samples: {len(embeddings)}")
print(f"   - Scraped: {scraped_count}")
print(f"   - Gold standard: {gold_count}")

# Load labels from both datasets
scraped_df = pd.read_csv('scraped_cases_text_only_CLEAN.csv')

# Use corrected gold standard if it exists, otherwise use original
import os
if os.path.exists('gold-standard-cases-CORRECTED.csv'):
    gold_df = pd.read_csv('gold-standard-cases-CORRECTED.csv')
    print("   ✓ Using CORRECTED gold standard CSV")
else:
    gold_df = pd.read_csv('gold-standard-cases.csv')
    print("   ⚠️  Using original gold standard CSV (may have label issues)")

gold_df = gold_df.dropna(how='all')

# Extract labels - match exactly with how embeddings were created
all_labels = []
all_texts = []

# Scraped data labels (first scraped_count)
for idx in range(min(scraped_count, len(scraped_df))):
    row = scraped_df.iloc[idx]
    label = None
    for col in ['label', 'infringement_label']:
        if col in row and pd.notna(row[col]):
            label = str(row[col]).strip().lower()
            break

    text = ''
    for col in ['description', 'design_elements']:
        if col in row and pd.notna(row[col]):
            text = str(row[col])
            break

    all_labels.append(label if label else 'unknown')
    all_texts.append(text)

# Gold standard labels (next gold_count)
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

    all_labels.append(label if label else 'unknown')
    all_texts.append(text)

print(f"   Created {len(all_labels)} labels")

# Verify and fix mismatch
if len(all_labels) != len(embeddings):
    print(f"\n   ⚠️  Mismatch detected:")
    print(f"   - Embeddings: {len(embeddings)}")
    print(f"   - Labels: {len(all_labels)}")
    print(f"   - Trimming embeddings to match available labels...")
    embeddings = embeddings[:len(all_labels)]

print(f"\n2. Label distribution ({len(all_labels)} samples):")
labels_series = pd.Series(all_labels)
print(labels_series.value_counts())

# ============================================================================
# 3. PREPARE NUMERICAL LABELS (keeping all original labels)
# ============================================================================
print("\n3. Converting labels to numerical format (keeping all original labels)...")

# Create label mapping
unique_labels = sorted(set(all_labels))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

print(f"   Label mapping:")
for label, idx in label_to_idx.items():
    count = all_labels.count(label)
    print(f"   - {label}: {idx} ({count} samples)")

# Convert to numerical
numerical_labels = [label_to_idx[label] for label in all_labels]

# ============================================================================
# 3. SPLIT DATA (70% train, 15% val, 15% test)
# ============================================================================
print("\n4. Splitting data...")

# Convert to numpy
X = embeddings.numpy()
y = np.array(numerical_labels)

# Check which classes have too few samples for stratification
class_counts = pd.Series(y).value_counts()
min_class_size = class_counts.min()
print(f"   Smallest class has {min_class_size} samples")

# Can't use stratification if any class has < 4 samples (need at least 2 per split)
use_stratify = min_class_size >= 4

if use_stratify:
    print("   Using stratified split to maintain label distribution")
    # First split: 70% train, 30% temp (for val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
else:
    print("   ⚠️  Some classes too small for stratification - using random split")
    # First split: 70% train, 30% temp (for val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42
    )

# Second split: 50% of temp = 15% val, 15% test
# Don't stratify second split due to small class sizes
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

print(f"\n   Split sizes:")
print(f"   - Training:   {len(X_train):3d} samples ({100*len(X_train)/len(X):.1f}%)")
print(f"   - Validation: {len(X_val):3d} samples ({100*len(X_val)/len(X):.1f}%)")
print(f"   - Testing:    {len(X_test):3d} samples ({100*len(X_test)/len(X):.1f}%)")

# ============================================================================
# 4. VERIFY LABEL DISTRIBUTION
# ============================================================================
print("\n5. Verifying label distribution across splits...")

def print_label_dist(y_split, split_name):
    print(f"\n   {split_name}:")
    for idx, label in idx_to_label.items():
        count = np.sum(y_split == idx)
        pct = 100 * count / len(y_split)
        print(f"   - {label}: {count:3d} ({pct:5.1f}%)")

print_label_dist(y_train, "Training")
print_label_dist(y_val, "Validation")
print_label_dist(y_test, "Testing")

# ============================================================================
# 5. SAVE SPLITS
# ============================================================================
print("\n6. Saving splits...")

# Save as PyTorch tensors
train_data = {
    'embeddings': torch.from_numpy(X_train),
    'labels': torch.from_numpy(y_train),
    'size': len(X_train)
}

val_data = {
    'embeddings': torch.from_numpy(X_val),
    'labels': torch.from_numpy(y_val),
    'size': len(X_val)
}

test_data = {
    'embeddings': torch.from_numpy(X_test),
    'labels': torch.from_numpy(y_test),
    'size': len(X_test)
}

# Save complete dataset info
dataset_info = {
    'train': train_data,
    'val': val_data,
    'test': test_data,
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'num_classes': len(unique_labels),
    'feature_dim': 512
}

torch.save(dataset_info, 'dataset_splits.pt')
print("   ✓ Saved to: dataset_splits.pt")

# Also save as separate files for convenience
torch.save(train_data, 'train_split.pt')
torch.save(val_data, 'val_split.pt')
torch.save(test_data, 'test_split.pt')

print("   ✓ Saved to: train_split.pt, val_split.pt, test_split.pt")

# ============================================================================
# 6. CREATE SUMMARY CSV
# ============================================================================
print("\n7. Creating summary CSV...")

summary = []
for split_name, (X_split, y_split) in [
    ('train', (X_train, y_train)),
    ('val', (X_val, y_val)),
    ('test', (X_test, y_test))
]:
    for idx, label in idx_to_label.items():
        count = np.sum(y_split == idx)
        summary.append({
            'split': split_name,
            'label': label,
            'count': count,
            'percentage': 100 * count / len(y_split)
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv('data_split_summary.csv', index=False)
print("   ✓ Saved to: data_split_summary.csv")

# ============================================================================
# 7. VISUALIZE SPLITS
# ============================================================================
print("\n8. Creating visualizations...")

# Bar chart showing split distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (split_name, y_split, ax) in enumerate([
    ('Training', y_train, axes[0]),
    ('Validation', y_val, axes[1]),
    ('Testing', y_test, axes[2])
]):
    counts = [np.sum(y_split == i) for i in range(len(unique_labels))]
    labels = [idx_to_label[i] for i in range(len(unique_labels))]

    ax.bar(labels, counts, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'][:len(labels)])
    ax.set_title(f'{split_name} Set\n({len(y_split)} samples)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Label', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data_split_visualization.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved to: data_split_visualization.png")
plt.close()

# Pie chart showing overall split proportions
fig, ax = plt.subplots(figsize=(8, 8))
sizes = [len(X_train), len(X_val), len(X_test)]
labels_pie = [f'Training\n{len(X_train)} ({100*len(X_train)/len(X):.1f}%)',
              f'Validation\n{len(X_val)} ({100*len(X_val)/len(X):.1f}%)',
              f'Testing\n{len(X_test)} ({100*len(X_test)/len(X):.1f}%)']
colors = ['#3498db', '#e74c3c', '#2ecc71']

ax.pie(sizes, labels=labels_pie, colors=colors, autopct='', startangle=90, textprops={'fontsize': 12})
ax.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('data_split_pie_chart.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved to: data_split_pie_chart.png")
plt.close()

# ============================================================================
# 8. EXAMPLE USAGE CODE
# ============================================================================
print("\n" + "="*80)
print("SPLITTING COMPLETE!")
print("="*80)

print("\n📁 Generated files:")
print("   1. dataset_splits.pt - Complete dataset with all splits")
print("   2. train_split.pt, val_split.pt, test_split.pt - Individual splits")
print("   3. data_split_summary.csv - Summary statistics")
print("   4. data_split_visualization.png - Bar charts")
print("   5. data_split_pie_chart.png - Pie chart")

print("\n💻 How to load the data:")
print("""
import torch

# Load all splits at once
data = torch.load('dataset_splits.pt')
train_X = data['train']['embeddings']
train_y = data['train']['labels']
val_X = data['val']['embeddings']
val_y = data['val']['labels']
test_X = data['test']['embeddings']
test_y = data['test']['labels']

# Get label mappings
label_to_idx = data['label_to_idx']
idx_to_label = data['idx_to_label']
num_classes = data['num_classes']

print(f"Training: {train_X.shape}, Validation: {val_X.shape}, Test: {test_X.shape}")
""")

print("\n✅ Ready to train your MLP classifier!")
