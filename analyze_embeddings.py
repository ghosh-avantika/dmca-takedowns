"""
Analyze the stabilized CLIP embeddings - compute similarity scores and patterns
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print("ANALYZING CLIP EMBEDDINGS - SIMILARITY ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading embeddings...")
data = torch.load('stabilized_text_embeddings.pt')
embeddings = data['embeddings']  # [186, 512]
scraped_count = data['scraped_count']  # 136
gold_count = data['gold_count']  # 50

print(f"   ✓ Loaded {embeddings.shape[0]} embeddings ({embeddings.shape[1]} dimensions)")
print(f"   - Scraped cases: {scraped_count}")
print(f"   - Gold standard: {gold_count}")

# Load the gold standard CSV to get labels
gold_df = pd.read_csv('gold-standard-cases.csv')
gold_df = gold_df.dropna(how='all')

# Extract labels from gold standard
labels = []
for idx, row in gold_df.iterrows():
    label = None
    for col in ['infringement_label', 'label']:
        if col in row and pd.notna(row[col]):
            label = str(row[col]).strip().lower()
            break
    labels.append(label if label else 'unknown')

print(f"\n2. Loaded {len(labels)} labels from gold standard")

# ============================================================================
# 2. COMPUTE SIMILARITY MATRIX
# ============================================================================
print("\n3. Computing similarity matrix...")
# Cosine similarity (embeddings are already normalized)
similarity_matrix = (embeddings @ embeddings.T).numpy()

print(f"   ✓ Similarity matrix shape: {similarity_matrix.shape}")

# Save similarity matrix
np.save('full_similarity_matrix.npy', similarity_matrix)
print(f"   ✓ Saved to: full_similarity_matrix.npy")

# ============================================================================
# 3. ANALYZE GOLD STANDARD SIMILARITIES
# ============================================================================
print("\n4. Analyzing gold standard cases (last 50)...")

# Extract gold standard portion
gold_start = scraped_count
gold_end = scraped_count + gold_count
gold_sim_matrix = similarity_matrix[gold_start:gold_end, gold_start:gold_end]

# Separate by label
knockoff_indices = [i for i, label in enumerate(labels[:gold_count]) if label == 'knockoff']
similar_indices = [i for i, label in enumerate(labels[:gold_count]) if label == 'similar']

print(f"   - Knockoff cases: {len(knockoff_indices)}")
print(f"   - Similar cases: {len(similar_indices)}")

# Within-group similarities
knockoff_sims = []
if len(knockoff_indices) > 1:
    for i in knockoff_indices:
        for j in knockoff_indices:
            if i < j:
                knockoff_sims.append(gold_sim_matrix[i, j])

similar_sims = []
if len(similar_indices) > 1:
    for i in similar_indices:
        for j in similar_indices:
            if i < j:
                similar_sims.append(gold_sim_matrix[i, j])

# Cross-group similarities
cross_sims = []
for i in knockoff_indices:
    for j in similar_indices:
        cross_sims.append(gold_sim_matrix[i, j])

print(f"\n   Gold Standard Similarity Patterns:")
if knockoff_sims:
    print(f"   - Avg similarity within knockoffs: {np.mean(knockoff_sims):.4f}")
if similar_sims:
    print(f"   - Avg similarity within similar:   {np.mean(similar_sims):.4f}")
if cross_sims:
    print(f"   - Avg similarity across groups:    {np.mean(cross_sims):.4f}")

# ============================================================================
# 4. OVERALL STATISTICS
# ============================================================================
print("\n5. Overall similarity statistics (all 186 cases)...")

# Get upper triangle (excluding diagonal)
upper_triangle = np.triu_indices_from(similarity_matrix, k=1)
all_sims = similarity_matrix[upper_triangle]

print(f"   Total pairwise comparisons: {len(all_sims)}")
print(f"   Mean:   {np.mean(all_sims):.4f}")
print(f"   Median: {np.median(all_sims):.4f}")
print(f"   Std:    {np.std(all_sims):.4f}")
print(f"   Min:    {np.min(all_sims):.4f}")
print(f"   Max:    {np.max(all_sims):.4f}")

# Distribution
print(f"\n   Similarity Distribution:")
print(f"   - Very high (> 0.9):  {np.sum(all_sims > 0.9):4d} pairs ({100*np.sum(all_sims > 0.9)/len(all_sims):.1f}%)")
print(f"   - High (0.8-0.9):     {np.sum((all_sims >= 0.8) & (all_sims <= 0.9)):4d} pairs ({100*np.sum((all_sims >= 0.8) & (all_sims <= 0.9))/len(all_sims):.1f}%)")
print(f"   - Moderate (0.7-0.8): {np.sum((all_sims >= 0.7) & (all_sims < 0.8)):4d} pairs ({100*np.sum((all_sims >= 0.7) & (all_sims < 0.8))/len(all_sims):.1f}%)")
print(f"   - Low (0.6-0.7):      {np.sum((all_sims >= 0.6) & (all_sims < 0.7)):4d} pairs ({100*np.sum((all_sims >= 0.6) & (all_sims < 0.7))/len(all_sims):.1f}%)")
print(f"   - Very low (< 0.6):   {np.sum(all_sims < 0.6):4d} pairs ({100*np.sum(all_sims < 0.6)/len(all_sims):.1f}%)")

# ============================================================================
# 5. FIND MOST SIMILAR PAIRS IN GOLD STANDARD
# ============================================================================
print("\n6. Top 10 most similar pairs in gold standard...")

gold_upper = np.triu_indices_from(gold_sim_matrix, k=1)
gold_sims = gold_sim_matrix[gold_upper]
top_10_indices = np.argsort(gold_sims)[-10:][::-1]

for rank, idx in enumerate(top_10_indices, 1):
    i = gold_upper[0][idx]
    j = gold_upper[1][idx]
    sim = gold_sims[idx]

    # Only process if indices are valid
    if i < len(labels) and j < len(labels):
        label_i = labels[i] if i < len(labels) else 'unknown'
        label_j = labels[j] if j < len(labels) else 'unknown'

        elem_i = ''
        elem_j = ''
        if i < len(gold_df):
            row_i = gold_df.iloc[i]
            for col in ['original_design_elements', 'elements']:
                if col in row_i and pd.notna(row_i[col]):
                    elem_i = str(row_i[col])[:50]
                    break
        if j < len(gold_df):
            row_j = gold_df.iloc[j]
            for col in ['original_design_elements', 'elements']:
                if col in row_j and pd.notna(row_j[col]):
                    elem_j = str(row_j[col])[:50]
                    break

        print(f"\n   {rank}. Similarity: {sim:.4f}")
        print(f"      [{label_i}] {elem_i}")
        print(f"      [{label_j}] {elem_j}")

# ============================================================================
# 6. EXPORT DETAILED SIMILARITY SCORES
# ============================================================================
print("\n7. Exporting detailed similarity scores...")

# Create a detailed report for gold standard
results = []
actual_gold_rows = min(gold_count, len(gold_df))
for i in range(actual_gold_rows):
    row = gold_df.iloc[i]

    # Get design elements
    design_elem = ''
    for col in ['original_design_elements', 'elements']:
        if col in row and pd.notna(row[col]):
            design_elem = str(row[col])
            break

    # Get case ID
    case_id = row.get('case_id', f'Case_{i+1}')

    # Find top 5 most similar
    sims = gold_sim_matrix[i]
    top_5_idx = np.argsort(sims)[-6:-1][::-1]  # Exclude self

    results.append({
        'case_id': case_id,
        'design_elements': design_elem,
        'label': labels[i],
        'avg_similarity_to_all': np.mean(sims),
        'max_similarity': np.max(sims[sims < 1.0]),  # Exclude self (1.0)
        'top_similar_1_score': sims[top_5_idx[0]] if len(top_5_idx) > 0 else 0,
        'top_similar_2_score': sims[top_5_idx[1]] if len(top_5_idx) > 1 else 0,
        'top_similar_3_score': sims[top_5_idx[2]] if len(top_5_idx) > 2 else 0,
    })

results_df = pd.DataFrame(results)
results_df.to_csv('gold_standard_similarity_scores.csv', index=False)
print(f"   ✓ Saved to: gold_standard_similarity_scores.csv")

# ============================================================================
# 7. VISUALIZE
# ============================================================================
print("\n8. Creating visualizations...")

# Heatmap of gold standard similarity
plt.figure(figsize=(14, 12))
sns.heatmap(gold_sim_matrix,
            cmap='YlOrRd',
            vmin=0, vmax=1,
            square=True,
            cbar_kws={'label': 'Cosine Similarity'})
plt.title('Gold Standard Cases - Similarity Heatmap (CLIP Embeddings)', fontsize=14)
plt.xlabel('Case Index', fontsize=12)
plt.ylabel('Case Index', fontsize=12)
plt.tight_layout()
plt.savefig('gold_standard_similarity_heatmap.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved heatmap: gold_standard_similarity_heatmap.png")
plt.close()

# Distribution histogram
plt.figure(figsize=(10, 6))
plt.hist(all_sims, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(all_sims), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_sims):.3f}')
plt.axvline(np.median(all_sims), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(all_sims):.3f}')
plt.xlabel('Cosine Similarity', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Pairwise Similarities (All 186 Cases)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('similarity_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved histogram: similarity_distribution.png")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. full_similarity_matrix.npy - All pairwise similarities")
print("  2. gold_standard_similarity_scores.csv - Detailed scores per case")
print("  3. gold_standard_similarity_heatmap.png - Visual matrix")
print("  4. similarity_distribution.png - Histogram")
