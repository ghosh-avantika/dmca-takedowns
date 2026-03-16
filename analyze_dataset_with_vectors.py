"""
Analyze the MLP Knockoff Dataset using CLIP feature vectors
Compute similarity scores and patterns
"""

import pandas as pd
import numpy as np
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================================
# 1. LOAD AND CLEAN THE DATASET
# ============================================================================
def load_and_clean_dataset(csv_path):
    """Load the CSV and normalize the varying column structures"""

    df = pd.read_csv(csv_path)

    # Remove empty rows
    df = df.dropna(how='all')

    # Standardize column names and extract relevant fields
    cleaned_data = []

    for idx, row in df.iterrows():
        if pd.isna(row.get('case_id')):
            continue

        case = {
            'case_id': row.get('case_id', '').strip(),
            'original_designer': '',
            'design_elements': '',
            'copier': '',
            'label': '',
            'confidence': ''
        }

        # Extract original designer (various column names)
        for col in ['original_designer_name', 'original_designer', 'original_group', 'original']:
            if col in row and pd.notna(row[col]):
                case['original_designer'] = str(row[col]).strip()
                break

        # Extract design elements (various column names)
        for col in ['original_design_elements', 'elements', 'item_type', 'category']:
            if col in row and pd.notna(row[col]):
                case['design_elements'] = str(row[col]).strip()
                break

        # Extract copier
        for col in ['copier_brand_name', 'copier_brand', 'copier']:
            if col in row and pd.notna(row[col]):
                case['copier'] = str(row[col]).strip()
                break

        # Extract label
        for col in ['infringement_label', 'label']:
            if col in row and pd.notna(row[col]):
                case['label'] = str(row[col]).strip().lower()
                break

        # Extract confidence
        for col in ['confidence']:
            if col in row and pd.notna(row[col]):
                case['confidence'] = str(row[col]).strip().lower()
                break

        cleaned_data.append(case)

    cleaned_df = pd.DataFrame(cleaned_data)

    # Filter out rows without design elements
    cleaned_df = cleaned_df[cleaned_df['design_elements'] != '']

    return cleaned_df


# ============================================================================
# 2. GENERATE FEATURE VECTORS FOR ALL CASES
# ============================================================================
def generate_feature_vectors(design_elements_list, device='cpu'):
    """Generate CLIP feature vectors for all design descriptions"""

    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    print(f"Encoding {len(design_elements_list)} design descriptions...")

    all_vectors = []
    batch_size = 8

    for i in range(0, len(design_elements_list), batch_size):
        batch = design_elements_list[i:i+batch_size]

        with torch.no_grad():
            tokens = clip.tokenize(batch, truncate=True).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=1, keepdim=True)
            all_vectors.append(features.cpu())

    all_vectors = torch.cat(all_vectors, dim=0)

    print(f"✓ Generated feature vectors: {all_vectors.shape}")

    return all_vectors


# ============================================================================
# 3. COMPUTE SIMILARITY MATRIX
# ============================================================================
def compute_similarity_matrix(vectors):
    """Compute pairwise cosine similarity between all vectors"""
    # Vectors are already normalized, so dot product = cosine similarity
    similarity_matrix = vectors @ vectors.T
    return similarity_matrix.numpy()


# ============================================================================
# 4. ANALYZE SIMILARITY PATTERNS
# ============================================================================
def analyze_similarity_patterns(df, similarity_matrix):
    """Analyze similarity patterns between knockoffs and similar cases"""

    print("\n" + "="*80)
    print("SIMILARITY PATTERN ANALYSIS")
    print("="*80)

    # Group by label
    knockoff_indices = df[df['label'] == 'knockoff'].index.tolist()
    similar_indices = df[df['label'] == 'similar'].index.tolist()

    # Average within-group similarities
    if len(knockoff_indices) > 1:
        knockoff_sims = []
        for i in knockoff_indices:
            for j in knockoff_indices:
                if i < j:
                    knockoff_sims.append(similarity_matrix[i, j])
        avg_knockoff_sim = np.mean(knockoff_sims) if knockoff_sims else 0
    else:
        avg_knockoff_sim = 0

    if len(similar_indices) > 1:
        similar_sims = []
        for i in similar_indices:
            for j in similar_indices:
                if i < j:
                    similar_sims.append(similarity_matrix[i, j])
        avg_similar_sim = np.mean(similar_sims) if similar_sims else 0
    else:
        avg_similar_sim = 0

    # Cross-group similarities
    cross_sims = []
    for i in knockoff_indices:
        for j in similar_indices:
            cross_sims.append(similarity_matrix[i, j])
    avg_cross_sim = np.mean(cross_sims) if cross_sims else 0

    print(f"\nKnockoff cases: {len(knockoff_indices)}")
    print(f"Similar cases: {len(similar_indices)}")
    print(f"\nAverage similarity within 'knockoff' cases: {avg_knockoff_sim:.4f}")
    print(f"Average similarity within 'similar' cases: {avg_similar_sim:.4f}")
    print(f"Average similarity between knockoff & similar: {avg_cross_sim:.4f}")

    # Overall statistics
    print(f"\n" + "-"*80)
    print("OVERALL SIMILARITY STATISTICS")
    print("-"*80)

    # Get upper triangle (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    all_similarities = similarity_matrix[upper_triangle_indices]

    print(f"Mean similarity: {np.mean(all_similarities):.4f}")
    print(f"Median similarity: {np.median(all_similarities):.4f}")
    print(f"Std deviation: {np.std(all_similarities):.4f}")
    print(f"Min similarity: {np.min(all_similarities):.4f}")
    print(f"Max similarity: {np.max(all_similarities):.4f}")

    # Distribution
    print(f"\nSimilarity distribution:")
    print(f"  > 0.9 (very high): {np.sum(all_similarities > 0.9)}")
    print(f"  0.8 - 0.9 (high): {np.sum((all_similarities >= 0.8) & (all_similarities <= 0.9))}")
    print(f"  0.7 - 0.8 (moderate): {np.sum((all_similarities >= 0.7) & (all_similarities < 0.8))}")
    print(f"  0.6 - 0.7 (low): {np.sum((all_similarities >= 0.6) & (all_similarities < 0.7))}")
    print(f"  < 0.6 (very low): {np.sum(all_similarities < 0.6)}")


# ============================================================================
# 5. FIND MOST SIMILAR PAIRS
# ============================================================================
def find_most_similar_pairs(df, similarity_matrix, top_k=10):
    """Find the most similar design pairs"""

    print("\n" + "="*80)
    print(f"TOP {top_k} MOST SIMILAR DESIGN PAIRS")
    print("="*80 + "\n")

    # Get upper triangle indices (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_triangle_indices]

    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    for rank, idx in enumerate(top_indices, 1):
        i = upper_triangle_indices[0][idx]
        j = upper_triangle_indices[1][idx]
        sim = similarities[idx]

        row1 = df.iloc[i]
        row2 = df.iloc[j]

        print(f"{rank}. Similarity: {sim:.4f}")
        print(f"   [{row1['case_id']}] {row1['design_elements'][:60]}")
        print(f"        {row1['original_designer']} → {row1['copier']} | Label: {row1['label']}")
        print(f"   [{row2['case_id']}] {row2['design_elements'][:60]}")
        print(f"        {row2['original_designer']} → {row2['copier']} | Label: {row2['label']}")
        print()


# ============================================================================
# 6. FIND SIMILAR CASES FOR EACH DESIGN
# ============================================================================
def find_similar_for_each(df, similarity_matrix, output_file='similarity_rankings.csv'):
    """For each case, find its most similar cases"""

    results = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        similarities = similarity_matrix[idx]

        # Get top 5 most similar (excluding itself)
        top_indices = np.argsort(similarities)[-6:-1][::-1]  # -6 to skip itself, then reverse

        similar_cases = []
        similar_scores = []

        for top_idx in top_indices:
            similar_row = df.iloc[top_idx]
            similar_cases.append(f"{similar_row['case_id']}: {similar_row['design_elements'][:40]}")
            similar_scores.append(f"{similarities[top_idx]:.4f}")

        results.append({
            'case_id': row['case_id'],
            'design_elements': row['design_elements'],
            'original_designer': row['original_designer'],
            'copier': row['copier'],
            'label': row['label'],
            'top_similar_1': similar_cases[0] if len(similar_cases) > 0 else '',
            'similarity_1': similar_scores[0] if len(similar_scores) > 0 else '',
            'top_similar_2': similar_cases[1] if len(similar_cases) > 1 else '',
            'similarity_2': similar_scores[1] if len(similar_scores) > 1 else '',
            'top_similar_3': similar_cases[2] if len(similar_cases) > 2 else '',
            'similarity_3': similar_scores[2] if len(similar_scores) > 2 else '',
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Similarity rankings saved to: {output_file}")

    return results_df


# ============================================================================
# 7. VISUALIZE SIMILARITY MATRIX
# ============================================================================
def visualize_similarity_matrix(df, similarity_matrix, output_file='similarity_heatmap.png'):
    """Create a heatmap of the similarity matrix"""

    plt.figure(figsize=(16, 14))

    # Create labels for axes (case IDs)
    labels = df['case_id'].tolist()

    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'},
        annot=False
    )

    plt.title('Design Similarity Matrix (CLIP Text Embeddings)', fontsize=16, pad=20)
    plt.xlabel('Case ID', fontsize=12)
    plt.ylabel('Case ID', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Similarity heatmap saved to: {output_file}")
    plt.close()


# ============================================================================
# 8. MAIN ANALYSIS PIPELINE
# ============================================================================
def main():
    print("="*80)
    print("MLP KNOCKOFF DATASET - SIMILARITY ANALYSIS WITH CLIP VECTORS")
    print("="*80)

    # Load dataset
    csv_path = "/Users/avantikaghosh/Downloads/MLP Datasets - Gold Standard Cases.csv"
    print(f"\nLoading dataset from: {csv_path}")
    df = load_and_clean_dataset(csv_path)

    print(f"✓ Loaded {len(df)} cases")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())

    # Generate feature vectors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    design_elements = df['design_elements'].tolist()
    feature_vectors = generate_feature_vectors(design_elements, device)

    # Save feature vectors
    torch.save({
        'vectors': feature_vectors,
        'design_elements': design_elements,
        'case_ids': df['case_id'].tolist()
    }, 'dataset_feature_vectors.pt')
    print("✓ Feature vectors saved to: dataset_feature_vectors.pt")

    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(feature_vectors)

    # Save similarity matrix
    np.save('similarity_matrix.npy', similarity_matrix)
    print("✓ Similarity matrix saved to: similarity_matrix.npy")

    # Analyze patterns
    analyze_similarity_patterns(df, similarity_matrix)

    # Find most similar pairs
    find_most_similar_pairs(df, similarity_matrix, top_k=15)

    # Generate similarity rankings for each case
    results_df = find_similar_for_each(df, similarity_matrix)

    # Visualize
    print("\nGenerating visualizations...")
    visualize_similarity_matrix(df, similarity_matrix)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. dataset_feature_vectors.pt - CLIP embeddings for all cases")
    print("  2. similarity_matrix.npy - Pairwise similarity scores")
    print("  3. similarity_rankings.csv - Top similar cases for each design")
    print("  4. similarity_heatmap.png - Visual similarity matrix")

    return df, feature_vectors, similarity_matrix, results_df


if __name__ == "__main__":
    df, vectors, sim_matrix, results = main()
