import torch
import numpy as np
import pandas as pd
from typing import List, Tuple

# Load the saved feature vectors and labels
data = torch.load("design_features_with_labels.pt")
text_vectors = data['features']
design_texts = data['labels']

# Create dataset from the PDF cases (first 24 cases)
dataset = pd.DataFrame({
    'case_id': [f'DP_{str(i+1).zfill(3)}' for i in range(24)],
    'design_elements': design_texts,
    'infringement_label': [
        'knockoff', 'knockoff', 'similar', 'knockoff', 'knockoff', 'knockoff',  # 1-6
        'similar', 'similar', 'knockoff', 'knockoff',  # 7-10
        'similar', 'similar', 'knockoff', 'knockoff', 'similar',  # 11-15
        'similar', 'knockoff', 'knockoff', 'similar', 'similar',  # 16-20
        'similar', 'knockoff', 'knockoff', 'similar'  # 21-24
    ],
    'original_designer': [
        'Tuesday Bassen', 'Elisa van Joolen', 'Colrs', 'Karen Perez', 'Art Garments',
        'Cecilia Monge', 'Jeff Staple', 'Wales Bonner', 'Dapper Dan', 'Katie Perry',
        'Jennifer Fisher', 'Machete Jewelry', 'Wolf Circus', 'Valet Studio', 'Petit Moments',
        'Miranda Bennett', 'Emma Brewin', 'Lisa Says Gah', 'Chromat', 'Bode',
        'Brother Vellies', 'Telfar Clemens', 'Brandon Blackwood', 'ERL'
    ],
    'copier': [
        'Zara', 'Off-White™', 'Off-White™', 'Multiple mass brands', 'Sarah Bernstein',
        'Converse', 'Fast-fashion footwear', 'Mass retailers', 'Gucci', 'Forever 21',
        'Shein', 'Anthropologie', 'Amazon sellers', 'Fast fashion', 'Shein',
        'Madewell', 'Urban Outfitters', 'Shein', 'Mass swimwear', 'Zara',
        'Fast fashion', 'Fast fashion', 'Amazon sellers', 'Mall brands'
    ]
})

# Add feature vectors to the dataset
dataset['feature_vector'] = [text_vectors[i] for i in range(len(text_vectors))]

print("=" * 80)
print("DESIGN KNOCKOFF DATASET - Feature Vector Analysis")
print("=" * 80)
print(f"\nDataset size: {len(dataset)} cases")
print(f"Feature vector dimensions: {text_vectors.shape[1]}")
print(f"\nLabel distribution:")
print(dataset['infringement_label'].value_counts())


# ============================================================================
# 1. COMPUTE SIMILARITY MATRIX
# ============================================================================
def compute_similarity_matrix(vectors):
    """Compute pairwise cosine similarity between all vectors"""
    # Vectors are already normalized, so dot product = cosine similarity
    return vectors @ vectors.T

similarity_matrix = compute_similarity_matrix(text_vectors)
print(f"\n\n{'='*80}")
print("SIMILARITY MATRIX COMPUTED")
print(f"{'='*80}")
print(f"Shape: {similarity_matrix.shape}")


# ============================================================================
# 2. FIND MOST SIMILAR DESIGNS
# ============================================================================
def find_similar_designs(query_idx: int, top_k: int = 5):
    """Find the most similar designs to a given design"""
    similarities = similarity_matrix[query_idx]
    # Get top-k indices (excluding itself)
    top_indices = similarities.argsort(descending=True)[1:top_k+1].tolist()

    print(f"\n{'='*80}")
    print(f"TOP {top_k} DESIGNS SIMILAR TO: {dataset.iloc[query_idx]['design_elements']}")
    print(f"Case: {dataset.iloc[query_idx]['case_id']} | Designer: {dataset.iloc[query_idx]['original_designer']}")
    print(f"{'='*80}\n")

    for rank, idx in enumerate(top_indices, 1):
        row = dataset.iloc[idx]
        sim_score = similarities[idx].item()
        print(f"{rank}. [{row['case_id']}] Similarity: {sim_score:.4f}")
        print(f"   Design: {row['design_elements']}")
        print(f"   Designer: {row['original_designer']} → Copied by: {row['copier']}")
        print(f"   Label: {row['infringement_label']}\n")


# Example: Find designs similar to "Hand-drawn feminist motifs, pastel palette"
find_similar_designs(0, top_k=5)


# ============================================================================
# 3. COMPARE ORIGINAL vs KNOCKOFF SIMILARITY
# ============================================================================
def analyze_label_similarities():
    """Analyze if knockoffs have different similarity patterns than similar cases"""
    knockoff_indices = dataset[dataset['infringement_label'] == 'knockoff'].index
    similar_indices = dataset[dataset['infringement_label'] == 'similar'].index

    # Average similarity within each group
    knockoff_sims = []
    for i in knockoff_indices:
        for j in knockoff_indices:
            if i < j:
                knockoff_sims.append(similarity_matrix[i, j].item())

    similar_sims = []
    for i in similar_indices:
        for j in similar_indices:
            if i < j:
                similar_sims.append(similarity_matrix[i, j].item())

    print(f"\n{'='*80}")
    print("LABEL-BASED SIMILARITY ANALYSIS")
    print(f"{'='*80}")
    print(f"\nAverage similarity within 'knockoff' cases: {np.mean(knockoff_sims):.4f}")
    print(f"Average similarity within 'similar' cases: {np.mean(similar_sims):.4f}")
    print(f"\nThis helps understand if knockoff cases cluster differently than similar cases")

analyze_label_similarities()


# ============================================================================
# 4. QUERY NEW DESIGN
# ============================================================================
def compare_to_dataset(new_design_text: str, top_k: int = 3):
    """
    Compare a new design description to the entire dataset
    Useful for checking if a new design is similar to known knockoff patterns
    """
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    # Encode the new design
    with torch.no_grad():
        tokens = clip.tokenize([new_design_text]).to(device)
        new_vector = model.encode_text(tokens)
        new_vector = new_vector / new_vector.norm(dim=1, keepdim=True)
        new_vector = new_vector.cpu()

    # Compare to all designs in dataset
    similarities = text_vectors @ new_vector.T
    similarities = similarities.squeeze()
    top_indices = similarities.argsort(descending=True)[:top_k].tolist()

    print(f"\n{'='*80}")
    print(f"COMPARING NEW DESIGN: '{new_design_text}'")
    print(f"{'='*80}\n")

    for rank, idx in enumerate(top_indices, 1):
        row = dataset.iloc[idx]
        sim_score = similarities[idx].item()
        print(f"{rank}. [{row['case_id']}] Similarity: {sim_score:.4f} | Label: {row['infringement_label']}")
        print(f"   Existing design: {row['design_elements']}")
        print(f"   Designer: {row['original_designer']}\n")

    return similarities, top_indices


# Example: Test with a new design
print("\n\n")
compare_to_dataset("Colorful geometric patterns with bold lines", top_k=5)


# ============================================================================
# 5. SAVE PROCESSED DATASET
# ============================================================================
# Save dataset with embeddings for future use
output_data = {
    'dataset': dataset,
    'similarity_matrix': similarity_matrix,
    'feature_vectors': text_vectors
}
torch.save(output_data, "knockoff_dataset_with_embeddings.pt")
print(f"\n{'='*80}")
print("✓ Saved dataset with embeddings to 'knockoff_dataset_with_embeddings.pt'")
print(f"{'='*80}")
