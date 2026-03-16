import torch
import clip

# Encode design-element descriptions using a frozen CLIP text encoder to obtain semantically grounded feature vectors representing design intent

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# we only need the text encoder
model.eval()


def encode_text(descriptions):
    """
    descriptions: list[str]
    returns: torch.Tensor [N, 512]
    """
    with torch.no_grad():
        tokens = clip.tokenize(descriptions).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    return text_features.cpu()


design_texts = [
    "Hand-drawn feminist motifs, pastel palette", 
    "Reconstructed second-hand shoe collage", 
    "Distinct color-blocking graphics", 
    "Original textile prints", 
    "Unique print + silhouette", 
    "Distinct color-blocked sneaker", 
    "Pigeon motif", 
    "Signature stripe & trim language", 
    "Logo remix tailoring", 
    "Flame illustration", 
    "Oversized hoop geometry", 
    "Italian acetate shapes", 
    "Sculptural gold forms", 
    "Chunky pastel claws", 
    "Minimalist chain motifs", 
    "Modular garment system", 
    "Furry oversized bucket", 
    "Retro prints", 
    "Architectural straps", 
    "Heirloom textile collage", 
    "Braided leather", 
    "Embossed logo tote", 
    "Structured trunk silhouette",
    "Distressed colorways"
]

text_vectors = encode_text(design_texts)
print(f"Feature vectors shape: {text_vectors.shape}")
print(f"Each vector has {text_vectors.shape[1]} dimensions")

# Option 1: Save as PyTorch tensor file (.pt)
torch.save(text_vectors, "design_features.pt")
print("\n✓ Saved to design_features.pt")

# Option 2: Save as numpy array (.npy)
import numpy as np
np.save("design_features.npy", text_vectors.numpy())
print("✓ Saved to design_features.npy")

# Option 3: Save with labels as a dictionary
design_data = {
    'features': text_vectors,
    'labels': design_texts
}
torch.save(design_data, "design_features_with_labels.pt")
print("✓ Saved to design_features_with_labels.pt")

# Example: Compute similarity between two designs
def compute_similarity(vec1, vec2):
    """Cosine similarity (already normalized, so just dot product)"""
    return torch.dot(vec1, vec2).item()

# Compare first two designs
sim = compute_similarity(text_vectors[0], text_vectors[1])
print(f"\nSimilarity between:")
print(f"  '{design_texts[0]}'")
print(f"  '{design_texts[1]}'")
print(f"  = {sim:.4f}")

# Example: Find most similar designs to a query
query_idx = 0
similarities = text_vectors @ text_vectors[query_idx]  # matrix-vector multiplication
top_5_indices = similarities.argsort(descending=True)[1:6]  # Skip first (itself)

print(f"\nTop 5 designs similar to '{design_texts[query_idx]}':")
for i, idx in enumerate(top_5_indices, 1):
    print(f"  {i}. {design_texts[idx]} (similarity: {similarities[idx]:.4f})")
