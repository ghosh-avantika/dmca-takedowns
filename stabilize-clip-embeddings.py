import torch
import clip
import pandas as pd
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRAPED_FILE = "scraped_cases_text_only_CLEAN.csv"
GOLD_FILE = "gold-standard-cases-CORRECTED.csv"

OUTPUT_EMBEDDINGS = "stabilized_text_embeddings.pt"
OUTPUT_STATS = "embedding_space_stats.npz"

# =========================
# LOAD CLIP
# =========================

print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# =========================
# LOAD DATA
# =========================

scraped_df = pd.read_csv(SCRAPED_FILE)
gold_df = pd.read_csv(GOLD_FILE)

scraped_texts = scraped_df["description"].astype(str).tolist()
gold_texts = gold_df["original_design_elements"].astype(str).tolist()

all_texts = scraped_texts + gold_texts

print(f"Loaded {len(scraped_texts)} scraped texts")
print(f"Loaded {len(gold_texts)} gold texts")
print(f"Total texts: {len(all_texts)}")

# =========================
# ENCODE TEXTS
# =========================

def encode_texts(texts, batch_size=32):
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(DEVICE)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=1, keepdim=True)  # ℓ2 normalization
            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)

print("Encoding texts with CLIP...")
all_embeddings = encode_texts(all_texts)

print("Embedding shape:", all_embeddings.shape)

# =========================
# STABILIZATION STATISTICS
# =========================

embedding_mean = all_embeddings.mean(dim=0)
embedding_std = all_embeddings.std(dim=0)

# Avoid divide-by-zero
embedding_std[embedding_std == 0] = 1e-6

# =========================
# SAVE OUTPUTS
# =========================

torch.save(
    {
        "embeddings": all_embeddings,
        "scraped_count": len(scraped_texts),
        "gold_count": len(gold_texts)
    },
    OUTPUT_EMBEDDINGS
)

np.savez(
    OUTPUT_STATS,
    mean=embedding_mean.numpy(),
    std=embedding_std.numpy()
)

print("✓ Saved stabilized embeddings to:", OUTPUT_EMBEDDINGS)
print("✓ Saved embedding statistics to:", OUTPUT_STATS)
