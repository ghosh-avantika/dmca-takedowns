import numpy as np
import torch
import pandas as pd

# Load the feature vectors
features_np = np.load("design_features.npy")

# Load the labels
data = torch.load("design_features_with_labels.pt")
labels = data['labels']

print("="*80)
print("DESIGN FEATURE VECTORS - DETAILED VIEW")
print("="*80)
print(f"\nArray shape: {features_np.shape}")
print(f"Total designs: {features_np.shape[0]}")
print(f"Vector dimensions: {features_np.shape[1]}")
print(f"Data type: {features_np.dtype}")
print(f"Memory size: {features_np.nbytes / 1024:.2f} KB")

# Show first few values of each vector
print("\n" + "="*80)
print("FEATURE VECTORS (first 10 dimensions shown)")
print("="*80 + "\n")

for i in range(len(features_np)):
    print(f"[{i+1:2d}] {labels[i][:50]:50s}")
    print(f"     Vector (first 10): {features_np[i][:10]}")
    print(f"     Min: {features_np[i].min():.4f} | Max: {features_np[i].max():.4f} | Mean: {features_np[i].mean():.4f}")
    print()

# Create a more detailed CSV export
print("\n" + "="*80)
print("EXPORTING TO CSV FORMAT")
print("="*80)

# Create DataFrame with all 512 dimensions
columns = [f"dim_{i}" for i in range(features_np.shape[1])]
df = pd.DataFrame(features_np, columns=columns)
df.insert(0, 'design_description', labels)
df.to_csv("design_features_readable.csv", index=False)
print(f"✓ Saved to design_features_readable.csv ({features_np.shape[0]} rows × {features_np.shape[1]+1} columns)")

# Create a compact JSON version
print("\n" + "="*80)
print("EXPORTING TO JSON FORMAT")
print("="*80)

import json
json_data = []
for i, label in enumerate(labels):
    json_data.append({
        'id': i,
        'design': label,
        'vector': features_np[i].tolist()
    })

with open("design_features_readable.json", "w") as f:
    json.dump(json_data, f, indent=2)
print(f"✓ Saved to design_features_readable.json")

# Create a Python dict representation
print("\n" + "="*80)
print("CREATING PYTHON DICT FILE")
print("="*80)

with open("design_features_dict.py", "w") as f:
    f.write("# Design Feature Vectors - Python Dictionary Format\n")
    f.write("# Each key is a design description, value is a 512-dimensional vector\n\n")
    f.write("import numpy as np\n\n")
    f.write("DESIGN_FEATURES = {\n")
    for i, label in enumerate(labels):
        f.write(f"    '{label}': np.array([\n")
        # Write vector in chunks of 8 values per line
        vec = features_np[i]
        for j in range(0, len(vec), 8):
            chunk = vec[j:j+8]
            values = ', '.join([f'{v:.6f}' for v in chunk])
            f.write(f"        {values},\n")
        f.write(f"    ]),\n\n")
    f.write("}\n\n")
    f.write("# Quick access by index\n")
    f.write("DESIGN_LABELS = [\n")
    for label in labels:
        f.write(f"    '{label}',\n")
    f.write("]\n")

print(f"✓ Saved to design_features_dict.py")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Overall min value: {features_np.min():.4f}")
print(f"Overall max value: {features_np.max():.4f}")
print(f"Overall mean value: {features_np.mean():.4f}")
print(f"Overall std deviation: {features_np.std():.4f}")
