"""
MLP Classifier for Fashion Design Knockoff Detection
Trains on CLIP embeddings to classify: knockoff vs similar
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("="*80)
print("MLP CLASSIFIER FOR DESIGN KNOCKOFF DETECTION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading dataset...")

data = torch.load('dataset_splits_WITH_PSEUDO_LABELS.pt')

train_X = data['train']['embeddings'].float()
train_y = data['train']['labels'].long()

val_X = data['val']['embeddings'].float()
val_y = data['val']['labels'].long()

test_X = data['test']['embeddings'].float()
test_y = data['test']['labels'].long()

label_to_idx = data['label_to_idx']
idx_to_label = data['idx_to_label']

print(f"   Training:   {len(train_X)} samples")
print(f"   Validation: {len(val_X)} samples")
print(f"   Testing:    {len(test_X)} samples")
print(f"   Features:   {train_X.shape[1]} dimensions")
print(f"   Classes:    {label_to_idx}")

# Check class distribution
train_knockoff = (train_y == 0).sum().item()
train_similar = (train_y == 1).sum().item()
print(f"\n   Train distribution: knockoff={train_knockoff}, similar={train_similar}")

# ============================================================================
# 2. CREATE DATALOADERS
# ============================================================================
print("\n2. Creating DataLoaders...")

BATCH_SIZE = 16

train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)
test_dataset = TensorDataset(test_X, test_y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"   Batch size: {BATCH_SIZE}")

# ============================================================================
# 3. DEFINE MLP MODEL
# ============================================================================
print("\n3. Defining MLP architecture...")

class MLPClassifier(nn.Module):
    """
    Simple MLP with dropout regularization for small dataset
    Architecture: 512 -> 128 -> 64 -> 2
    """
    def __init__(self, input_dim=512, hidden1=128, hidden2=64, num_classes=2, dropout=0.4):
        super(MLPClassifier, self).__init__()

        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Second hidden layer
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output layer
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Initialize model
model = MLPClassifier(input_dim=512, hidden1=128, hidden2=64, num_classes=2, dropout=0.4)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Architecture: 512 -> 128 -> 64 -> 2")
print(f"   Total parameters: {total_params:,}")
print(f"   Dropout: 0.4")

# ============================================================================
# 4. TRAINING SETUP
# ============================================================================
print("\n4. Setting up training...")

# Handle class imbalance with weighted loss
class_counts = torch.bincount(train_y)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()  # Normalize
print(f"   Class weights: knockoff={class_weights[0]:.3f}, similar={class_weights[1]:.3f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# Training config
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 20

print(f"   Optimizer: Adam (lr=0.001, weight_decay=1e-4)")
print(f"   Max epochs: {NUM_EPOCHS}")
print(f"   Early stopping patience: {EARLY_STOP_PATIENCE}")

# ============================================================================
# 5. TRAINING LOOP
# ============================================================================
print("\n5. Training...")
print("-" * 60)

train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float('inf')
best_val_acc = 0
patience_counter = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.numpy())
            val_true.extend(batch_y.numpy())

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    val_acc = accuracy_score(val_true, val_preds)
    val_accuracies.append(val_acc)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:3d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n   Early stopping at epoch {epoch+1}")
            break

print("-" * 60)
print(f"   Best validation loss: {best_val_loss:.4f}")
print(f"   Best validation accuracy: {best_val_acc:.4f}")

# Load best model
model.load_state_dict(best_model_state)

# ============================================================================
# 6. EVALUATION ON TEST SET
# ============================================================================
print("\n6. Evaluating on test set...")

model.eval()
test_preds = []
test_true = []
test_probs = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        test_preds.extend(preds.numpy())
        test_true.extend(batch_y.numpy())
        test_probs.extend(probs.numpy())

test_preds = np.array(test_preds)
test_true = np.array(test_true)
test_probs = np.array(test_probs)

# Calculate metrics
accuracy = accuracy_score(test_true, test_preds)
precision = precision_score(test_true, test_preds, average='weighted')
recall = recall_score(test_true, test_preds, average='weighted')
f1 = f1_score(test_true, test_preds, average='weighted')

print(f"\n{'='*60}")
print("TEST SET RESULTS")
print("="*60)
print(f"\n   Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1 Score:  {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(test_true, test_preds)
print(f"\n   Confusion Matrix:")
print(f"                    Predicted")
print(f"                 knockoff  similar")
print(f"   Actual knockoff    {cm[0,0]:3d}      {cm[0,1]:3d}")
print(f"   Actual similar     {cm[1,0]:3d}      {cm[1,1]:3d}")

# Classification report
print(f"\n   Classification Report:")
print(classification_report(test_true, test_preds, target_names=['knockoff', 'similar']))

# ============================================================================
# 7. SAVE MODEL AND RESULTS
# ============================================================================
print("\n7. Saving model and results...")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': {
        'input_dim': 512,
        'hidden1': 128,
        'hidden2': 64,
        'num_classes': 2,
        'dropout': 0.4
    },
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'test_accuracy': accuracy,
    'test_f1': f1
}, 'mlp_knockoff_classifier.pt')

print(f"   ✓ Saved model to: mlp_knockoff_classifier.pt")

# Save training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print(f"   ✓ Saved training curves to: training_curves.png")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"""
Model: MLP Classifier (512 -> 128 -> 64 -> 2)
Dataset: {len(train_X)} train / {len(val_X)} val / {len(test_X)} test

Results:
  - Test Accuracy: {accuracy*100:.1f}%
  - Test F1 Score: {f1:.4f}

Files saved:
  - mlp_knockoff_classifier.pt (trained model)
  - training_curves.png (loss/accuracy plots)

To use the model for prediction:

  import torch

  # Load model
  checkpoint = torch.load('mlp_knockoff_classifier.pt')
  model = MLPClassifier(**checkpoint['model_architecture'])
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  # Predict on new CLIP embedding
  with torch.no_grad():
      output = model(new_embedding)
      pred = torch.argmax(output, dim=1)
      label = checkpoint['idx_to_label'][pred.item()]
""")