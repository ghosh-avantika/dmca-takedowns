import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Sklearn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score

print("="*80)
print("BASELINE EXPERIMENT: FASHION KNOCKOFF DETECTION (500 SAMPLES)")
print("="*80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "="*80)
print("1. LOADING DATASET")
print("="*80)

data = torch.load('dataset_clean_real_test.pt')

train_X = data['train']['embeddings'].numpy()
train_y = data['train']['labels'].numpy()

val_X = data['val']['embeddings'].numpy()
val_y = data['val']['labels'].numpy()

test_X = data['test']['embeddings'].numpy()
test_y = data['test']['labels'].numpy()

label_names = ['knockoff', 'similar']

print(f"\nDataset Statistics:")
print(f"  Training:    {len(train_X)} samples")
print(f"  Validation:  {len(val_X)} samples")
print(f"  Testing:     {len(test_X)} samples")
print(f"  Total:       {len(train_X) + len(val_X) + len(test_X)} samples")
print(f"  Features:    {train_X.shape[1]} dimensions (CLIP embeddings)")

print(f"\nClass Distribution:")
print(f"  Train - knockoff: {(train_y == 0).sum()}, similar: {(train_y == 1).sum()}")
print(f"  Val   - knockoff: {(val_y == 0).sum()}, similar: {(val_y == 1).sum()}")
print(f"  Test  - knockoff: {(test_y == 0).sum()}, similar: {(test_y == 1).sum()}")

# ============================================================================
print("\n" + "="*80)
print("2. BASELINE MODELS")
print("="*80)

models = {
    # Linear Models
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'Logistic Regression (L1)': LogisticRegression(
        max_iter=1000,
        penalty='l1',
        solver='saga',
        class_weight='balanced',
        random_state=42
    ),

    # Support Vector Machines
    'SVM (Linear)': SVC(
        kernel='linear',
        class_weight='balanced',
        probability=True,
        random_state=42
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42
    ),

    # Tree-based Models
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    ),

    # Instance-based
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    ),

    # Probabilistic
    'Naive Bayes': GaussianNB()
}

print(f"\nTesting {len(models)} traditional ML baseline models:")
for name in models.keys():
    print(f"  - {name}")

print("\n" + "="*80)
print("3. TRAINING TRADITIONAL ML BASELINES")
print("="*80)

results = []

for name, model in models.items():
    print(f"\n{'─'*60}")
    print(f"Training: {name}")
    print(f"{'─'*60}")

    try:
        # Cross-validation on training set
        cv_scores = cross_val_score(model, train_X, train_y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"  5-Fold CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # Train on training set
        model.fit(train_X, train_y)

        # Evaluate on all splits
        train_preds = model.predict(train_X)
        train_acc = accuracy_score(train_y, train_preds)
        train_f1 = f1_score(train_y, train_preds, average='weighted')

        val_preds = model.predict(val_X)
        val_acc = accuracy_score(val_y, val_preds)
        val_f1 = f1_score(val_y, val_preds, average='weighted')

        test_preds = model.predict(test_X)
        test_acc = accuracy_score(test_y, test_preds)
        test_precision = precision_score(test_y, test_preds, average='weighted')
        test_recall = recall_score(test_y, test_preds, average='weighted')
        test_f1 = f1_score(test_y, test_preds, average='weighted')

        # ROC-AUC if available
        if hasattr(model, 'predict_proba'):
            test_probs = model.predict_proba(test_X)[:, 1]
            roc_auc = roc_auc_score(test_y, test_probs)
        else:
            roc_auc = None

        # Confusion matrix
        cm = confusion_matrix(test_y, test_preds)

        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val Accuracy:   {val_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Test F1 Score:  {test_f1:.4f}")
        if roc_auc:
            print(f"  Test ROC-AUC:   {roc_auc:.4f}")

        results.append({
            'model': name,
            'type': 'Traditional ML',
            'cv_accuracy_mean': cv_mean,
            'cv_accuracy_std': cv_std,
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            'model': name,
            'type': 'Traditional ML',
            'error': str(e)
        })

print("\n" + "="*80)
print("4. NEURAL NETWORK BASELINES")
print("="*80)

# Convert to tensors for PyTorch
train_X_t = torch.from_numpy(train_X).float()
train_y_t = torch.from_numpy(train_y).long()
val_X_t = torch.from_numpy(val_X).float()
val_y_t = torch.from_numpy(val_y).long()
test_X_t = torch.from_numpy(test_X).float()
test_y_t = torch.from_numpy(test_y).long()

# Class weights for imbalanced data
class_counts = torch.bincount(train_y_t)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()

def train_neural_network(model, name, epochs=100, patience=15):
    """Train a neural network with early stopping"""
    print(f"\n{'─'*60}")
    print(f"Training: {name}")
    print(f"{'─'*60}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_dataset = TensorDataset(train_X_t, train_y_t)
    val_dataset = TensorDataset(val_X_t, val_y_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                val_preds.extend(torch.argmax(outputs, dim=1).numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_y, val_preds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate
    with torch.no_grad():
        train_preds = torch.argmax(model(train_X_t), dim=1).numpy()
        val_preds = torch.argmax(model(val_X_t), dim=1).numpy()
        test_outputs = model(test_X_t)
        test_preds = torch.argmax(test_outputs, dim=1).numpy()
        test_probs = torch.softmax(test_outputs, dim=1)[:, 1].numpy()

    train_acc = accuracy_score(train_y, train_preds)
    val_acc = accuracy_score(val_y, val_preds)
    test_acc = accuracy_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds, average='weighted')
    test_precision = precision_score(test_y, test_preds, average='weighted')
    test_recall = recall_score(test_y, test_preds, average='weighted')
    roc_auc = roc_auc_score(test_y, test_probs)
    cm = confusion_matrix(test_y, test_preds)

    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy:   {val_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Test F1 Score:  {test_f1:.4f}")
    print(f"  Test ROC-AUC:   {roc_auc:.4f}")

    return {
        'model': name,
        'type': 'Neural Network',
        'cv_accuracy_mean': None,
        'cv_accuracy_std': None,
        'train_accuracy': train_acc,
        'train_f1': f1_score(train_y, train_preds, average='weighted'),
        'val_accuracy': val_acc,
        'val_f1': f1_score(val_y, val_preds, average='weighted'),
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }

# Simple Linear Neural Network (no hidden layers)
class LinearNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 2)

    def forward(self, x):
        return self.linear(x)

# Shallow MLP (1 hidden layer)
class ShallowMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

# Medium MLP (2 hidden layers)
class MediumMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

# Deep MLP (3 hidden layers) - Our proposed model
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

# Train neural network baselines
nn_models = [
    (LinearNN(), "Linear NN (No Hidden Layers)"),
    (ShallowMLP(), "Shallow MLP (1 Hidden Layer)"),
    (MediumMLP(), "Medium MLP (2 Hidden Layers)"),
    (DeepMLP(), "Deep MLP (3 Hidden Layers) [Proposed]"),
]

for model, name in nn_models:
    result = train_neural_network(model, name)
    results.append(result)


print("\n" + "="*80)
print("5. RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values('test_accuracy', ascending=False)

print("\n" + "─"*100)
print("ALL BASELINES RANKED BY TEST ACCURACY")
print("─"*100)
print(f"\n{'Model':<35} {'Type':<15} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Test F1':>10}")
print("─"*100)

for _, row in results_df_sorted.iterrows():
    train_acc = f"{row['train_accuracy']:.4f}" if pd.notna(row.get('train_accuracy')) else "N/A"
    val_acc = f"{row['val_accuracy']:.4f}" if pd.notna(row.get('val_accuracy')) else "N/A"
    test_acc = f"{row['test_accuracy']:.4f}" if pd.notna(row.get('test_accuracy')) else "N/A"
    test_f1 = f"{row['test_f1']:.4f}" if pd.notna(row.get('test_f1')) else "N/A"
    model_type = row.get('type', 'Unknown')[:14]
    print(f"{row['model']:<35} {model_type:<15} {train_acc:>10} {val_acc:>10} {test_acc:>10} {test_f1:>10}")

# Best models by category
print("\n" + "="*80)
print("BEST MODELS BY CATEGORY")
print("="*80)

traditional_df = results_df[results_df['type'] == 'Traditional ML']
nn_df = results_df[results_df['type'] == 'Neural Network']

if len(traditional_df) > 0:
    best_traditional = traditional_df.sort_values('test_accuracy', ascending=False).iloc[0]
    print(f"\nBest Traditional ML: {best_traditional['model']}")
    print(f"  Test Accuracy: {best_traditional['test_accuracy']:.4f} ({best_traditional['test_accuracy']*100:.1f}%)")
    print(f"  Test F1 Score: {best_traditional['test_f1']:.4f}")

if len(nn_df) > 0:
    best_nn = nn_df.sort_values('test_accuracy', ascending=False).iloc[0]
    print(f"\nBest Neural Network: {best_nn['model']}")
    print(f"  Test Accuracy: {best_nn['test_accuracy']:.4f} ({best_nn['test_accuracy']*100:.1f}%)")
    print(f"  Test F1 Score: {best_nn['test_f1']:.4f}")

best_overall = results_df_sorted.iloc[0]
print(f"\nBest Overall: {best_overall['model']}")
print(f"  Test Accuracy: {best_overall['test_accuracy']:.4f} ({best_overall['test_accuracy']*100:.1f}%)")
print(f"  Test F1 Score: {best_overall['test_f1']:.4f}")


print("\n" + "="*80)
print("6. SAVING RESULTS")
print("="*80)

# Save to CSV
results_df.to_csv('baseline_results_500.csv', index=False)
print(f"  Saved to: baseline_results_500.csv")

# Save detailed results to JSON
experiment_log = {
    'experiment_name': 'Baseline Comparison (500 samples)',
    'date': datetime.now().isoformat(),
    'dataset': {
        'train_samples': len(train_X),
        'val_samples': len(val_X),
        'test_samples': len(test_X),
        'total_samples': len(train_X) + len(val_X) + len(test_X),
        'feature_dim': train_X.shape[1],
        'classes': label_names
    },
    'models_tested': len(results),
    'best_traditional_ml': {
        'name': best_traditional['model'],
        'test_accuracy': float(best_traditional['test_accuracy']),
        'test_f1': float(best_traditional['test_f1'])
    } if len(traditional_df) > 0 else None,
    'best_neural_network': {
        'name': best_nn['model'],
        'test_accuracy': float(best_nn['test_accuracy']),
        'test_f1': float(best_nn['test_f1'])
    } if len(nn_df) > 0 else None,
    'best_overall': {
        'name': best_overall['model'],
        'test_accuracy': float(best_overall['test_accuracy']),
        'test_f1': float(best_overall['test_f1'])
    },
    'all_results': results
}

with open('baseline_experiment_log_500.json', 'w') as f:
    json.dump(experiment_log, f, indent=2, default=str)
print(f"  Saved to: baseline_experiment_log_500.json")


print("\n" + "="*80)
print("7. SUMMARY TABLE (FOR INTERIM REPORT)")
print("="*80)

print("""
Copy this table for your report:

| Model                              | Type           | Train Acc | Val Acc | Test Acc | Test F1 |
|------------------------------------|----------------|-----------|---------|----------|---------|""")

for _, row in results_df_sorted.iterrows():
    train_acc = f"{row['train_accuracy']:.3f}" if pd.notna(row.get('train_accuracy')) else "N/A"
    val_acc = f"{row['val_accuracy']:.3f}" if pd.notna(row.get('val_accuracy')) else "N/A"
    test_acc = f"{row['test_accuracy']:.3f}" if pd.notna(row.get('test_accuracy')) else "N/A"
    test_f1 = f"{row['test_f1']:.3f}" if pd.notna(row.get('test_f1')) else "N/A"
    model_type = row.get('type', 'Unknown')
    print(f"| {row['model']:<34} | {model_type:<14} | {train_acc:>9} | {val_acc:>7} | {test_acc:>8} | {test_f1:>7} |")

print("\n" + "="*80)
print("BASELINE EXPERIMENT COMPLETE")
print("="*80)