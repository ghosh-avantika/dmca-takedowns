"""
Computer Vision Knockoff Detection System
Uses CLIP to encode both images and text, then trains an MLP classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ============================================================================
# 1. DATASET CLASS
# ============================================================================
class KnockoffDataset(Dataset):
    """Dataset for knockoff detection with images and text"""

    def __init__(self, image_paths, text_descriptions, labels, clip_preprocess):
        self.image_paths = image_paths
        self.text_descriptions = text_descriptions
        self.labels = labels
        self.clip_preprocess = clip_preprocess

        # Convert labels to numeric: knockoff=0, similar=1, original=2
        self.label_map = {'knockoff': 0, 'similar': 1, 'original': 2}
        self.numeric_labels = [self.label_map.get(l, 1) for l in labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.clip_preprocess(image)
        except:
            # If image fails, create blank image
            image = torch.zeros(3, 224, 224)

        text = self.text_descriptions[idx]
        label = self.numeric_labels[idx]

        return image, text, label


# ============================================================================
# 2. FEATURE EXTRACTOR (CLIP-based)
# ============================================================================
class CLIPFeatureExtractor:
    """Extract visual and text features using CLIP"""

    def __init__(self, device='cpu'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

    def extract_image_features(self, images):
        """Extract features from images (batch)"""
        with torch.no_grad():
            features = self.model.encode_image(images.to(self.device))
            features = features / features.norm(dim=1, keepdim=True)
        return features.cpu()

    def extract_text_features(self, texts):
        """Extract features from text descriptions (batch)"""
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=1, keepdim=True)
        return features.cpu()

    def extract_combined_features(self, images, texts):
        """Extract and concatenate image + text features"""
        img_features = self.extract_image_features(images)
        txt_features = self.extract_text_features(texts)
        # Concatenate: 512 (image) + 512 (text) = 1024 dims
        return torch.cat([img_features, txt_features], dim=1)


# ============================================================================
# 3. MLP CLASSIFIER
# ============================================================================
class KnockoffClassifier(nn.Module):
    """MLP classifier for knockoff detection"""

    def __init__(self, input_dim=1024, hidden_dims=[512, 256, 128], num_classes=3, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ============================================================================
# 4. TRAINING PIPELINE
# ============================================================================
class KnockoffDetector:
    """Complete training and inference pipeline"""

    def __init__(self, device='cpu', use_text=True, use_image=True):
        self.device = device
        self.use_text = use_text
        self.use_image = use_image

        # Determine input dimension
        if use_text and use_image:
            input_dim = 1024  # 512 image + 512 text
        else:
            input_dim = 512  # Just one modality

        self.feature_extractor = CLIPFeatureExtractor(device)
        self.classifier = KnockoffClassifier(input_dim=input_dim).to(device)

        self.label_names = ['knockoff', 'similar', 'original']

    def train(self, train_loader, val_loader, epochs=20, lr=0.001):
        """Train the classifier"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=lr)

        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Training
            self.classifier.train()
            total_loss = 0

            for images, texts, labels in train_loader:
                optimizer.zero_grad()

                # Extract features
                if self.use_image and self.use_text:
                    features = self.feature_extractor.extract_combined_features(images, texts)
                elif self.use_image:
                    features = self.feature_extractor.extract_image_features(images)
                else:
                    features = self.feature_extractor.extract_text_features(texts)

                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.classifier(features)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Validation
            val_acc = self.evaluate(val_loader)
            val_accuracies.append(val_acc)

            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        return train_losses, val_accuracies

    def evaluate(self, data_loader):
        """Evaluate the classifier"""
        self.classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, texts, labels in data_loader:
                if self.use_image and self.use_text:
                    features = self.feature_extractor.extract_combined_features(images, texts)
                elif self.use_image:
                    features = self.feature_extractor.extract_image_features(images)
                else:
                    features = self.feature_extractor.extract_text_features(texts)

                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.classifier(features)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def predict(self, image_path, text_description):
        """Predict for a single design"""
        self.classifier.eval()

        with torch.no_grad():
            # Load image
            image = Image.open(image_path).convert('RGB')
            image = self.feature_extractor.preprocess(image).unsqueeze(0)

            # Extract features
            if self.use_image and self.use_text:
                features = self.feature_extractor.extract_combined_features(image, [text_description])
            elif self.use_image:
                features = self.feature_extractor.extract_image_features(image)
            else:
                features = self.feature_extractor.extract_text_features([text_description])

            features = features.to(self.device)

            # Predict
            outputs = self.classifier(features)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()

            return {
                'prediction': self.label_names[predicted_class],
                'probabilities': {
                    label: prob.item()
                    for label, prob in zip(self.label_names, probabilities)
                }
            }

    def save(self, path):
        """Save the trained model"""
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'use_text': self.use_text,
            'use_image': self.use_image
        }, path)

    def load(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.use_text = checkpoint['use_text']
        self.use_image = checkpoint['use_image']


# ============================================================================
# 5. SIMILARITY COMPARISON SYSTEM
# ============================================================================
class VisualSimilarityDetector:
    """Compare designs visually to detect copies"""

    def __init__(self, device='cpu'):
        self.device = device
        self.feature_extractor = CLIPFeatureExtractor(device)

    def compute_similarity(self, image1_path, image2_path, text1=None, text2=None):
        """
        Compute similarity between two designs
        Returns score from 0 (different) to 1 (identical)
        """
        # Load images
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')

        img1 = self.feature_extractor.preprocess(img1).unsqueeze(0)
        img2 = self.feature_extractor.preprocess(img2).unsqueeze(0)

        # Extract image features
        feat1_img = self.feature_extractor.extract_image_features(img1)
        feat2_img = self.feature_extractor.extract_image_features(img2)

        # Image similarity
        img_sim = (feat1_img @ feat2_img.T).item()

        # If text provided, also compute text similarity
        if text1 and text2:
            feat1_txt = self.feature_extractor.extract_text_features([text1])
            feat2_txt = self.feature_extractor.extract_text_features([text2])
            txt_sim = (feat1_txt @ feat2_txt.T).item()

            # Combined similarity (average)
            return {
                'image_similarity': img_sim,
                'text_similarity': txt_sim,
                'combined_similarity': (img_sim + txt_sim) / 2,
                'is_likely_copy': (img_sim + txt_sim) / 2 > 0.85
            }

        return {
            'image_similarity': img_sim,
            'is_likely_copy': img_sim > 0.90
        }

    def find_similar_designs(self, query_image, database_images, top_k=5):
        """Find the most visually similar designs in a database"""
        # Extract query features
        query_img = Image.open(query_image).convert('RGB')
        query_img = self.feature_extractor.preprocess(query_img).unsqueeze(0)
        query_feat = self.feature_extractor.extract_image_features(query_img)

        similarities = []

        for db_image in database_images:
            db_img = Image.open(db_image).convert('RGB')
            db_img = self.feature_extractor.preprocess(db_img).unsqueeze(0)
            db_feat = self.feature_extractor.extract_image_features(db_img)

            sim = (query_feat @ db_feat.T).item()
            similarities.append((db_image, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


# ============================================================================
# 6. EXAMPLE USAGE
# ============================================================================
def main():
    """Example usage of the knockoff detection system"""

    print("="*80)
    print("KNOCKOFF DETECTION SYSTEM - Computer Vision Pipeline")
    print("="*80)

    # Check if GPU available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    print("\n" + "="*80)
    print("SYSTEM COMPONENTS")
    print("="*80)
    print("\n1. CLIPFeatureExtractor - Extracts visual & text features")
    print("2. KnockoffClassifier - MLP that predicts knockoff/similar/original")
    print("3. VisualSimilarityDetector - Compares designs for copying")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Collect images for your dataset:")
    print("   - Create a folder: mkdir images/originals images/copies")
    print("   - Download or scrape images of the designs from your dataset")

    print("\n2. Prepare your dataset CSV:")
    print("   - Columns: image_path, design_text, label (knockoff/similar/original)")

    print("\n3. Train the model:")
    print("   See example code in train_knockoff_detector.py")

    print("\n" + "="*80)
    print("SIMILARITY THRESHOLDS (based on CLIP features)")
    print("="*80)
    print("   > 0.95 : Near-identical (definite copy)")
    print("   0.85-0.95 : Very similar (likely knockoff)")
    print("   0.75-0.85 : Similar design elements")
    print("   0.60-0.75 : Some resemblance")
    print("   < 0.60 : Different designs")


if __name__ == "__main__":
    main()
