import torch
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class TextEmotionTransformer(nn.Module):
    def __init__(self, num_labels=7, hidden_dim=768):
        super().__init__()

        # Block 3: Contextual Modelling 
        # We use a Transformer Encoder to further process the RoBERTa embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.contextual_modeller = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Block 5: Classifier 
        # Using a slightly denser classifier to handle the semantic richness of text
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # Lower dropout than speech as text is less noisy
            nn.Linear(256, num_labels)
        )

    def forward(self, x, mask):
        # x: [batch_size, seq_len, 768]
        # mask: [batch_size, seq_len] (1 for data, 0 for padding)

        # Transformer expects True for tokens that should be IGNORED (padding)
        src_key_padding_mask = (mask == 0)

        # Contextual Modelling: Learn meaning across tokens
        c_out = self.contextual_modeller(x, src_key_padding_mask=src_key_padding_mask)

        # Masked Global Average Pooling
        # This reduces the sequence to a single 'emotional' vector 
        mask_expanded = mask.unsqueeze(-1) # [batch_size, seq_len, 1]
        pooled = (c_out * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        # Predict the emotion label 
        return self.classifier(pooled)


class FeatureDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, weights_only=False)
        self.x, self.m, self.y = data['x'], data['m'], data['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.m[i], self.y[i]


def get_class_names(csv_path="/content/test_split.csv"):
    """Extract ordered class names from CSV split"""
    df = pd.read_csv(csv_path)
    class_names = df.groupby('label')['emotion_name'].first().sort_index().tolist()
    return class_names


def plot_confusion_matrix(cm, class_names, save_path="results/confusion_matrix.png"):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))

    # Calculate percentages for annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Speech Emotion Recognition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def run_test(model_path, test_data_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 1. Load the Test Dataset
    test_loader = DataLoader(FeatureDataset(test_data_path), batch_size=32, shuffle=False)

    # 2. Initialize and Load Model
    model = TextEmotionTransformer(num_labels=len(class_names)).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    # 3. Inference Loop
    print("Running inference...")
    with torch.no_grad():
        for x, m, y in test_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            logits = model(x, m)

            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 4. Calculate Metrics
    test_cm = confusion_matrix(all_labels, all_preds)

    # Print Classification Report (Precision, Recall, F1)
    print("\n" + "="*30)
    print("TEST SET PERFORMANCE")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return test_cm


if __name__ == "__main__":
    # Create the directory structure if it doesn't exist
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    class_names = get_class_names()

    # Replace with your actual test file path
    test_path = "/content/drive/MyDrive/Assignment 2_SLFI/Text_Features/text_test_features.pt"
    model_path = "/content/models/content/models/text_pipeline/best_text_model.pt"

    # Run test
    test_cm = run_test(model_path, test_path, class_names)

    # Plot using your existing function
    plot_confusion_matrix(
        test_cm,
        class_names,
        save_path="results/plots/test_confusion_matrix.png"
    )
