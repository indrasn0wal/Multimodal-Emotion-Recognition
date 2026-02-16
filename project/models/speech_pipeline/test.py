import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================================
# Model Architecture
# ============================================================================
class SpeechEmotionTransformer(nn.Module):
    def __init__(self, num_labels=7):
        super().__init__()

        # Block 3: Temporal Modelling
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.temporal_modeller = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Block 5: Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, x, mask):
        # mask is 1 for data, 0 for padding. Transformer needs True for padding.
        src_key_padding_mask = (mask == 0)

        # Learn emotional patterns over time
        t_out = self.temporal_modeller(x, src_key_padding_mask=src_key_padding_mask)

        # Masked Mean Pooling (Prevents silence from diluting the signal)
        mask_expanded = mask.unsqueeze(-1)
        pooled = (t_out * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        return self.classifier(pooled)

# ============================================================================
# Dataset Class
# ============================================================================
class FeatureDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.x, self.m, self.y = data['x'], data['m'], data['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.m[i], self.y[i]

# ============================================================================
# Utility Functions
# ============================================================================
def get_class_names(csv_path="data_splits/test_split.csv"):
    """Extract ordered class names from CSV split"""
    if not os.path.exists(csv_path):
        return ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']
    df = pd.read_csv(csv_path)
    class_names = df.groupby('label')['emotion_name'].first().sort_index().tolist()
    return class_names

def plot_confusion_matrix(cm, class_names, save_path="results/plots/test_confusion_matrix.png"):
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
    plt.title('Confusion Matrix - Speech Emotion Recognition (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

# ============================================================================
# Test Function
# ============================================================================
def run_test(model_path, test_data_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 1. Load the Test Dataset
    test_loader = DataLoader(FeatureDataset(test_data_path), batch_size=32, shuffle=False)

    # 2. Initialize and Load Model
    model = SpeechEmotionTransformer(num_labels=len(class_names)).to(device)
    
    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
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

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    # Configuration
    TEST_FEAT_PATH = "test_features.pt"
    MODEL_PATH = "models/speech_pipeline/best_speech_model.pt"
    TEST_SPLIT_CSV = "data_splits/test_split.csv"
    
    # Output paths
    RESULTS_DIR = "results"
    PLOTS_DIR = "results/plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if os.path.exists(TEST_FEAT_PATH) and os.path.exists(MODEL_PATH):
        # Get class names
        class_names = get_class_names(TEST_SPLIT_CSV)
        
        # Run test
        test_cm = run_test(MODEL_PATH, TEST_FEAT_PATH, class_names)
        
        if test_cm is not None:
            # Plot confusion matrix
            plot_confusion_matrix(
                test_cm,
                class_names,
                save_path=os.path.join(PLOTS_DIR, "test_confusion_matrix.png")
            )
            print("\n✓ Test completed!")
            print(f"  - Confusion matrix: {os.path.join(PLOTS_DIR, 'test_confusion_matrix.png')}")
    else:
        print("Required files not found:")
        if not os.path.exists(TEST_FEAT_PATH):
            print(f"  - Test features: {TEST_FEAT_PATH} (Missing)")
        if not os.path.exists(MODEL_PATH):
            print(f"  - Model: {MODEL_PATH} (Missing)")
        print("Please run train.py first.")
