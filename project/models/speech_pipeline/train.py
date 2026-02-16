import os
import glob
import json
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ============================================================================
# Model Architecture
# ============================================================================
class SpeechEmotionTransformer(nn.Module):
    def __init__(self, num_labels=7):
        super().__init__()

        # Block 3: Temporal Modelling [cite: 36, 37]
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.temporal_modeller = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Block 5: Classifier [cite: 41, 42]
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
# Data Loading & Processing
# ============================================================================
def load_tess_data(data_dir):
    data = []

    # 1. Expanded mapping for TESS inconsistencies
    emotion_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'pleasant_surprise': 5,
        'sad': 6
    }

    # 2. Normalization map for common folder/file typos
    label_norm = {
        'surprised': 'pleasant_surprise',
        'surprise': 'pleasant_surprise',
    }

    if not os.path.exists(data_dir):
        print(f"Error: Path {data_dir} does not exist.")
        return pd.DataFrame()

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)

        # Skip hidden files like .DS_Store
        if not os.path.isdir(folder_path) or folder.startswith('.'):
            continue

        # Extract raw label (e.g., 'angry' from 'OAF_angry')
        raw_label = folder.split('_')[-1].lower()

        # Normalize the label
        normalized_label = label_norm.get(raw_label, raw_label)

        if normalized_label in emotion_map:
            label_id = emotion_map[normalized_label]
            # Use glob to find all wav files in the subdirectory
            files = glob.glob(os.path.join(folder_path, "*.wav"))
            for file_path in files:
                data.append({
                    "path": file_path,
                    "label": label_id,
                    "emotion_name": normalized_label # Useful for Error Analysis
                })
        else:
            print(f"Warning: Skipping unknown emotion folder: {folder}")

    df = pd.DataFrame(data)
    if not df.empty:
        print(f"Loaded {len(df)} samples across {len(df['label'].unique())} emotions.")
    return df

def extract_and_cache(df, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    all_features, all_labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting Features to {output_path}"):
        waveform, sr = torchaudio.load(row['path'])
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        # Trim silence (Preprocessing) [cite: 31]
        energy = waveform.abs().squeeze()
        mask = energy > (energy.max() * 0.01)
        if mask.any():
            indices = torch.where(mask)[0]
            waveform = waveform[:, indices[0]:indices[-1]+1]

        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)

        with torch.no_grad():
            # Feature Extraction block [cite: 33, 34]
            features = model(inputs).last_hidden_state.squeeze(0).cpu()

        all_features.append(features)
        all_labels.append(row['label'])

    # Pad sequences & Create Attention Mask [cite: 31]
    padded_x = torch.nn.utils.rnn.pad_sequence(all_features, batch_first=True)
    attention_mask = torch.zeros(padded_x.shape[:2])
    for i, f in enumerate(all_features):
        attention_mask[i, :f.shape[0]] = 1

    torch.save({'x': padded_x, 'm': attention_mask, 'y': torch.tensor(all_labels)}, output_path)

# ============================================================================
# Utility Functions
# ============================================================================
def get_class_names(csv_path="data_splits/train_split.csv"):
    """Extract ordered class names from CSV split"""
    if not os.path.exists(csv_path):
        return ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']
    df = pd.read_csv(csv_path)
    class_names = df.groupby('label')['emotion_name'].first().sort_index().tolist()
    return class_names

def plot_training_history(history, save_path="results/plots/speech_training_curves.png"):
    """Plot training/validation loss and accuracy"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Learning Rate
    axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path="results/plots/speech_confusion_matrix.png"):
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def save_metrics_table(history, class_names, cm, save_path="results/speech_metrics.txt"):
    """Save accuracy metrics as a text table"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SPEECH-ONLY EMOTION RECOGNITION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        # Overall metrics
        f.write(f"Best Validation Accuracy: {max(history['val_acc']):.4f}\n")
        f.write(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}\n\n")

        # Per-class accuracy
        f.write("-" * 60 + "\n")
        f.write("PER-CLASS ACCURACY\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Emotion':<20} {'Accuracy':<10} {'Correct':<10} {'Total':<10}\n")
        f.write("-" * 60 + "\n")

        for i, emotion in enumerate(class_names):
            correct = cm[i, i]
            total = cm[i, :].sum()
            accuracy = correct / total if total > 0 else 0
            f.write(f"{emotion:<20} {accuracy:<10.4f} {correct:<10} {total:<10}\n")

        f.write("-" * 60 + "\n")

    print(f"Metrics table saved to {save_path}")

# ============================================================================
# Training Function
# ============================================================================
def run_training(train_path, val_path, class_names, model_save_path):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print(f"Classes: {class_names}\n")

    train_loader = DataLoader(FeatureDataset(train_path), batch_size=32, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_path), batch_size=32)

    model = SpeechEmotionTransformer(num_labels=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    print("Starting training...\n")
    print("=" * 80)
    epoch_num = 30

    for epoch in range(epoch_num):
        # ========== Training Phase ==========
        model.train()
        total_loss, correct = 0, 0

        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, m)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        # ========== Validation Phase ==========
        model.eval()
        val_correct = 0
        val_total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, m, y in val_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
                logits = model(x, m)
                loss = criterion(logits, y)

                val_total_loss += loss.item()
                preds = logits.argmax(1)
                val_correct += (preds == y).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss = val_total_loss / len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)

        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Print progress
        print(f"Epoch {epoch+1:02d}/{epoch_num} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            cm = confusion_matrix(all_labels, all_preds)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'confusion_matrix': cm,
                'class_names': class_names
            }, model_save_path)
            print(f"  → New best model saved! (Val Acc: {val_acc:.4f})")
            
        scheduler.step(val_acc)

    print("=" * 80)
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}\n")

    return history

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "TESS_Toronto_emotional_speech_set_data" 
    
    # Paths for outputs
    TRAIN_FEAT_PATH = "train_features.pt"
    VAL_FEAT_PATH = "val_features.pt"
    TEST_FEAT_PATH = "test_features.pt"
    
    TRAIN_SPLIT_CSV = "data_splits/train_split.csv"
    VAL_SPLIT_CSV = "data_splits/val_split.csv"
    TEST_SPLIT_CSV = "data_splits/test_split.csv"
    
    MODEL_SAVE_PATH = "models/speech_pipeline/best_speech_model.pt"
    HISTORY_SAVE_PATH = "models/speech_pipeline/training_history.pt"
    
    RESULTS_DIR = "results"
    PLOTS_DIR = "results/plots"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs("data_splits", exist_ok=True)

    # 1. Load and Split Data
    if os.path.exists(DATA_DIR):
        print(f"Loading data from {DATA_DIR}...")
        df = load_tess_data(DATA_DIR)
        
        if not df.empty:
            train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

            # Save splits
            train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
            val_df.to_csv(VAL_SPLIT_CSV, index=False)
            test_df.to_csv(TEST_SPLIT_CSV, index=False)
            
            # Extract features
            extract_and_cache(train_df, TRAIN_FEAT_PATH)
            extract_and_cache(val_df, VAL_FEAT_PATH)
            extract_and_cache(test_df, TEST_FEAT_PATH)
            
            # Save split info
            split_info = {
                "random_seed": 42,
                "test_size": 0.15,
                "val_size": 0.15,
                "train_size": 0.70,
                "stratified": True,
                "total_samples": len(df),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "emotion_distribution": {
                    "train": train_df['emotion_name'].value_counts().to_dict(),
                    "val": val_df['emotion_name'].value_counts().to_dict(),
                    "test": test_df['emotion_name'].value_counts().to_dict()
                }
            }
            with open("split_info.json", "w") as f:
                json.dump(split_info, f, indent=2)
    else:
        print(f"Data directory {DATA_DIR} not found. checking for existing features/splits...")

    # 2. Train the model
    if os.path.exists(TRAIN_FEAT_PATH) and os.path.exists(VAL_FEAT_PATH):
        # Get class names
        class_names = get_class_names(TRAIN_SPLIT_CSV)
        
        history = run_training(TRAIN_FEAT_PATH, VAL_FEAT_PATH, class_names, MODEL_SAVE_PATH)

        # Save training history
        torch.save(history, HISTORY_SAVE_PATH)

        # Plot training curves
        plot_training_history(history, save_path=os.path.join(PLOTS_DIR, "speech_training_curves.png"))

        # Load best model and plot confusion matrix
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=torch.device("cpu"), weights_only=False)
        cm = checkpoint['confusion_matrix']

        plot_confusion_matrix(cm, class_names, save_path=os.path.join(PLOTS_DIR, "speech_confusion_matrix.png"))

        # Save metrics table
        save_metrics_table(history, class_names, cm, save_path=os.path.join(RESULTS_DIR, "speech_metrics.txt"))

        print("\n✓ All results saved!")
        print(f"  - Model: {MODEL_SAVE_PATH}")
        print(f"  - Training curves: {os.path.join(PLOTS_DIR, 'speech_training_curves.png')}")
        print(f"  - Confusion matrix: {os.path.join(PLOTS_DIR, 'speech_confusion_matrix.png')}")
        print(f"  - Metrics table: {os.path.join(RESULTS_DIR, 'speech_metrics.txt')}")
    else:
        print("Feature files not found. Please ensure data is loaded and features extracted.")
