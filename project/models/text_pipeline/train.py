import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_tess_text_data(data_dir):
    """
    Load TESS data and extract text transcripts and string emotion labels.
    """
    data = []

    # Mapping for classification 
    emotion_map = {
        'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
        'neutral': 4, 'pleasant_surprise': 5, 'sad': 6
    }

    # Normalization for TESS folder naming inconsistencies
    label_norm = {'surprised': 'pleasant_surprise', 'surprise': 'pleasant_surprise'}

    if not os.path.exists(data_dir):
        print(f"Error: Path {data_dir} does not exist.")
        return pd.DataFrame(), emotion_map

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path) or folder.startswith('.'):
            continue

        # Extract raw label from folder name (e.g., 'angry' from 'OAF_angry')
        raw_label = folder.split('_')[-1].lower()
        normalized_label = label_norm.get(raw_label, raw_label)

        if normalized_label in emotion_map:
            label_id = emotion_map[normalized_label]
            files = glob.glob(os.path.join(folder_path, "*.wav"))

            for file_path in files:
                filename = os.path.basename(file_path)
                parts = filename.replace('.wav', '').split('_')

                if len(parts) >= 2:
                    target_word = parts[1]
                    # Providing full context for the Contextual Modelling block
                    full_text = f"Say the word {target_word}"

                    data.append({
                        "path": file_path,
                        "text": full_text,
                        "label": label_id,
                        "emotion_name": normalized_label # Added the string emotion column
                    })

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples across {len(df['label'].unique())} emotions.")
    return df, emotion_map


def get_text_from_speech_df(speech_df):
    """
    Extracts the 'Say the word [word]' phrase from the existing speech CSV paths.
    """
    data = []
    for _, row in speech_df.iterrows():
        # Get filename from the existing path in your speech CSV
        filename = os.path.basename(row['path'])

        # Extract word: 'OAF_back_angry.wav' -> ['OAF', 'back', 'angry']
        parts = filename.replace('.wav', '').split('_')

        if len(parts) >= 2:
            target_word = parts[1]
            full_text = f"Say the word {target_word}"

            data.append({
                "path": row['path'],      # Keep path for matching
                "text": full_text,        # New text feature
                "label": row['label'],    # Keep original label
                "emotion_name": row.get('emotion_name', '') # Keep string label
            })

    return pd.DataFrame(data)


def extract_and_cache_text(df, output_path):
    """
    Extract RoBERTa features and save labels for the Text Pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Preprocessing Block 
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base").to(device).eval()

    all_features, all_labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Text Features"):
        # Preprocessing: Tokenization 
        inputs = tokenizer(row['text'], return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            # 2. Feature Extraction & 3. Contextual Modelling
            outputs = model(**inputs)
            features = outputs.last_hidden_state.squeeze(0).cpu() # (seq_len, 768)

        all_features.append(features)
        all_labels.append(row['label'])

    # Dynamic padding to align sequences 
    padded_x = torch.nn.utils.rnn.pad_sequence(all_features, batch_first=True)

    # Create Attention Mask for the Classifier 
    attention_mask = torch.zeros(padded_x.shape[:2])
    for i, f in enumerate(all_features):
        attention_mask[i, :f.shape[0]] = 1

    # Save features, masks, and labels
    torch.save({
        'x': padded_x,
        'm': attention_mask,
        'y': torch.tensor(all_labels)
    }, output_path)

    print(f"Saved features to {output_path}. Feature shape: {padded_x.shape}")


def extract_text_features_from_split(csv_path, output_pt_path):
    # Load your existing speech split
    speech_df = pd.read_csv(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the text dataframe based on those exact files
    text_df = get_text_from_speech_df(speech_df)

    # 1. Preprocessing Block 
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base").to(device).eval()
    all_features, all_labels = [], []

    for _, row in tqdm(text_df.iterrows(), total=len(text_df), desc=f"Processing {csv_path}"):
        inputs = tokenizer(row['text'], return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            # 2. Feature Extraction & 3. Contextual Modelling 
            outputs = model(**inputs)
            features = outputs.last_hidden_state.squeeze(0).cpu()

        all_features.append(features)
        all_labels.append(row['label'])

    # Pad to align for the Classifier block 
    padded_x = torch.nn.utils.rnn.pad_sequence(all_features, batch_first=True)

    # Create Attention Mask
    attention_mask = torch.zeros(padded_x.shape[:2])
    for i, f in enumerate(all_features):
        attention_mask[i, :f.shape[0]] = 1

    # Save the aligned features
    torch.save({
        'x': padded_x,
        'm': attention_mask,
        'y': torch.tensor(all_labels)
    }, output_pt_path)

    print(f"Successfully aligned and saved text features to {output_pt_path}")


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


class TextDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path, weights_only=False)
        self.x = data['x']  # Features (num_samples, seq_len, 768)
        self.m = data['m']  # Attention mask
        self.y = data['y']  # Labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.m[idx], self.y[idx]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x, m)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, m, y in loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            outputs = model(x, m)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / len(loader.dataset), all_preds, all_labels


def run_text_training(epochs=30, batch_size=32, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Aligned Datasets
    train_ds = TextDataset("text_train_features.pt")
    val_ds = TextDataset("text_val_features.pt")
    test_ds = TextDataset("text_test_features.pt")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Initialize Model (From previous architecture block)
    model = TextEmotionTransformer(num_labels=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/text_pipeline/best_text_model.pt")

    # Final Test Evaluation
    model.load_state_dict(torch.load("models/text_pipeline/best_text_model.pt"))
    _, test_acc, y_pred, y_true = validate(model, test_loader, criterion, device)

    cm = confusion_matrix(y_true, y_pred)
    return history, cm, test_acc


def get_class_names(csv_path="data/train_split.csv"):
    """Extract ordered class names from CSV split"""
    df = pd.read_csv(csv_path)
    class_names = df.groupby('label')['emotion_name'].first().sort_index().tolist()
    return class_names


def plot_text_results(history, cm, class_names):
    # 1. Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Text Model: Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Text Model: Accuracy')
    plt.legend()
    plt.savefig("results/plots/text_training_curves.png")

    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Text Emotion Recognition')
    plt.savefig("results/plots/text_confusion_matrix.png")


def save_text_metrics_table(history, class_names, cm, save_path="results/text_metrics.txt"):
    """Save accuracy metrics for the Text-only model as a text table"""
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TEXT-ONLY EMOTION RECOGNITION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        # Overall metrics
        f.write(f"Best Validation Accuracy: {max(history['val_acc']):.4f}\n")
        f.write(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}\n\n")

        # Per-class accuracy 
        f.write("-" * 60 + "\n")
        f.write("PER-CLASS ACCURACY (Text Baseline)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Emotion':<20} {'Accuracy':<10} {'Correct':<10} {'Total':<10}\n")
        f.write("-" * 60 + "\n")

        for i, emotion in enumerate(class_names):
            correct = cm[i, i]
            total = cm[i, :].sum()
            accuracy = correct / total if total > 0 else 0
            f.write(f"{emotion:<20} {accuracy:<10.4f} {correct:<10} {total:<10}\n")

        f.write("-" * 60 + "\n")
        f.write("\nNote: Low text accuracy is expected for TESS as text is static across emotions.")

    print(f"Text metrics table saved to {save_path}")


if __name__ == "__main__":
    # Ensure directories exist 
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("models/text_pipeline", exist_ok=True)

    # Get class names from your split to ensure alignment
    class_names = get_class_names("/content/train_split.csv")

    # 1. Train the model
    history, cm, test_acc = run_text_training()

    # 2. Plot results for the report 
    plot_text_results(history, cm, class_names)

    # 3. Save the metrics table (Matching your speech format) 
    save_text_metrics_table(history, class_names, cm, save_path="results/text_metrics.txt")

    print("\n✓ Text Pipeline Results Saved!")
    print("  - Metrics Table: results/text_metrics.txt")
