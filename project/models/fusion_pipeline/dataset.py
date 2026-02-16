"""
Multimodal Emotion Recognition - Dataset and Data Loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class MultimodalFeatureDataset(Dataset):
    """Dataset for pre-extracted multimodal features"""
    def __init__(self, speech_path, text_path):
        # Load speech features: {'x': features, 'm': mask, 'y': labels}
        s_data = torch.load(speech_path)
        # Load text features: {'x': features, 'm': mask, 'y': labels}
        t_data = torch.load(text_path)

        self.speech_x, self.speech_m = s_data['x'], s_data['m']
        self.text_x, self.text_m = t_data['x'], t_data['m']
        self.labels = s_data['y']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.speech_x[i], self.speech_m[i],
                self.text_x[i], self.text_m[i],
                self.labels[i])


def get_class_names(csv_path="data/train_split.csv"):
    """Extract ordered class names from CSV split"""
    df = pd.read_csv(csv_path)
    class_names = df.groupby('label')['emotion_name'].first().sort_index().tolist()
    return class_names


def get_dataloaders(train_speech_path, train_text_path,
                   val_speech_path, val_text_path,
                   batch_size=32):
    """Create train and validation dataloaders"""
    train_set = MultimodalFeatureDataset(train_speech_path, train_text_path)
    val_set = MultimodalFeatureDataset(val_speech_path, val_text_path)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    return train_loader, val_loader, train_set, val_set
