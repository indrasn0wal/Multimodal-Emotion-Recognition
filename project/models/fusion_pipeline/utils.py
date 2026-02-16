"""
Multimodal Emotion Recognition - Utilities
Visualization, metrics, and helper functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(history, fusion_type, save_path):
    """Plot training/validation loss and accuracy"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{fusion_type.upper()} Fusion Training', fontsize=16, fontweight='bold')

    # Plot 1: Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Learning Rate
    axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14)
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, fusion_type, save_path):
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
    plt.title(f'Confusion Matrix - {fusion_type.upper()} Fusion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_comparison(all_histories, save_path):
    """Compare all fusion variants"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Fusion Variants Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for (fusion_type, history), color in zip(all_histories.items(), colors):
        epochs = range(1, len(history['val_loss']) + 1)
        
        # Plot validation loss
        axes[0].plot(epochs, history['val_loss'], color=color, label=fusion_type.capitalize(), linewidth=2)
        
        # Plot validation accuracy
        axes[1].plot(epochs, history['val_acc'], color=color, label=fusion_type.capitalize(), linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Validation Loss Comparison', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy Comparison', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def save_comparison_report(all_results, class_names, save_path):
    """Generate detailed comparison report"""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTIMODAL FUSION VARIANTS COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall comparison table
        f.write("1. OVERALL PERFORMANCE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Fusion Type':<25} {'Best Val Acc':<15} {'Final Val Acc':<15} {'Rank':<10}\n")
        f.write("-"*80 + "\n")
        
        # Sort by best validation accuracy
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_acc'], reverse=True)
        
        for rank, (fusion_type, results) in enumerate(sorted_results, 1):
            best_acc = results['best_acc']
            final_acc = results['history']['val_acc'][-1]
            f.write(f"{fusion_type.capitalize():<25} {best_acc:<15.4f} {final_acc:<15.4f} {rank:<10}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed analysis for each variant
        f.write("2. DETAILED ANALYSIS BY VARIANT\n")
        f.write("="*80 + "\n\n")
        
        for fusion_type, results in all_results.items():
            f.write(f"\n{fusion_type.upper()} FUSION\n")
            f.write("-"*80 + "\n")
            
            history = results['history']
            cm = results['confusion_matrix']
            
            f.write(f"Best Validation Accuracy: {results['best_acc']:.4f}\n")
            f.write(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}\n")
            f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}\n\n")
            
            # Per-class accuracy
            f.write("Per-Class Accuracy:\n")
            f.write(f"{'Emotion':<20} {'Accuracy':<10} {'Correct':<10} {'Total':<10}\n")
            f.write("-"*50 + "\n")
            
            for i, emotion in enumerate(class_names):
                correct = cm[i, i]
                total = cm[i, :].sum()
                accuracy = correct / total if total > 0 else 0
                f.write(f"{emotion:<20} {accuracy:<10.4f} {correct:<10} {total:<10}\n")
            
            f.write("\n")
        
        # Key insights
        f.write("\n" + "="*80 + "\n")
        f.write("3. KEY INSIGHTS\n")
        f.write("="*80 + "\n\n")
        
        best_fusion = sorted_results[0][0]
        worst_fusion = sorted_results[-1][0]
        
        f.write(f"- Best performing fusion: {best_fusion.upper()} "
                f"({sorted_results[0][1]['best_acc']:.4f})\n")
        f.write(f"- Worst performing fusion: {worst_fusion.upper()} "
                f"({sorted_results[-1][1]['best_acc']:.4f})\n")
        f.write(f"- Performance gap: {sorted_results[0][1]['best_acc'] - sorted_results[-1][1]['best_acc']:.4f}\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-"*80 + "\n")
        if best_fusion == 'cross_attention':
            f.write("✓ Cross-modal attention provides the best performance by allowing\n")
            f.write("  each modality to focus on relevant information from the other.\n")
        elif best_fusion == 'gated':
            f.write("✓ Gated fusion provides the best performance by learning to weight\n")
            f.write("  modalities based on their informativeness for each sample.\n")
        elif best_fusion == 'weighted':
            f.write("✓ Weighted average provides good performance with simpler architecture.\n")
        else:
            f.write("✓ Simple concatenation is surprisingly effective despite its simplicity.\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Comparison report saved to {save_path}")


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss, total_correct = 0, 0
    
    for s_x, s_m, t_x, t_m, y in train_loader:
        s_x, s_m = s_x.to(device), s_m.to(device)
        t_x, t_m = t_x.to(device), t_m.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, _, _, _ = model(s_x, t_x, s_m, t_m)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(1) == y).sum().item()
    
    return total_loss / len(train_loader), total_correct


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss, total_correct = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for s_x, s_m, t_x, t_m, y in val_loader:
            s_x, s_m = s_x.to(device), s_m.to(device)
            t_x, t_m = t_x.to(device), t_m.to(device)
            y = y.to(device)
            
            logits, _, _, _ = model(s_x, t_x, s_m, t_m)
            
            total_loss += criterion(logits, y).item()
            preds = logits.argmax(1)
            total_correct += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return total_loss / len(val_loader), total_correct, all_preds, all_labels
