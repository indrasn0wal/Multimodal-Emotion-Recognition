"""
Multimodal Emotion Recognition - Training Script
Trains all 4 fusion variants and generates comparison reports
"""

import torch
import torch.nn as nn
import argparse
import os
from sklearn.metrics import confusion_matrix

from models import MultimodalEmotionRecognition
from dataset import get_dataloaders, get_class_names
from utils import (train_one_epoch, evaluate, plot_training_history,
                  plot_confusion_matrix, plot_comparison, save_comparison_report)


def train_fusion_variant(fusion_type, train_loader, val_loader, train_size, val_size,
                        config, device, class_names, epochs=30, save_dir='models/fusion_pipeline'):
    """
    Train a specific fusion variant
    
    Args:
        fusion_type: 'cross_attention', 'concatenation', 'gated', or 'weighted'
        train_loader: training data loader
        val_loader: validation data loader
        train_size: number of training samples
        val_size: number of validation samples
        config: model configuration dict
        device: torch device
        class_names: list of emotion class names
        epochs: number of training epochs
        save_dir: directory to save models
    
    Returns:
        history: training history dict
        best_val_acc: best validation accuracy achieved
    """
    print(f"\n{'='*80}")
    print(f"Training Fusion Variant: {fusion_type.upper()}")
    print(f"{'='*80}\n")
    
    # Initialize model
    model = MultimodalEmotionRecognition(config, fusion_type=fusion_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_acc = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Train
        train_loss, train_correct = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss, val_correct, all_preds, all_labels = evaluate(model, val_loader, criterion, device)
        
        # Calculate metrics
        train_acc = train_correct / train_size
        val_acc = val_correct / val_size
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            print(f"  → Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving best model...")
            best_val_acc = val_acc
            
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'confusion_matrix': confusion_matrix(all_labels, all_preds),
                'class_names': class_names,
                'fusion_type': fusion_type
            }, f"{save_dir}/best_{fusion_type}_model.pt")
        
        # Step scheduler
        scheduler.step(val_acc)
    
    print(f"\n{fusion_type.upper()} - Best Validation Accuracy: {best_val_acc:.4f}\n")
    
    return history, best_val_acc


def run_all_fusion_experiments(args):
    """Train and compare all fusion variants"""
    
    # Setup directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/plots", exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load class names
    class_names = get_class_names(args.csv_path)
    config = {
        'speech_dim': 768,
        'text_dim': 768,
        'hidden': 256,
        'layers': 2,
        'f_hidden': 512,
        'classes': len(class_names)
    }
    
    # Load datasets
    train_loader, val_loader, train_set, val_set = get_dataloaders(
        args.train_speech_features,
        args.train_text_features,
        args.val_speech_features,
        args.val_text_features,
        batch_size=args.batch_size
    )
    
    print(f"Train samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}\n")
    
    # Fusion variants to test
    fusion_types = ['cross_attention', 'concatenation', 'gated', 'weighted']
    
    # Store results
    all_results = {}
    all_histories = {}
    
    # Train each variant
    for fusion_type in fusion_types:
        history, best_acc = train_fusion_variant(
            fusion_type=fusion_type,
            train_loader=train_loader,
            val_loader=val_loader,
            train_size=len(train_set),
            val_size=len(val_set),
            config=config,
            device=device,
            class_names=class_names,
            epochs=args.epochs,
            save_dir=args.model_dir
        )
        
        # Load best model for confusion matrix
        checkpoint = torch.load(f"{args.model_dir}/best_{fusion_type}_model.pt", weights_only=False)
        cm = checkpoint['confusion_matrix']
        
        # Store results
        all_results[fusion_type] = {
            'history': history,
            'best_acc': best_acc,
            'confusion_matrix': cm
        }
        all_histories[fusion_type] = history
        
        # Plot individual results
        plot_training_history(
            history,
            fusion_type,
            save_path=f"{args.results_dir}/plots/{fusion_type}_training_curves.png"
        )
        plot_confusion_matrix(
            cm,
            class_names,
            fusion_type,
            save_path=f"{args.results_dir}/plots/{fusion_type}_confusion_matrix.png"
        )
    
    # Generate comparison visualizations
    plot_comparison(all_histories, save_path=f"{args.results_dir}/plots/fusion_variants_comparison.png")
    
    # Generate comparison report
    save_comparison_report(all_results, class_names, save_path=f"{args.results_dir}/fusion_variants_comparison.txt")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL FUSION VARIANTS")
    print("="*80)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_acc'], reverse=True)
    for rank, (fusion_type, results) in enumerate(sorted_results, 1):
        print(f"{rank}. {fusion_type.upper():<20} - Best Val Acc: {results['best_acc']:.4f}")
    print("="*80 + "\n")
    
    print(f"\nAll results saved to: {args.results_dir}")
    print(f"All models saved to: {args.model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Emotion Recognition - All Fusion Variants')
    
    # Data paths
    parser.add_argument('--train_speech_features', type=str, required=True,
                       help='Path to training speech features (.pt file)')
    parser.add_argument('--train_text_features', type=str, required=True,
                       help='Path to training text features (.pt file)')
    parser.add_argument('--val_speech_features', type=str, required=True,
                       help='Path to validation speech features (.pt file)')
    parser.add_argument('--val_text_features', type=str, required=True,
                       help='Path to validation text features (.pt file)')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to train CSV file for class names')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    
    # Output directories
    parser.add_argument('--model_dir', type=str, default='models/fusion_pipeline',
                       help='Directory to save models (default: models/fusion_pipeline)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    # Run experiments
    run_all_fusion_experiments(args)


if __name__ == "__main__":
    main()
