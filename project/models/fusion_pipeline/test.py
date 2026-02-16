"""
Multimodal Emotion Recognition - Testing Script
Tests all fusion variants and performs comprehensive analysis including:
- Performance evaluation
- Separability visualization (t-SNE/UMAP)
- Failure case analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

from models import MultimodalEmotionRecognition
from dataset import MultimodalFeatureDataset, get_class_names
from utils import plot_confusion_matrix


def test_fusion_variant(fusion_type, test_loader, test_size, config, device, class_names, model_path):
    """
    Test a specific fusion variant on test set
    
    Returns:
        dict with test_acc, predictions, labels, confusion_matrix, fused_features
    """
    print(f"\n{'='*80}")
    print(f"Testing Fusion Variant: {fusion_type.upper()}")
    print(f"{'='*80}\n")
    
    # Load best model
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Initialize model with same architecture
    model = MultimodalEmotionRecognition(config, fusion_type=fusion_type).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test
    all_preds, all_labels = [], []
    all_fused_features = []
    all_temporal_features = []
    all_contextual_features = []
    
    with torch.no_grad():
        for s_x, s_m, t_x, t_m, y in test_loader:
            s_x, s_m = s_x.to(device), s_m.to(device)
            t_x, t_m = t_x.to(device), t_m.to(device)
            y = y.to(device)
            
            logits, s_feat, t_feat, fused = model(s_x, t_x, s_m, t_m)
            
            # Pool temporal features
            s_mask_expanded = s_m.unsqueeze(-1)
            s_pooled = (s_feat * s_mask_expanded).sum(1) / s_mask_expanded.sum(1)
            
            # Pool contextual features
            t_mask_expanded = t_m.unsqueeze(-1)
            t_pooled = (t_feat * t_mask_expanded).sum(1) / t_mask_expanded.sum(1)
            
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_fused_features.append(fused.cpu().numpy())
            all_temporal_features.append(s_pooled.cpu().numpy())
            all_contextual_features.append(t_pooled.cpu().numpy())
    
    # Calculate metrics
    test_acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Test Accuracy: {test_acc:.4f}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    return {
        'test_acc': test_acc,
        'predictions': all_preds,
        'labels': all_labels,
        'confusion_matrix': cm,
        'fused_features': np.vstack(all_fused_features),
        'temporal_features': np.vstack(all_temporal_features),
        'contextual_features': np.vstack(all_contextual_features)
    }


def visualize_separability(features, labels, class_names, method='tsne', title='', save_path=None):
    """Visualize feature separability using t-SNE, UMAP, or PCA"""
    print(f"Computing {method.upper()} projection for {title}...")
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embedded = reducer.fit_transform(features)
    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedded = reducer.fit_transform(features)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embedded = reducer.fit_transform(features)
    else:
        print(f"Warning: {method} not available, using PCA instead")
        reducer = PCA(n_components=2, random_state=42)
        embedded = reducer.fit_transform(features)
        method = 'pca'
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Color palette
    colors = sns.color_palette("husl", len(class_names))
    
    # Plot each emotion class
    for idx, emotion in enumerate(class_names):
        mask = labels == idx
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=[colors[idx]], label=emotion, 
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    plt.title(f'{title}\n{method.upper()} Visualization of Emotion Clusters', 
              fontsize=14, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.close()


def analyze_separability(all_test_results, class_names, results_dir):
    """Generate separability visualizations for all variants"""
    os.makedirs(f"{results_dir}/separability", exist_ok=True)
    
    print(f"\n{'='*80}")
    print("GENERATING SEPARABILITY VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    methods = ['tsne', 'pca']
    if UMAP_AVAILABLE:
        methods.append('umap')
    
    for fusion_type, results in all_test_results.items():
        labels = np.array(results['labels'])
        
        for method in methods:
            # Temporal features
            visualize_separability(
                results['temporal_features'], labels, class_names,
                method=method,
                title=f'{fusion_type.upper()} - Temporal Modeling Block (Speech)',
                save_path=f"{results_dir}/separability/{fusion_type}_temporal_{method}.png"
            )
            
            # Contextual features
            visualize_separability(
                results['contextual_features'], labels, class_names,
                method=method,
                title=f'{fusion_type.upper()} - Contextual Modeling Block (Text)',
                save_path=f"{results_dir}/separability/{fusion_type}_contextual_{method}.png"
            )
            
            # Fused features (MOST IMPORTANT)
            visualize_separability(
                results['fused_features'], labels, class_names,
                method=method,
                title=f'{fusion_type.upper()} - Fusion Block (Multimodal)',
                save_path=f"{results_dir}/separability/{fusion_type}_fusion_{method}.png"
            )
    
    # Comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('Fusion Block Separability Comparison (t-SNE)', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    colors = sns.color_palette("husl", len(class_names))
    
    for idx, (fusion_type, results) in enumerate(all_test_results.items()):
        labels = np.array(results['labels'])
        features = results['fused_features']
        
        print(f"Computing t-SNE for {fusion_type} comparison...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(features)
        
        for class_idx, emotion in enumerate(class_names):
            mask = labels == class_idx
            axes[idx].scatter(embedded[mask, 0], embedded[mask, 1],
                            c=[colors[class_idx]], label=emotion,
                            alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        
        axes[idx].set_title(f'{fusion_type.upper()} Fusion', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('t-SNE 1', fontsize=10)
        axes[idx].set_ylabel('t-SNE 2', fontsize=10)
        axes[idx].legend(loc='best', fontsize=8, framealpha=0.9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/separability/fusion_comparison_all.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison to {results_dir}/separability/fusion_comparison_all.png")
    plt.close()


def analyze_failure_cases(fusion_type, test_results, class_names, csv_path, results_dir, num_cases=5):
    """Identify and analyze failure cases"""
    print(f"\n{'='*80}")
    print(f"Failure Case Analysis for {fusion_type.upper()}")
    print(f"{'='*80}\n")
    
    all_preds = np.array(test_results['predictions'])
    all_labels = np.array(test_results['labels'])
    cm = test_results['confusion_matrix']
    
    # Find misclassifications
    misclassified_mask = all_preds != all_labels
    misclassified_indices = np.where(misclassified_mask)[0]
    
    print(f"Total misclassifications: {len(misclassified_indices)} / {len(all_labels)}")
    print(f"Error rate: {len(misclassified_indices) / len(all_labels) * 100:.2f}%\n")
    
    # Group errors by (true_label, predicted_label) pairs
    error_groups = defaultdict(list)
    for idx in misclassified_indices:
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        
        error_groups[(true_label, pred_label)].append(idx)
    
    # Sort error groups by frequency
    sorted_error_groups = sorted(error_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Load CSV for filenames if available
    try:
        test_df = pd.read_csv(csv_path)
        has_filenames = True
    except:
        has_filenames = False
        print("Warning: Could not load CSV file for filenames\n")
    
    # Collect failure cases
    failure_cases = []
    
    print("="*80)
    print("TOP ERROR PATTERNS")
    print("="*80 + "\n")
    
    for (true_label, pred_label), indices in sorted_error_groups[:5]:
        true_emotion = class_names[true_label]
        pred_emotion = class_names[pred_label]
        
        print(f"\nError Pattern: {true_emotion} → {pred_emotion} ({len(indices)} cases)")
        print("-" * 80)
        
        for i, idx in enumerate(indices[:num_cases]):
            if has_filenames and idx < len(test_df):
                filename = test_df.iloc[idx].get('file_path', f"sample_{idx}")
                text = test_df.iloc[idx].get('transcript', "N/A")
            else:
                filename = f"sample_{idx}"
                text = "N/A"
            
            print(f"\nCase {i+1}:")
            print(f"  File: {filename}")
            print(f"  Text: '{text}'")
            print(f"  True Label: {true_emotion}")
            print(f"  Predicted: {pred_emotion}")
            
            failure_cases.append({
                'filename': filename,
                'text': text,
                'true_emotion': true_emotion,
                'predicted_emotion': pred_emotion,
                'error_pattern': f"{true_emotion} → {pred_emotion}"
            })
    
    return failure_cases


def save_test_metrics(all_test_results, class_names, save_path):
    """Save comprehensive test metrics report"""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST SET PERFORMANCE - ALL FUSION VARIANTS\n")
        f.write("="*80 + "\n\n")
        
        # Overall comparison
        f.write("1. OVERALL TEST ACCURACY COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Fusion Type':<25} {'Test Accuracy':<15} {'Rank':<10}\n")
        f.write("-"*80 + "\n")
        
        sorted_results = sorted(all_test_results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
        
        for rank, (fusion_type, results) in enumerate(sorted_results, 1):
            test_acc = results['test_acc']
            f.write(f"{fusion_type.capitalize():<25} {test_acc:<15.4f} {rank:<10}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed per-class metrics
        f.write("2. PER-CLASS TEST ACCURACY BY VARIANT\n")
        f.write("="*80 + "\n\n")
        
        for fusion_type, results in all_test_results.items():
            f.write(f"\n{fusion_type.upper()} FUSION\n")
            f.write("-"*80 + "\n")
            f.write(f"Overall Test Accuracy: {results['test_acc']:.4f}\n\n")
            
            cm = results['confusion_matrix']
            f.write(f"{'Emotion':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
            f.write("-"*70 + "\n")
            
            for i, emotion in enumerate(class_names):
                correct = cm[i, i]
                total = cm[i, :].sum()
                accuracy = correct / total if total > 0 else 0
                
                pred_total = cm[:, i].sum()
                precision = correct / pred_total if pred_total > 0 else 0
                recall = accuracy
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"{emotion:<20} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("3. SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        best_fusion = sorted_results[0][0]
        best_acc = sorted_results[0][1]['test_acc']
        worst_fusion = sorted_results[-1][0]
        worst_acc = sorted_results[-1][1]['test_acc']
        
        f.write(f"Best Performing Fusion: {best_fusion.upper()} ({best_acc:.4f})\n")
        f.write(f"Worst Performing Fusion: {worst_fusion.upper()} ({worst_acc:.4f})\n")
        f.write(f"Performance Gap: {best_acc - worst_acc:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nTest metrics saved to {save_path}")


def save_failure_case_report(all_failure_cases, save_path):
    """Save detailed failure case report"""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FAILURE CASE ANALYSIS - ALL FUSION VARIANTS\n")
        f.write("="*80 + "\n\n")
        
        for fusion_type, cases in all_failure_cases.items():
            f.write(f"\n{fusion_type.upper()} FUSION - FAILURE CASES\n")
            f.write("-"*80 + "\n\n")
            
            for i, case in enumerate(cases, 1):
                f.write(f"Failure Case #{i}:\n")
                f.write(f"  Error Pattern: {case['error_pattern']}\n")
                f.write(f"  File: {case['filename']}\n")
                f.write(f"  Transcript: '{case['text']}'\n")
                f.write(f"  True Emotion: {case['true_emotion']}\n")
                f.write(f"  Predicted: {case['predicted_emotion']}\n\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Failure case report saved to {save_path}")


def run_all_tests(args):
    """Run comprehensive testing on all fusion variants"""
    
    # Setup directories
    os.makedirs(f"{args.results_dir}/test_results", exist_ok=True)
    
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
    
    # Load test dataset
    test_set = MultimodalFeatureDataset(
        args.test_speech_features,
        args.test_text_features
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    print(f"Test samples: {len(test_set)}")
    print(f"Number of classes: {len(class_names)}\n")
    
    # Test all variants
    fusion_types = ['cross_attention', 'concatenation', 'gated', 'weighted']
    all_test_results = {}
    all_failure_cases = {}
    
    for fusion_type in fusion_types:
        model_path = f"{args.model_dir}/best_{fusion_type}_model.pt"
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found for {fusion_type}, skipping...")
            continue
        
        # Test the model
        test_results = test_fusion_variant(
            fusion_type=fusion_type,
            test_loader=test_loader,
            test_size=len(test_set),
            config=config,
            device=device,
            class_names=class_names,
            model_path=model_path
        )
        
        all_test_results[fusion_type] = test_results
        
        # Plot confusion matrix
        plot_confusion_matrix(
            test_results['confusion_matrix'],
            class_names,
            fusion_type,
            save_path=f"{args.results_dir}/test_results/{fusion_type}_test_confusion_matrix.png"
        )
        
        # Analyze failure cases
        failure_cases = analyze_failure_cases(
            fusion_type=fusion_type,
            test_results=test_results,
            class_names=class_names,
            csv_path=args.test_csv_path,
            results_dir=args.results_dir,
            num_cases=3
        )
        all_failure_cases[fusion_type] = failure_cases
    
    # Generate comprehensive analysis
    if all_test_results:
        # Separability analysis
        analyze_separability(all_test_results, class_names, args.results_dir)
        
        # Save test metrics
        save_test_metrics(
            all_test_results,
            class_names,
            save_path=f"{args.results_dir}/test_results/test_metrics_comparison.txt"
        )
        
        # Save failure case report
        save_failure_case_report(
            all_failure_cases,
            save_path=f"{args.results_dir}/test_results/failure_cases_report.txt"
        )
        
        # Print summary
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        sorted_results = sorted(all_test_results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
        for rank, (fusion_type, results) in enumerate(sorted_results, 1):
            print(f"{rank}. {fusion_type.upper():<20} - Test Acc: {results['test_acc']:.4f}")
        print("="*80 + "\n")
        
        print(f"All test results saved to: {args.results_dir}/test_results")
        print(f"Separability visualizations saved to: {args.results_dir}/separability")


def main():
    parser = argparse.ArgumentParser(description='Test Multimodal Emotion Recognition - All Fusion Variants')
    
    # Data paths
    parser.add_argument('--test_speech_features', type=str, required=True,
                       help='Path to test speech features (.pt file)')
    parser.add_argument('--test_text_features', type=str, required=True,
                       help='Path to test text features (.pt file)')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to train CSV file for class names')
    parser.add_argument('--test_csv_path', type=str, required=True,
                       help='Path to test CSV file for filenames and transcripts')
    
    # Model and results paths
    parser.add_argument('--model_dir', type=str, default='models/fusion_pipeline',
                       help='Directory containing trained models (default: models/fusion_pipeline)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save test results (default: results)')
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    # Run tests
    run_all_tests(args)


if __name__ == "__main__":
    main()
