# Multimodal Emotion Recognition - Fusion Pipeline

This project implements and compares 4 different fusion strategies for multimodal emotion recognition using speech and text inputs.

## Fusion Variants

1. **Cross-Modal Attention Fusion** - Bi-directional attention between modalities
2. **Concatenation Fusion** - Simple concatenation baseline
3. **Gated Fusion** - Learned modality gating
4. **Weighted Average Fusion** - Attention-based modality weighting

## Project Structure

```
fusion_pipeline/
├── models.py           # Model architectures (Temporal, Contextual, Fusion, Classifier)
├── dataset.py          # Data loading utilities
├── utils.py            # Training/evaluation utilities and visualization
├── train.py            # Training script for all fusion variants
├── test.py             # Testing script with separability & failure analysis
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train all 4 fusion variants:

```bash
python train.py \
    --train_speech_features path/to/train_speech_features.pt \
    --train_text_features path/to/train_text_features.pt \
    --val_speech_features path/to/val_speech_features.pt \
    --val_text_features path/to/val_text_features.pt \
    --csv_path path/to/train_split.csv \
    --epochs 30 \
    --batch_size 32 \
    --model_dir models/fusion_pipeline \
    --results_dir results
```

**Outputs:**
- `models/fusion_pipeline/best_{variant}_model.pt` - Trained models
- `results/plots/{variant}_training_curves.png` - Training curves
- `results/plots/{variant}_confusion_matrix.png` - Confusion matrices
- `results/plots/fusion_variants_comparison.png` - Comparison plot
- `results/fusion_variants_comparison.txt` - Detailed comparison report

### Testing

Test all variants and perform comprehensive analysis:

```bash
python test.py \
    --test_speech_features path/to/test_speech_features.pt \
    --test_text_features path/to/test_text_features.pt \
    --csv_path path/to/train_split.csv \
    --test_csv_path path/to/test_split.csv \
    --model_dir models/fusion_pipeline \
    --results_dir results \
    --batch_size 32
```

**Outputs:**
- `results/test_results/test_metrics_comparison.txt` - Test performance metrics
- `results/test_results/{variant}_test_confusion_matrix.png` - Test confusion matrices
- `results/test_results/failure_cases_report.txt` - Detailed failure case analysis
- `results/separability/{variant}_temporal_tsne.png` - Temporal features separability
- `results/separability/{variant}_contextual_tsne.png` - Contextual features separability
- `results/separability/{variant}_fusion_tsne.png` - Fused features separability (CRITICAL)
- `results/separability/fusion_comparison_all.png` - Cross-variant comparison

## Key Features

### Training Script (`train.py`)
- ✅ Trains all 4 fusion variants automatically
- ✅ Best model checkpointing based on validation accuracy
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Individual training curves and confusion matrices
- ✅ Cross-variant comparison visualization
- ✅ Detailed performance report

### Testing Script (`test.py`)
- ✅ Comprehensive test set evaluation
- ✅ **Separability Visualization** (t-SNE, UMAP, PCA)
  - Temporal modeling block (speech)
  - Contextual modeling block (text)
  - Fusion block (multimodal) ← **Most Important**
- ✅ **Failure Case Analysis**
  - Top 5 error patterns per variant
  - 3-5 specific failure cases with filenames
  - Cross-variant error comparison
- ✅ Per-class metrics (Precision, Recall, F1)
- ✅ Test confusion matrices

## Model Configuration

Default configuration (can be modified in code):
```python
config = {
    'speech_dim': 768,      # Speech feature dimension
    'text_dim': 768,        # Text feature dimension
    'hidden': 256,          # Transformer hidden dimension
    'layers': 2,            # Transformer layers
    'f_hidden': 512,        # Fusion hidden dimension
    'classes': 7            # Number of emotion classes
}
```

## Expected Results Structure

After running both scripts:

```
results/
├── plots/
│   ├── cross_attention_training_curves.png
│   ├── cross_attention_confusion_matrix.png
│   ├── concatenation_training_curves.png
│   ├── concatenation_confusion_matrix.png
│   ├── gated_training_curves.png
│   ├── gated_confusion_matrix.png
│   ├── weighted_training_curves.png
│   ├── weighted_confusion_matrix.png
│   └── fusion_variants_comparison.png
├── test_results/
│   ├── cross_attention_test_confusion_matrix.png
│   ├── concatenation_test_confusion_matrix.png
│   ├── gated_test_confusion_matrix.png
│   ├── weighted_test_confusion_matrix.png
│   ├── test_metrics_comparison.txt
│   └── failure_cases_report.txt
├── separability/
│   ├── cross_attention_temporal_tsne.png
│   ├── cross_attention_contextual_tsne.png
│   ├── cross_attention_fusion_tsne.png
│   ├── (similar for other variants...)
│   └── fusion_comparison_all.png
└── fusion_variants_comparison.txt

models/fusion_pipeline/
├── best_cross_attention_model.pt
├── best_concatenation_model.pt
├── best_gated_model.pt
└── best_weighted_model.pt
```

## Analysis Sections Covered

This implementation satisfies all assignment requirements:

### A. Architecture Decisions ✅
- 4 different fusion architectures implemented
- Each with clear documentation of advantages/disadvantages
- Justification through empirical comparison

### B. Experiments ✅
- Speech-only: Temporal modeling block
- Text-only: Contextual modeling block
- Multimodal: All 4 fusion variants
- Comprehensive comparison metrics

### C. Analysis ✅
1. **Easiest/Hardest Emotions**: Per-class accuracy in reports
2. **When Fusion Helps Most**: Comparison between variants
3. **Error Analysis**: 3-5 failure cases per variant with filenames
4. **Separability Visualization**:
   - Temporal modeling block
   - Contextual modeling block
   - Fusion block (t-SNE, UMAP, PCA)

## Tips for Best Results

1. **Training**: Run for at least 30 epochs to see full convergence
2. **Separability**: t-SNE visualizations are most informative for fusion block
3. **Failure Analysis**: Review the failure_cases_report.txt for insights
4. **Model Selection**: Check both validation AND test performance

## Citation

If you use this code, please ensure proper attribution in your report.
