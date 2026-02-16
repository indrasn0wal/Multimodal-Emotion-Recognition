#!/bin/bash
# Multimodal Emotion Recognition - Complete Pipeline
# This script runs training and testing for all fusion variants

# Configuration - UPDATE THESE PATHS
TRAIN_SPEECH="path/to/train_speech_features.pt"
TRAIN_TEXT="path/to/train_text_features.pt"
VAL_SPEECH="path/to/val_speech_features.pt"
VAL_TEXT="path/to/val_text_features.pt"
TEST_SPEECH="path/to/test_speech_features.pt"
TEST_TEXT="path/to/test_text_features.pt"
TRAIN_CSV="path/to/train_split.csv"
TEST_CSV="path/to/test_split.csv"

# Training parameters
EPOCHS=30
BATCH_SIZE=32
MODEL_DIR="models/fusion_pipeline"
RESULTS_DIR="results"

echo "========================================="
echo "Multimodal Emotion Recognition Pipeline"
echo "========================================="
echo ""

# Step 1: Training
echo "Step 1: Training all fusion variants..."
echo "========================================="
python train.py \
    --train_speech_features "$TRAIN_SPEECH" \
    --train_text_features "$TRAIN_TEXT" \
    --val_speech_features "$VAL_SPEECH" \
    --val_text_features "$VAL_TEXT" \
    --csv_path "$TRAIN_CSV" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --model_dir "$MODEL_DIR" \
    --results_dir "$RESULTS_DIR"

if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

echo ""
echo "Training complete!"
echo ""

# Step 2: Testing
echo "Step 2: Testing and comprehensive analysis..."
echo "========================================="
python test.py \
    --test_speech_features "$TEST_SPEECH" \
    --test_text_features "$TEST_TEXT" \
    --csv_path "$TRAIN_CSV" \
    --test_csv_path "$TEST_CSV" \
    --model_dir "$MODEL_DIR" \
    --results_dir "$RESULTS_DIR" \
    --batch_size $BATCH_SIZE

if [ $? -ne 0 ]; then
    echo "Testing failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Models saved to: $MODEL_DIR"
echo ""
echo "Check the following files:"
echo "  - $RESULTS_DIR/fusion_variants_comparison.txt"
echo "  - $RESULTS_DIR/test_results/test_metrics_comparison.txt"
echo "  - $RESULTS_DIR/test_results/failure_cases_report.txt"
echo "  - $RESULTS_DIR/separability/ (visualization folder)"
echo ""
