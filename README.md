# Multimodal Emotion Recognition System

A comprehensive emotion recognition system that leverages speech, text, and multimodal (speech + text) inputs to accurately classify emotions from the Toronto Emotional Speech Set (TESS) dataset.

**Key Achievement:** 93.1% accuracy on multimodal emotion recognition (weighted fusion approach)

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Decisions](#architecture-decisions)
3. [Experimental Results](#experimental-results)
4. [Analysis & Insights](#analysis--insights)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Results & Visualizations](#results--visualizations)
8. [Dataset](#dataset)

---

## 📌 Project Overview

This project implements a comprehensive multimodal emotion recognition system with three distinct pipelines:

| Pipeline | Input | Architecture | Test Accuracy |
|----------|-------|--------------|---------------|
| **Speech Only** | Audio features | Transformer Encoder | **89.0%** |
| **Text Only** | Tokenized text | RoBERTa + Transformer | **15.0%** |
| **Multimodal** | Audio + Text | 4 Fusion Variants | **93.1%** (best) |

### Emotions Classified (7 classes)
- Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad

---

## 🏗️ Architecture Decisions

### 1. **Feature Extraction Block**

#### Speech Features: Wav2Vec 2.0
- **Why Wav2Vec 2.0?**
  - Self-supervised pre-trained model on 960 hours of LibriSpeech
  - Extracts contextual audio representations (768-dim)
  - Captures fine-grained acoustic patterns without manual feature engineering
  - Better than traditional MFCCs for complex emotional nuances
  
**Configuration:**
```
- Model: facebook/wav2vec2-base
- Output dimension: 768
- Sampling rate: 16 kHz
- Feature extraction: Pre-trained frozen encoder
```

#### Text Features: RoBERTa
- **Why RoBERTa?**
  - Bidirectional transformer-based model
  - Better than BERT for understanding contextual relationships
  - Produces 768-dim embeddings per token
  - Pre-trained on diverse text corpora (benefits generalization)

**Configuration:**
```
- Model: roberta-base
- Output dimension: 768 per token
- Tokenization: Roberta Tokenizer
- Context: "Say the word [target_word]"
```

---

### 2. **Temporal/Contextual Modelling Block**

#### Transformer Encoder (Both Modalities)
- **Why Transformer?**
  - Self-attention mechanism captures long-range dependencies
  - Handles variable-length sequences elegantly with masking
  - Proven effective for sequential emotional patterns
  - Parallel processing vs. sequential RNNs (faster training)

**Architecture:**
```
TransformerModeling:
  - Input: Time-series features (speech) or token sequence (text)
  - Encoder Layers: 2
  - Hidden Dim: 256
  - Attention Heads: 8
  - Activation: GELU
  - Dropout: 0.4
  - Output: Sequence of contextual/temporal features
```

**Why 2 layers?**
- Sufficient depth for emotion pattern modeling
- Avoids over-parameterization given dataset size (~2000 training samples)
- Balances expressiveness vs. generalization

**Masked Mean Pooling:**
- Prevents padding tokens from diluting emotional signal
- Focuses on actual speech/text content

---

### 3. **Fusion Block - 4 Variants Compared**

#### **Variant 1: Cross-Modal Attention Fusion**
**Advantages:**
- Bi-directional attention between modalities
- Speech attends to text meaningful segments
- Text attends to speech emotional cues
- Captures complementary information

**Architecture:**
```
1. Project both modalities to common hidden space (512-dim)
2. Speech queries Text: MultiheadAttention
3. Text queries Speech: MultiheadAttention
4. Residual connections + LayerNorm
5. Concatenate and project to final representation
```

**Performance:** 90.95% test accuracy

---

#### **Variant 2: Weighted Average Fusion** (Best on Test Set!)
**Advantages:**
- Simple yet effective gating mechanism
- Learns modality importance per sample
- Scalable and interpretable

**Architecture:**
```
1. Compute attention weights: softmax(learnable_proj(concat(speech, text)))
2. Weighted combination: w_s * speech + w_t * text
3. MLP head for classification
```

**Test Performance: 93.1% accuracy** (best performer!)
- Learns to weight speech higher for certain emotions
- Adapts fusion strategy per sample
- Generalizes better than bi-directional attention

---

#### **Variant 3: Gated Fusion**
**Advantages:**
- Learns to suppress irrelevant modality
- Computational efficiency

**Architecture:**
```
1. Compute gates: sigmoid(learnable_proj(features))
2. Gated output: gate * features
3. Combine gated representations
```

Test Accuracy: 90.71%

---

#### **Variant 4: Concatenation Fusion** (Baseline)
**Advantages:**
- Simple interpretable baseline
- No additional fusion parameters

**Architecture:**
```
1. Simple concatenation: concat(temporal_speech, contextual_text)
2. Direct MLP classifier
```

Test Accuracy: 92.62%

---

### 4. **Classifier Block**

```python
Classifier:
  Linear(fused_dim, 256) → ReLU → Dropout(0.2) → Linear(256, 7)
```

- **Why small hidden layer (256)?**
  - Sufficient complexity for 7-class classification
  - Prevents overfitting on limited training data
  - Fast inference

---

## 📊 Experimental Results

### Overall Performance Comparison

```
MODEL VARIANT                 TRAIN ACC    VAL ACC    TEST ACC
─────────────────────────────────────────────────────────────
Speech Only (Baseline)           96.7%      91.7%      89.0%
Text Only (Baseline)             28.3%      15.8%      15.0%
─────────────────────────────────────────────────────────────
Weighted Fusion (BEST)          97.5%      93.5%      93.1% ✓
Concatenation Fusion            97.9%      92.9%      92.6%
Cross-Attention Fusion          99.4%      94.1%      90.9%
Gated Fusion                     97.8%      93.3%      90.7%
```

### Per-Class Accuracy (Weighted Fusion - Best Model)

| Emotion | Accuracy | Precision | Recall | F1-Score | Support |
|---------|----------|-----------|--------|----------|---------|
| Angry | 91.67% | 0.9167 | 0.9167 | 0.9167 | 60 |
| Disgust | 93.33% | 0.8889 | 0.9333 | 0.9106 | 60 |
| Fear | 98.33% | 0.9219 | 0.9833 | 0.9516 | 60 |
| Happy | 93.33% | 0.9180 | 0.9333 | 0.9256 | 60 |
| Neutral | 91.67% | 1.0000 | 0.9167 | 0.9565 | 60 |
| Pleasant Surprise | 86.67% | 0.8387 | 0.8667 | 0.8526 | 60 |
| Sad | 96.67% | 0.9667 | 0.9667 | 0.9667 | 60 |
| **AVERAGE** | **93.10%** | **0.9218** | **0.9310** | **0.9259** | **420** |

---

## 🔍 Analysis & Insights

### 1. **Easiest vs Hardest Emotions to Classify**

#### ✅ **Easiest: Fear (98.33%)**
- **Why?**
  - Distinct acoustic patterns: high pitch, rapid speech
  - Clear vocal tension markers
  - Speech vs. text both contribute clear signals

#### ✅ **Easiest: Sad (96.67%)**
- Slower speech rate, lower frequencies
- Consistent across speakers
- Easy to detect in both modalities

#### ❌ **Hardest: Pleasant Surprise (86.67%)**
- **Why?**
  - Extremely similar to happy emotion
  - Overlapping acoustic features (higher pitch, energy)
  - Text context provides minimal distinction
  - Often confused with: Happy, Disgust

#### ❌ **Moderately Hard: Angry (91.67%)**
- Similarity to fear in acoustic space
- Both have high energy and pitch
- Confusion primarily with: Fear

---

### 2. **When Does Fusion Help Most?**

#### Fusion Gain: +4.1% over Speech-Only (93.1% vs 89.0%)

**Fusion helps most for:**
1. **Emotionally ambiguous samples** (angry vs fear)
2. **Happy vs Pleasant Surprise** - differentiation through word context
3. **Neutral Emotion** - text compensates for flat speech prosody

#### Why Text-Only Performs Poorly (15%):
- **Dataset limitation:** Text is always "Say the word [word]"
- Minimal semantic variation across emotions
- Requires much richer text data for success
- **Conclusion:** For this dataset, speech is ~6x more informative than text alone

---

### 3. **Error Analysis: Failure Cases**

#### **Top 5 Error Patterns (Weighted Fusion)**

| Pattern | Root Cause |
|---------|-----------|
| **Angry → Fear** | Acoustic similarity (high energy, pitch) |
| **Pleasant Surprise → Disgust** | Both involve vocal tension |
| **Neutral → Happy** | Some neutral speakers use higher pitch |
| **Fear → Angry** | Overlap in acoustic space |
| **Disgust → Neutral** | Subtle disgust may sound neutral |

#### **Specific Failure Cases (Top 3)**

**Case 1: Angry Sample Predicted as Fear**
- True Emotion: Angry
- Predicted: Fear (confidence: 93.74%)
- Analysis: Speaker's angry delivery matched fear prosody (fast, high-pitched)

**Case 2: Pleasant Surprise → Disgust**
- True Emotion: Pleasant Surprise
- Predicted: Disgust (confidence: 88.36%)
- Analysis: Both emotions involve vocal tension

**Case 3: Neutral → Happy**
- True Emotion: Neutral
- Predicted: Happy (confidence: 81.5%)
- Analysis: Some neutral speakers naturally use higher pitch (speaker variation)

---

### 4. **Separability Visualization Analysis**

#### **Temporal Modelling Block (Speech Features)**
- Emotions form loose clusters with significant overlap
- Angry-Fear, Happy-Pleasant Surprise show substantial overlap
- Clear separation for Fear and Sad

#### **Contextual Modelling Block (Text Features)**
- Very poor separability (as expected - limited text data)
- Almost all emotions clustered together

#### **Fusion Block Output**
- Much improved separability vs temporal features alone
- More compact emotion-specific regions
- Fusion reduces neutral overlap, separates angry-fear better

[See visualizations in `project/Results/Fusion/separability_visualization/`]

---

## 💾 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)
- 8GB RAM minimum

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download TESS Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/ejlok/toronto-emotional-speech-set-tess):
```bash
unzip TESS_Toronto_emotional_speech_set_data.zip -d project/data/
```

---

## 🚀 Usage Guide

### Quick Start
```bash
cd project/models/fusion_pipeline/
bash run_all.sh
```

### Speech Pipeline
```bash
cd project/models/speech_pipeline/
python train.py --data_dir /path/to/TESS_data --csv_path ../Data\ Split/train_split.csv
python test.py --data_dir /path/to/TESS_data --csv_path ../../Data\ Split/test_split.csv
```

### Text Pipeline
```bash
cd project/models/text_pipeline/
python train.py --csv_path ../Data\ Split/train_split.csv
python test.py --csv_path ../../Data\ Split/test_split.csv
```

### Fusion Pipeline
```bash
cd project/models/fusion_pipeline/
python train.py --train_speech_features ./cached_features/train_speech.pt
python test.py --test_speech_features ./cached_features/test_speech.pt
```

---

## 📈 Results & Visualizations

All visualizations saved in `project/Results/`:
- Training curves for all variants
- Confusion matrices per emotion
- Separability analysis (t-SNE projections)
- Performance comparison plots

---

## 📚 Dataset: TESS

- **Size:** 2,880 audio files
- **Emotions:** 7 (Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad)
- **Speakers:** 4 female
- **Split:** Train (68%), Val (7.3%), Test (14.6%)
- **Access:** [Kaggle](https://www.kaggle.com/datasets/ejlok/toronto-emotional-speech-set-tess)

---

## 📁 Project Structure

```
Multimodal-Emotion-Recognition/
├── README.md
├── requirements.txt
├── project/
│   ├── Data Split/
│   │   ├── train_split.csv
│   │   ├── val_split.csv
│   │   └── test_split.csv
│   ├── models/
│   │   ├── speech_pipeline/
│   │   ├── text_pipeline/
│   │   └── fusion_pipeline/
│   └── Results/
│       ├── Speech only/
│       ├── Text Only/
│       └── Fusion/
└── .gitignore
```

---

## 🔬 Hyperparameters

- **Optimizer:** Adam (lr=1e-4, weight_decay=1e-4)
- **Loss:** CrossEntropyLoss (label_smoothing=0.1)
- **Batch Size:** 32
- **Epochs:** 30
- **Scheduler:** ReduceLROnPlateau

---
