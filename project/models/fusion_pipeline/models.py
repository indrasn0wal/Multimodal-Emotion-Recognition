"""
Multimodal Emotion Recognition - Model Architectures
Contains: Temporal/Contextual Modeling, Fusion Variants, Classifier
"""

import torch
import torch.nn as nn


class TransformerModeling(nn.Module):
    """
    Standard Transformer Encoder for both Temporal (Speech)
    and Contextual (Text) sequence modeling.
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, dropout=0.4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = input_dim

    def forward(self, x, mask=None):
        # mask: 1 for valid, 0 for padding. Transformer needs inverted mask (True to ignore)
        attn_mask = ~mask.bool() if mask is not None else None
        output = self.transformer(x, src_key_padding_mask=attn_mask)
        return self.dropout(output)


class CrossModalAttentionFusion(nn.Module):
    """
    Bi-Directional Cross-Attention Fusion.
    Speech attends to Text AND Text attends to Speech.
    
    Advantages:
    - Captures complementary information between modalities
    - Allows each modality to focus on relevant parts of the other
    - State-of-the-art performance in multimodal tasks
    """
    def __init__(self, speech_dim=768, text_dim=768, hidden_dim=512, num_heads=8, dropout=0.4):
        super().__init__()
        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Cross-modal attention layers
        self.s_to_t_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.t_to_s_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.ln_s = nn.LayerNorm(hidden_dim)
        self.ln_t = nn.LayerNorm(hidden_dim)

        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        self.output_dim = hidden_dim

    def forward(self, speech_feats, text_feats, speech_mask=None, text_mask=None):
        s_p = self.speech_proj(speech_feats)
        t_p = self.text_proj(text_feats)

        s_mask = ~speech_mask.bool() if speech_mask is not None else None
        t_mask = ~text_mask.bool() if text_mask is not None else None

        # Cross-Modal Interaction
        s_attended, _ = self.s_to_t_attn(query=s_p, key=t_p, value=t_p, key_padding_mask=t_mask)
        t_attended, _ = self.t_to_s_attn(query=t_p, key=s_p, value=s_p, key_padding_mask=s_mask)

        # Residual and Pooling
        speech_mask_expanded = speech_mask.unsqueeze(-1) if speech_mask is not None else torch.ones_like(s_p[:, :, :1])
        text_mask_expanded = text_mask.unsqueeze(-1) if text_mask is not None else torch.ones_like(t_p[:, :, :1])
        
        s_vec = (self.ln_s(s_attended + s_p) * speech_mask_expanded).sum(1) / speech_mask_expanded.sum(1)
        t_vec = (self.ln_t(t_attended + t_p) * text_mask_expanded).sum(1) / text_mask_expanded.sum(1)

        return self.fusion_proj(torch.cat([s_vec, t_vec], dim=-1))


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation of speech and text representations.
    
    Advantages:
    - Simple and interpretable baseline
    - Low computational cost
    - No additional parameters for fusion
    
    Disadvantages:
    - No interaction between modalities
    - Treats both modalities independently
    """
    def __init__(self, speech_dim=768, text_dim=768, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.speech_proj = nn.Linear(speech_dim, hidden_dim // 2)
        self.text_proj = nn.Linear(text_dim, hidden_dim // 2)
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        self.output_dim = hidden_dim

    def forward(self, speech_feats, text_feats, speech_mask=None, text_mask=None):
        # Average pooling with mask
        speech_mask_expanded = speech_mask.unsqueeze(-1) if speech_mask is not None else torch.ones_like(speech_feats[:, :, :1])
        text_mask_expanded = text_mask.unsqueeze(-1) if text_mask is not None else torch.ones_like(text_feats[:, :, :1])
        
        s_pooled = (speech_feats * speech_mask_expanded).sum(1) / speech_mask_expanded.sum(1)
        t_pooled = (text_feats * text_mask_expanded).sum(1) / text_mask_expanded.sum(1)
        
        # Project and concatenate
        s_proj = self.speech_proj(s_pooled)
        t_proj = self.text_proj(t_pooled)
        
        concatenated = torch.cat([s_proj, t_proj], dim=-1)
        return self.fusion_proj(concatenated)


class GatedFusion(nn.Module):
    """
    Gated fusion with learned modality importance.
    
    Advantages:
    - Learns to weight modalities based on their informativeness
    - Can suppress noisy modalities
    - Adaptive to different emotion types
    
    Mechanism:
    - Uses sigmoid gates to control information flow from each modality
    """
    def __init__(self, speech_dim=768, text_dim=768, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Gate networks
        self.speech_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        self.output_dim = hidden_dim

    def forward(self, speech_feats, text_feats, speech_mask=None, text_mask=None):
        # Average pooling with mask
        speech_mask_expanded = speech_mask.unsqueeze(-1) if speech_mask is not None else torch.ones_like(speech_feats[:, :, :1])
        text_mask_expanded = text_mask.unsqueeze(-1) if text_mask is not None else torch.ones_like(text_feats[:, :, :1])
        
        s_pooled = (speech_feats * speech_mask_expanded).sum(1) / speech_mask_expanded.sum(1)
        t_pooled = (text_feats * text_mask_expanded).sum(1) / text_mask_expanded.sum(1)
        
        # Project
        s_proj = self.speech_proj(s_pooled)
        t_proj = self.text_proj(t_pooled)
        
        # Apply gates
        s_gate = self.speech_gate(s_proj)
        t_gate = self.text_gate(t_proj)
        
        # Gated fusion
        fused = s_gate * s_proj + t_gate * t_proj
        
        return self.fusion_proj(fused)


class WeightedAverageFusion(nn.Module):
    """
    Attention-based weighted average of modalities.
    
    Advantages:
    - Simple yet effective
    - Learns global importance of each modality
    - Interpretable weights
    
    Mechanism:
    - Uses attention to compute scalar weights for each modality
    """
    def __init__(self, speech_dim=768, text_dim=768, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Attention for modality weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        self.output_dim = hidden_dim

    def forward(self, speech_feats, text_feats, speech_mask=None, text_mask=None):
        # Average pooling with mask
        speech_mask_expanded = speech_mask.unsqueeze(-1) if speech_mask is not None else torch.ones_like(speech_feats[:, :, :1])
        text_mask_expanded = text_mask.unsqueeze(-1) if text_mask is not None else torch.ones_like(text_feats[:, :, :1])
        
        s_pooled = (speech_feats * speech_mask_expanded).sum(1) / speech_mask_expanded.sum(1)
        t_pooled = (text_feats * text_mask_expanded).sum(1) / text_mask_expanded.sum(1)
        
        # Project
        s_proj = self.speech_proj(s_pooled)
        t_proj = self.text_proj(t_pooled)
        
        # Compute attention weights
        s_attn = self.attention(s_proj)
        t_attn = self.attention(t_proj)
        
        # Softmax over modalities
        attn_weights = torch.softmax(torch.cat([s_attn, t_attn], dim=-1), dim=-1)
        
        # Weighted combination
        fused = attn_weights[:, 0:1] * s_proj + attn_weights[:, 1:2] * t_proj
        
        return self.fusion_proj(fused)


class EmotionClassifier(nn.Module):
    """Final classification head"""
    def __init__(self, input_dim=512, num_classes=7, dropout=0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class MultimodalEmotionRecognition(nn.Module):
    """Complete multimodal emotion recognition model"""
    def __init__(self, config, fusion_type='cross_attention'):
        super().__init__()
        # Blocks 3: Modeling
        self.temporal_model = TransformerModeling(config['speech_dim'], config['hidden'], config['layers'])
        self.contextual_model = TransformerModeling(config['text_dim'], config['hidden'], config['layers'])

        # Block 4: Fusion (Choose variant)
        if fusion_type == 'cross_attention':
            self.fusion = CrossModalAttentionFusion(config['speech_dim'], config['text_dim'], config['f_hidden'])
        elif fusion_type == 'concatenation':
            self.fusion = ConcatenationFusion(config['speech_dim'], config['text_dim'], config['f_hidden'])
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(config['speech_dim'], config['text_dim'], config['f_hidden'])
        elif fusion_type == 'weighted':
            self.fusion = WeightedAverageFusion(config['speech_dim'], config['text_dim'], config['f_hidden'])
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Block 5: Classifier
        self.classifier = EmotionClassifier(self.fusion.output_dim, config['classes'])
        self.fusion_type = fusion_type

    def forward(self, s_x, t_x, s_m, t_m):
        s_feat = self.temporal_model(s_x, s_m)
        t_feat = self.contextual_model(t_x, t_m)
        fused = self.fusion(s_feat, t_feat, s_m, t_m)
        return self.classifier(fused), s_feat, t_feat, fused
