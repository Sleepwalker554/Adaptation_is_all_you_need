"""
Domain Adversarial Neural Network (DANN) Models for AD Detection

This module contains:
1. Gradient Reversal Layer (GRL)
2. Domain Classifier
3. DANN versions of AD detection models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import Tensor
from model import PoolAttFF


############################################################
# Domain Adversarial Components
############################################################

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL)
    
    Forward: y = x (identity mapping)
    Backward: dy/dx = -lambda * grad_output (reverse gradient)
    
    This is the core of domain adversarial training:
    - During forward pass, features flow through unchanged
    - During backward pass, gradients are reversed and scaled by alpha
    - This makes the feature extractor learn domain-invariant features
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        """
        Args:
            x: input tensor
            alpha: scaling factor for gradient reversal (0 to 1)
        
        Returns:
            x unchanged
        """
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Reverse the gradient and scale by alpha
        
        Args:
            grad_output: gradient from next layer
        
        Returns:
            - Reversed and scaled gradient for x
            - None for alpha (no gradient needed)
        """
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Wrapper module for Gradient Reversal Function
    
    Usage:
        grl = GradientReversalLayer()
        reversed_features = grl(features, alpha=0.5)
    """
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
    
    def forward(self, x, alpha=1.0):
        """
        Args:
            x: input features (any shape)
            alpha: gradient reversal strength (0 to 1)
                - 0: no reversal (early training)
                - 1: full reversal (late training)
        
        Returns:
            x unchanged (but gradients will be reversed in backward pass)
        """
        return GradientReversalFunction.apply(x, alpha)


class DomainClassifier(nn.Module):
    """
    Domain Classifier for DANN
    
    Classifies whether features come from source domain (0) or target domain (1).
    Works in adversarial manner with feature extractor through GRL.
    
    Architecture:
        Input (feature_dim) → Hidden → Hidden/2 → Output (2 classes)
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        """
        Args:
            input_dim: dimension of input features (pooled feature size)
            hidden_dim: hidden layer dimension
            dropout: dropout rate
        """
        super(DomainClassifier, self).__init__()
        
        self.domain_classifier = nn.Sequential(
            # Layer 1: input_dim -> hidden_dim
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2: hidden_dim -> hidden_dim/2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer: hidden_dim/2 -> 2 (source vs target)
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) - pooled features
        
        Returns:
            domain_logits: (batch_size, 2) - domain classification logits
        """
        return self.domain_classifier(x)


def compute_alpha(current_step, total_steps, alpha_max=1.0):
    """
    Compute alpha for gradient reversal (progressive schedule)
    
    Alpha increases from 0 to alpha_max following the schedule from DANN paper:
    alpha = 2 * alpha_max / (1 + exp(-10 * p)) - alpha_max
    where p = current_step / total_steps
    
    This schedule ensures:
    - Early training (p≈0): alpha≈0, weak domain adversarial (focus on classification)
    - Mid training (p≈0.5): alpha increases gradually
    - Late training (p≈1): alpha≈alpha_max, strong domain adversarial
    
    Args:
        current_step: current training step (0-indexed)
        total_steps: total number of training steps
        alpha_max: maximum alpha value (default: 1.0)
    
    Returns:
        alpha: gradient reversal strength [0, alpha_max]
    """
    p = float(current_step) / float(total_steps)
    alpha = 2.0 * alpha_max / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - alpha_max
    return alpha.item()


############################################################
# DANN Models for Domain Adaptation
############################################################

class AD_XLSR_Model_DANN(nn.Module):
    """
    DANN version of AD XLSR Model with domain classifier
    
    Architecture:
        Input (B, L, 1024) → Feature Extractor → Features (B, L, 32)
                                                        ↓
                                                  Attention Pool
                                                        ↓
                                                Pooled Features (B, 32)
                                    ┌─────────────────┴─────────────────┐
                                    ↓                                   ↓
                            Class Classifier                   Gradient Reversal Layer
                                    ↓                                   ↓
                            AD Prediction (B, 2)              Domain Classifier
                                                                        ↓
                                                              Domain Prediction (B, 2)
    
    Training:
        - AD Classifier: learns to distinguish Control vs Dementia (source domain only)
        - Domain Classifier: learns to distinguish Source vs Target domain
        - Feature Extractor: learns features good for AD classification but bad for domain classification
    
    Inference:
        - Only use AD Classifier output (set return_features=False)
    """
    
    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        # ========== Feature Extractor (Shared) ==========
        # Same as original AD_XLSR_Model
        
        self.norm = nn.BatchNorm1d(1024)
        
        # Progressive down projection: 1024 → 512 → 256 → 128 → 64 → 32
        self.down_proj1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.down_proj2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.down_proj3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.down_proj4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.down_proj5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Attention pooling (aggregate time dimension)
        self.pool_att = PoolAttFF(dim_hidden=32, dropout=dropout)
        
        # ========== Task-specific Heads ==========
        
        # Class classifier (AD detection: Control vs Dementia)
        self.class_classifier = nn.Linear(32, 2)
        
        # Gradient Reversal Layer
        self.grl = GradientReversalLayer()
        
        # Domain classifier (dataset discrimination: Source vs Target)
        self.domain_classifier = DomainClassifier(
            input_dim=32,
            hidden_dim=64,
            dropout=dropout
        )
    
    def extract_features(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Extract features before pooling (shared feature extractor)
        
        Args:
            x: (batch_size, seq_len, 1024) - XLSR features
            mask: (batch_size, seq_len) - attention mask (1=real, 0=padding)
        
        Returns:
            features: (batch_size, seq_len, 32) - extracted features
        """
        # BatchNorm: (B, L, 1024) -> (B, 1024, L) -> normalize -> (B, L, 1024)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Progressive down projection with BatchNorm and ReLU
        # Layer 1: 1024 → 512
        x = self.down_proj1(x)
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        
        # Layer 2: 512 → 256
        x = self.down_proj2(x)
        x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        
        # Layer 3: 256 → 128
        x = self.down_proj3(x)
        x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        
        # Layer 4: 128 → 64
        x = self.down_proj4(x)
        x = self.bn4(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # Layer 5: 64 → 32
        x = self.down_proj5(x)
        x = self.bn5(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        return x  # (B, L, 32)
    
    def forward(self, x: Tensor, mask: Tensor = None, alpha: float = 0.0, return_features: bool = False):
        """
        Forward pass with optional domain classification
        
        Args:
            x: (batch_size, seq_len, 1024) - XLSR features
            mask: (batch_size, seq_len) - attention mask (1=real, 0=padding)
            alpha: gradient reversal strength (0 to 1)
                - Set to 0 in early training
                - Gradually increase to 1 during training
            return_features: whether to return domain classification
                - True: training mode (return both AD and domain predictions)
                - False: inference mode (return only AD prediction)
        
        Returns:
            if return_features=False:
                class_logits: (batch_size, 2) - AD classification logits
            if return_features=True:
                class_logits: (batch_size, 2) - AD classification logits
                domain_logits: (batch_size, 2) - domain classification logits
        """
        # Extract features (shared feature extractor)
        features = self.extract_features(x, mask)  # (B, L, 32)
        
        # Attention pooling (aggregate time dimension)
        pooled_features = self.pool_att(features, mask)  # (B, 32)
        
        # Class classification (AD detection)
        class_logits = self.class_classifier(pooled_features)  # (B, 2)
        
        if return_features:
            # Domain classification with gradient reversal
            reversed_features = self.grl(pooled_features, alpha)  # (B, 32), gradients reversed
            domain_logits = self.domain_classifier(reversed_features)  # (B, 2)
            return class_logits, domain_logits
        else:
            # Inference mode: only return AD classification
            return class_logits


class AD_EGE_Model_DANN(nn.Module):
    """
    DANN version of AD eGeMAPS Model with domain classifier

    Uses Baseline eGeMAPS architecture (25→64→32) as feature extractor for better classification performance

    Architecture:
        Input (B, 10, 25) → Feature Extractor → Features (B, 10, 32)
                                                      ↓
                                                Attention Pool
                                                      ↓
                                              Pooled Features (B, 32)
                                  ┌─────────────────┴─────────────────┐
                                  ↓                                   ↓
                          Class Classifier                   Gradient Reversal Layer
                                  ↓                                   ↓
                          AD Prediction (B, 2)              Domain Classifier
                                                                      ↓
                                                            Domain Prediction (B, 2)
    """

    def __init__(self, dim_input=25, dim_hidden=32, dropout=0.3):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dropout = dropout

        # ========== Feature Extractor (Shared) ==========
        # Adopts Baseline AD_EGE_Model architecture: 25 → 64 → 32

        # Layer 1: 25 → 64
        self.linear_layer1 = nn.Linear(25, 64)
        self.norm1 = nn.BatchNorm1d(64)

        # Layer 2: 64 → 32
        self.linear_layer2 = nn.Linear(64, 32)
        self.norm2 = nn.BatchNorm1d(32)

        self.dropout_layer = nn.Dropout(dropout)

        # Attention pooling (aggregate time dimension)
        self.pool_att = PoolAttFF(dim_hidden=32, dropout=dropout)

        # ========== Task-specific Heads ==========

        # Class classifier (AD detection: Control vs Dementia)
        self.class_classifier = nn.Linear(32, 2)

        # Gradient Reversal Layer
        self.grl = GradientReversalLayer()

        # Domain classifier (dataset discrimination: Source vs Target)
        self.domain_classifier = DomainClassifier(
            input_dim=32,
            hidden_dim=64,  # Increased from 32 to align with XLSR-DANN
            dropout=dropout
        )
    
    def extract_features(self, x: Tensor) -> Tensor:
        """
        Extract features before pooling (shared feature extractor)

        Args:
            x: (batch_size, seq_len, 25) - eGeMAPS features

        Returns:
            features: (batch_size, seq_len, 32) - extracted features
        """
        # Layer 1: 25 → 64
        x = self.linear_layer1(x)
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # Layer 2: 64 → 32
        x = self.linear_layer2(x)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout_layer(x)

        return x  # (B, seq_len, 32)
    
    def forward(self, x: Tensor, mask: Tensor = None, alpha: float = 0.0, return_features: bool = False):
        """
        Forward pass with optional domain classification

        Args:
            x: (batch_size, seq_len, 25) - eGeMAPS features
            mask: not used for eGeMAPS (no padding needed)
            alpha: gradient reversal strength (0 to 1)
            return_features: whether to return domain classification

        Returns:
            if return_features=False:
                class_logits: (batch_size, 2) - AD classification logits
            if return_features=True:
                class_logits: (batch_size, 2) - AD classification logits
                domain_logits: (batch_size, 2) - domain classification logits
        """
        # Extract features (shared feature extractor)
        features = self.extract_features(x)  # (B, seq_len, 32)

        # Attention pooling (aggregate time dimension)
        pooled_features = self.pool_att(features, mask)  # (B, 32)

        # Class classification (AD detection)
        class_logits = self.class_classifier(pooled_features)  # (B, 2)

        if return_features:
            # Domain classification with gradient reversal
            reversed_features = self.grl(pooled_features, alpha)  # (B, 32), gradients reversed
            domain_logits = self.domain_classifier(reversed_features)  # (B, 2)
            return class_logits, domain_logits
        else:
            # Inference mode: only return AD classification
            return class_logits