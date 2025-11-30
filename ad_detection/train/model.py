import torch
from torch import Tensor, nn
import torch.nn.functional as F
from config import XLSR_DIM_INPUT
import fairseq

########################XLSR-53-300m####################################
class SSLModel(nn.Module):
    """
    Args:
        device: Device (cuda/cpu)
        freeze_xlsr: Whether to freeze XLSR parameters
            - True: Freeze all parameters, only extract features (no XLSR update)
            - False: Unfreeze parameters, allow fine-tuning (will update XLSR)
    """
    def __init__(self, device, freeze_xlsr=False):
        super(SSLModel, self).__init__()
        
        if not freeze_xlsr:
            print("XLSR:Using fine-tuned XLSR model")
            cp_path = ''
        else:
            print("XLSR:Using original XLSR model")
            cp_path = '/Users/sleepwalker/Library/Mobile Documents/com~apple~CloudDocs/Code-In-iCloud/Adaptation_is_all_you_need/ad_detection/train/xlsr2_300m.pt'
        
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].to(device)
        self.device = device
        self.out_dim = XLSR_DIM_INPUT  # XLSR_FEATURE_DIM
        self.freeze_xlsr = freeze_xlsr
        
        # Control whether to freeze model based on freeze_xlsr parameter
        if freeze_xlsr:
            self.model.eval()  # Use eval mode when frozen
            """Freeze all XLSR parameters (no fine-tuning)"""
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()  # Use train mode for fine-tuning
            """Unfreeze all XLSR parameters (allow fine-tuning)"""
            for param in self.model.parameters():
                param.requires_grad = True
        

    def extract_feat(self, input_data):
        """
        Extract XLSR features
        
        Args:
            input_data: Audio input
        
        Returns:
            embedding: Output features from the last layer
            layerresult: Outputs from all layers
        """
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(device=input_data.device)
        if next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(dtype=input_data.dtype)
        
        # The XLSR-53 model expects input in the format: (Batch, Time)
        # Stereo audio is (Batch, Time, Channels), e.g., (32, 48000, 2)
        #   32 = batch size, 48000 = time steps, 2 = stereo channels
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]  # Keep only the first channel; it will automatically
                                            # be reduced to 2D (Batch, Time)
        # Mono audio, e.g., (32, 48000)
        else:
            input_tmp = input_data

        # Extract features
        model_output = self.model(input_tmp, mask=False, features_only=True)
        embedding = model_output['x']  # Features from the last layer (Batch, Time_downsampled, 1024),
                                    # Time_downsampled = original time steps / 320 (downsampling rate)
        layerresult = model_output['layer_results']  # Features from all layers

        return embedding, layerresult

# def XLSR_Average_Pooling(layerResult):
#     """
#     Average pooling for XLSR layer outputs.
    
#     For each layer, apply adaptive average pooling along the time dimension
#     to get a fixed-length representation.
    
#     Args:
#         layerResult: List of layer outputs, each with shape (Time, Batch, Feature)
    
#     Returns:
#         layery: Pooled features from all layers, shape (Batch, num_layers, XLSR_FEATURE_DIM)
#         fullfeature: Full features from all layers (concatenated)
#     """
#     poollayerResult = []
#     fullf = []
#     for layer in layerResult:
#         #layer[0] = (Time=201, Batch=32, Feature=XLSR_FEATURE_DIM)
#         layery = layer[0].permute(1, 2 ,0) #(Time, Batch, Feature=XLSR_FEATURE_DIM) —> (Batch, Feature=XLSR_FEATURE_DIM, Time)
#         layery = F.adaptive_avg_pool1d(layery, 1) #(Batch,Feature=XLSR_FEATURE_DIM,Time=1)
#         layery = layery.transpose(1, 2) # (B,F,1) → (B,1,F)
#         poollayerResult.append(layery)

#         x = layer[0].transpose(0, 1) # (T,B,F) → (B,T,F)
#         x = x.view(x.size(0), -1,x.size(1), x.size(2))
#         fullf.append(x)

#     layery = torch.cat(poollayerResult, dim=1)
#     fullfeature = torch.cat(fullf, dim=1)
#     return layery, fullfeature
############################################################


class PoolAttFF(nn.Module):
    """
    Attention pooling module

    Uses attention mechanism to pool variable-length sequences into fixed-length vectors.
    This module only performs attention pooling, without output mapping.

    Args:
        dim_hidden: Hidden dimension of input features
        dropout: Dropout rate for attention network
    """

    def __init__(self, dim_hidden, dropout):
        super().__init__()
        self.dim_hidden = dim_hidden

        # Attention network: hidden -> 2*hidden -> 1 (attention weights)
        self.linear1 = nn.Linear(self.dim_hidden, 2 * self.dim_hidden)
        self.linear2 = nn.Linear(2 * self.dim_hidden, 1)

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) - 1 for real data, 0 for padding (optional)

        Returns:
            x_pooled: (batch_size, hidden_dim) - pooled features
        """
        # Step 1: Compute attention scores
        # x: (B, L, H) -> (B, L, 2H) -> (B, L, 1)
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))

        # Step 2: Transpose for masking and softmax
        # (B, L, 1) -> (B, 1, L)
        att = att.transpose(2, 1)

        # Step 3: Apply mask - set padding positions to -inf so they become 0 after softmax
        if mask is not None:
            # (batch, seq_len) -> (batch, 1, seq_len)
            expanded_mask = mask.unsqueeze(1)
            
            #  1 -> False -> keep, 0 -> True ->mask out
            mask_positions = (expanded_mask == 0)

            # Fill masked positions with -inf
            att = att.masked_fill(mask_positions, float('-inf'))

        # softmax(-inf) = 0
        att = F.softmax(att, dim=2)  # (B, 1, L), sum over L = 1.0

        # att: (B, 1, L), x: (B, L, H) -> bmm -> (B, 1, H) -> squeeze -> (B, H)
        x_pooled = torch.bmm(att, x).squeeze(1)

        return x_pooled


############################################################
# Model classes for XLSR and eGeMAPS features
############################################################

class AD_XLSR_Model(nn.Module):
    """
    AD detection model specifically for XLSR features (1024-dim)

    Architecture:
        1. BatchNorm normalization
        2. Progressive 5-layer down projection: 1024→512→256→128→64→32
        3. Attention pooling (aggregate time dimension)
        4. Output mapping layer (32 → 2)

    Input:
        - x: (batch_size, seq_len, 1024) - XLSR features
        - mask: (batch_size, seq_len) - attention mask (optional)

    Output:
        - logits: (batch_size, 2) - Control and Dementia logits
    """

    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        # 1. BatchNorm normalization
        self.norm = nn.BatchNorm1d(1024)

        # 2. Progressive 5-layer down projection: 1024 → 512 → 256 → 128 → 64 → 32
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

        self.dropout = nn.Dropout(dropout)

        # 3. Attention pooling (aggregate time dimension)
        self.pool_ad = PoolAttFF(
            dim_hidden=32,
            dropout=dropout)

        # 4. Output mapping layer (32 → 2)
        self.output_layer = nn.Linear(32, 2)  # Binary classification
    
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, 1024) - XLSR features
            mask: (batch_size, seq_len) - attention mask (1=real, 0=padding), optional

        Returns:
            out: (batch_size, 2) - AD classification logits
        """
        # 1. BatchNorm: (B, L, 1024) -> (B, 1024, L) -> normalize -> (B, L, 1024)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. Progressive down projection
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
        x = self.dropout(x)

        # Layer 5: 64 → 32
        x = self.down_proj5(x)
        x = self.bn5(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)
        # x: (B, L, 32)

        # 3. Attention pooling (aggregate time dimension with mask)
        x_pooled = self.pool_ad(x, mask)  # (B, L, 32) -> (B, 32)

        # 4. Output mapping layer
        out = self.output_layer(x_pooled)  # (B, 32) -> (B, 2)

        return out


class AD_EGE_Model(nn.Module):
    """
    AD detection model specifically for eGeMAPS features (25-dim)

    Architecture:
        1. BatchNorm normalization
        2. Single-layer down projection to hidden dimension
        3. Attention pooling (aggregate time dimension)
        4. Output mapping layer (14 → 2)

    Input:
        - x: (batch_size, 10, 25) - eGeMAPS features

    Output:
        - logits: (batch_size, 2) - Control and Dementia logits
    """

    def __init__(self, dim_input=25, dim_hidden=14, dropout=0.2):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dropout = dropout

        # 1. BatchNorm normalization
        self.norm = nn.BatchNorm1d(self.dim_input)

        # 2. Single-layer down projection
        self.down_proj = nn.Linear(
            in_features=self.dim_input,
            out_features=self.dim_hidden,
        )
        self.down_proj_drop = nn.Dropout(self.dropout)
        self.down_proj_act = nn.ReLU()

        # 3. Attention pooling (aggregate time dimension)
        self.pool_ad = PoolAttFF(
            dim_hidden=self.dim_hidden,
            dropout=self.dropout)

        # 4. Output mapping layer (14 → 2)
        self.output_layer = nn.Linear(self.dim_hidden, 2)  # Binary classification
    
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, 25) - eGeMAPS features
            mask: Not used for eGeMAPS (no padding needed)
        
        Returns:
            out: (batch_size, 2) - AD classification logits
        """
        # 1. BatchNorm: (B, L, C) -> (B, C, L) -> normalize -> (B, L, C)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 2. Down projection to hidden dimension
        x = self.down_proj(x)
        x = self.down_proj_act(x)
        x = self.down_proj_drop(x)
        # x: (B, 10, 14)

        # 3. Attention pooling (aggregate time dimension)
        x_pooled = self.pool_ad(x, mask)  # (B, 10, 14) -> (B, 14)

        # 4. Output mapping layer
        out = self.output_layer(x_pooled)  # (B, 14) -> (B, 2)

        return out
