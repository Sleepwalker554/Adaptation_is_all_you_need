import torch
from torch import Tensor, nn
import torch.nn.functional as F
from config import ModelConfig, XLSR_FEATURE_DIM
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
            print("Using fine-tuned model")
            cp_path = ''
        else:
            print("Using original model")
            cp_path = '/Users/sleepwalker/Library/Mobile Documents/com~apple~CloudDocs/Code-In-iCloud/Adaptation_is_all_you_need/ad_detection/train/xlsr2_300m.pt'
        
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].to(device)
        self.device = device
        self.out_dim = XLSR_FEATURE_DIM
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
        
        # Handle input dimensions
        # XLSR-53 模型期望的输入格式是：(Batch, Time)
        #立体声音频(Batch, Time, Channels), eg: (32, 48000, 2) 32=Batch size, 48000=time step, 2=双声道
        if input_data.ndim == 3: 
            input_tmp = input_data[:, :, 0] #只留第一个声道，会自动降维为2维 (Batch, Time)
        #单声道音频 eg: (32, 48000)
        else:  
            input_tmp = input_data
        
        # Extract features
        model_output = self.model(input_tmp, mask=False, features_only=True)
        embedding = model_output['x'] # 最后一层特征
        layerresult = model_output['layer_results']  # 所有层特征
        
        return embedding, layerresult

def XLSR_Average_Pooling(layerResult):
    """
    Average pooling for XLSR layer outputs.
    
    For each layer, apply adaptive average pooling along the time dimension
    to get a fixed-length representation.
    
    Args:
        layerResult: List of layer outputs, each with shape (Time, Batch, Feature)
    
    Returns:
        layery: Pooled features from all layers, shape (Batch, num_layers, XLSR_FEATURE_DIM)
        fullfeature: Full features from all layers (concatenated)
    """
    poollayerResult = []
    fullf = []
    for layer in layerResult:
        #layer[0] = (Time=201, Batch=32, Feature=XLSR_FEATURE_DIM)
        layery = layer[0].permute(1, 2 ,0) #(Time, Batch, Feature=XLSR_FEATURE_DIM) —> (Batch, Feature=XLSR_FEATURE_DIM, Time)
        layery = F.adaptive_avg_pool1d(layery, 1) #(Batch,Feature=XLSR_FEATURE_DIM,Time=1)
        layery = layery.transpose(1, 2) # (B,F,1) → (B,1,F)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1) # (T,B,F) → (B,T,F)
        x = x.view(x.size(0), -1,x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature


def XLSR_Attentive_Statistic_Pooling(layerResult):
    """
    Attentive Statistics Pooling for XLSR layer outputs.
    
    For each layer, use attention mechanism to compute weighted mean and standard deviation.
    This captures both the central tendency and variability of features across time.
    
    The attention weights are computed based on the feature magnitude (L2 norm),
    which helps focus on more informative time frames.
    
    Args:
        layerResult: List of layer outputs, each with shape (Time, Batch, Feature)
    
    Returns:
        layery: Concatenated [mean, std] for all layers, shape (Batch, num_layers, XLSR_FEATURE_DIM*2)
        fullfeature: Full features from all layers (concatenated)
    """
    poollayerResult = []
    fullf = []
    
    for layer in layerResult:
        # layer[0] = (Time=201, Batch=32, Feature=XLSR_FEATURE_DIM)
        x = layer[0].transpose(0, 1)  # (T,B,F) → (B,T,F)
        
        # === Attention mechanism ===
        # Compute attention weights based on feature magnitude
        # Higher magnitude features are typically more informative
        attention_logits = torch.norm(x, p=2, dim=-1, keepdim=True)  # (B,T,1) - L2 norm per time step
        attention_weights = F.softmax(attention_logits, dim=1)  # (B,T,1) - Normalize over time
        
        # === Weighted mean ===
        mean = torch.sum(x * attention_weights, dim=1, keepdim=True)  # (B,1,F)
        
        # === Weighted standard deviation ===
        # std = sqrt(E_w[(x - mean)^2]) where E_w is weighted expectation
        variance = torch.sum(
            attention_weights * (x - mean) ** 2,
            dim=1,
            keepdim=True
        )  # (B,1,F)
        std = torch.sqrt(variance.clamp(min=1e-8))  # (B,1,F), clamp to avoid NaN
        
        # === Concatenate mean and std ===
        stats = torch.cat([mean, std], dim=-1)  # (B,1,XLSR_FEATURE_DIM*2)
        poollayerResult.append(stats)
        
        # Full feature (same as average pooling)
        x_full = x.view(x.size(0), -1, x.size(1), x.size(2))
        fullf.append(x_full)
    
    layery = torch.cat(poollayerResult, dim=1)  # (B, num_layers, XLSR_FEATURE_DIM*2)
    fullfeature = torch.cat(fullf, dim=1)
    
    return layery, fullfeature

############################################################


class PoolAttFF(nn.Module):
    """
    Attention pooling module
    
    Uses attention mechanism to pool sequence into a single vector,
    then maps to output through feedforward network.
    
    Args:
        config: Model configuration
        out_dim: Output dimension (2 for AD binary classification)
    """
    
    def __init__(self, config: ModelConfig, out_dim: int):
        super().__init__()
        
        self.config = config
        
        # Attention network: hidden -> 2*hidden -> 1 (attention weights)
        self.linear1 = nn.Linear(config.dim_hidden, 2 * config.dim_hidden)
        self.linear2 = nn.Linear(2 * config.dim_hidden, 1)
        
        # Output mapping: hidden -> out_dim
        self.linear3 = nn.Linear(config.dim_hidden, out_dim)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """        
        Args:
            x: (batch_size, seq_len, hidden_dim)
        
        Returns:
            out: (batch_size, out_dim)
        """
        # x: (B, L, H) -> (B, L, 2H) -> (B, L, 1)
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        # Transpose and apply softmax: (B, L, 1) -> (B, 1, L) -> softmax
        att = att.transpose(2, 1)  # (B, 1, L)
        att = F.softmax(att, dim=2)  # Normalize on sequence dimension
        
        # att: (B, 1, L), x: (B, L, H) -> (B, 1, H) -> (B, H)
        x_pooled = torch.bmm(att, x).squeeze(1)
        
        # Map to output dimension
        out = self.linear3(x_pooled)  # (B, out_dim)
        
        return out


class ADModel(nn.Module):
    """
    1. BatchNorm normalization
    2. Down projection to hidden dimension
    3. Attention pooling
    4. Output 2-class logits
    
    Input:
        - x: (batch_size, 10, 25) - 10 time segments, each with 25-dim eGeMAPS features
    
    Output:
        - logits: (batch_size, 2) - Control and Dementia logits
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # 1. BatchNorm normalization (on feature dimension)
        self.norm = nn.BatchNorm1d(config.dim_input)
        
        # 2. Down projection layer: 25(eGeMAPS) or XLSR_FEATURE_DIM(XLSR) -> dim_hidden (default)
        self.down_proj = nn.Linear(
            in_features=config.dim_input,
            out_features=config.dim_hidden,
        )
        self.down_proj_drop = nn.Dropout(config.dropout)
        self.down_proj_act = nn.ReLU()
        
        # 3. Attention pooling + output layer
        self.pool_ad = PoolAttFF(config, out_dim=2)  # Binary classification
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, 10, 25) - eGeMAPS features / XLSR features
        
        Returns:
            out: (batch_size, 2) - AD classification logits
        """
        # 1. BatchNorm: (B, L, C) -> (B, C, L) -> normalize -> (B, L, C)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 2. Down projection to hidden dimension
        x = self.down_proj(x)         # (B, L, H)
        x = self.down_proj_act(x)
        x = self.down_proj_drop(x)
        
        # 3. Attention pooling + output
        out = self.pool_ad(x)          # (B, 2)

        return out


def create_model(config: ModelConfig = None) -> ADModel:
    """
    Args:
        config: Model configuration (optional, defaults to DEFAULT_CONFIG)
    """
    if config is None:
        from config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    return ADModel(config)