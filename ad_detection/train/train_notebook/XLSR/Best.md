最后两层dropout = 0.2

class AD_XLSR_Model(nn.Module):
    """
    AD detection model specifically for XLSR features (1024-dim)
    
    Architecture:
        1. BatchNorm normalization
        2. Progressive 5-layer down projection: 1024→512→256→128→64→32
        3. Attention pooling
        4. Output 2-class logits
    
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

        # 3. Attention pooling + output layer
        self.pool_ad = PoolAttFF(
            dim_hidden=32,
            dropout=dropout,
            out_dim=2)  # Binary classification
    
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
        # x = self.dropout(x)
        
        # Layer 2: 512 → 256
        x = self.down_proj2(x)
        x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        # x = self.dropout(x)

        # Layer 3: 256 → 128
        x = self.down_proj3(x)
        x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        # x = self.dropout(x)

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

        # 3. Attention pooling + output (with mask)
        out = self.pool_ad(x, mask)   # (B, 2)

        return out