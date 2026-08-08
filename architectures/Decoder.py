from typing import Type
import math
import warnings

import torch
import torch.nn as nn

# Export both decoders so they can be imported in your main scripts
__all__ = ["Decoder", "DeepONetDecoder"]

# ==============================================================================
# 1. THE STANDARD MLP DECODER (Concatenation-based)
# ==============================================================================
class Decoder(nn.Module):
    """
    Standard Decoder network for transforming concatenated latent representations 
    and target coordinates back to the output space via a deep MLP.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 n_hidden: int = 1,
                 activation: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.0,
                 is_normalized: bool = True,
                 norm_type: str = 'layer',
                 norm_position: str = 'pre'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = dropout
        self.is_normalized = is_normalized
        self.norm_type = norm_type
        self.norm_position = norm_position
        
        layers = []
        curr_dim = input_dim
        
        for i in range(n_hidden):
            # Pre-normalization
            if is_normalized and norm_position == 'pre':
                if norm_type == 'layer':
                    layers.append(nn.LayerNorm(curr_dim))
                elif norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(curr_dim))
                else:
                    raise ValueError(f"norm_type must be 'layer' or 'batch', got '{norm_type}'")
            
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(activation())
            
            # Post-normalization
            if is_normalized and norm_position == 'post':
                if norm_type == 'layer':
                    layers.append(nn.LayerNorm(hidden_dim))
                elif norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(hidden_dim))
                else:
                    raise ValueError(f"norm_type must be 'layer' or 'batch', got '{norm_type}'")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                    
            curr_dim = hidden_dim
        
        # Output Layer (Outputs raw logits for mean and variance)
        layers.append(nn.Linear(curr_dim, 2 * output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> tuple:
        y_params = self.decoder(x)
        output1, output2 = torch.chunk(y_params, 2, dim=-1)
        return output1, output2


# ==============================================================================
# 2. THE DEEPONET SPLIT DECODER (Branch/Trunk-based)
# ==============================================================================
class DeepONetDecoder(nn.Module):
    """
    DeepONet Split Decoder.
    
    Transforms the global latent representation (Branch) and the local 
    spatial/attentive representations (Trunk) independently before merging 
    them via a scaled dot product. Solves the memory explosion of standard MLPs.
    """
    def __init__(self,
                 branch_dim: int,
                 trunk_dim: int,
                 output_dim: int,
                 p: int = 128,
                 hidden_dim: int = 64,
                 n_hidden: int = 1,
                 activation: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.0,
                 is_normalized: bool = True,
                 norm_type: str = 'layer',
                 norm_position: str = 'pre'):
        super().__init__()
        
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.output_dim = output_dim
        self.p = p
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = dropout
        self.is_normalized = is_normalized
        self.norm_type = norm_type
        self.norm_position = norm_position

        # Helper Methods for Network Construction
        def get_norm(dim):
            if norm_type == 'layer':
                return nn.LayerNorm(dim)
            elif norm_type == 'batch':
                return nn.BatchNorm1d(dim)
            else:
                raise ValueError(f"norm_type must be 'layer' or 'batch', got '{norm_type}'")

        def build_base(in_features):
            """Builds the shared Base MLP for Branch or Trunk."""
            layers = []
            curr_dim = in_features
            for _ in range(n_hidden):
                if is_normalized and norm_position == 'pre':
                    layers.append(get_norm(curr_dim))
                
                layers.append(nn.Linear(curr_dim, hidden_dim))
                layers.append(activation())
                
                if is_normalized and norm_position == 'post':
                    layers.append(get_norm(hidden_dim))
                    
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                    
                curr_dim = hidden_dim
            return nn.Sequential(*layers), curr_dim

        def build_head(in_features):
            """Builds the dedicated feature heads (1 hidden layer)."""
            return nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, output_dim * p)
            )

        # Network Architecture: Shared Bases
        self.branch_base, branch_out_dim = build_base(branch_dim)
        self.trunk_base, trunk_out_dim = build_base(trunk_dim)

        # Independent Heads (Mean and Variance paths)
        self.branch_mean_head = build_head(branch_out_dim)
        self.branch_var_head = build_head(branch_out_dim)
        
        self.trunk_mean_head = build_head(trunk_out_dim)
        self.trunk_var_head = build_head(trunk_out_dim)

        # Final Biases (Raw output space)
        self.mean_bias = nn.Parameter(torch.zeros(output_dim))
        self.var_bias = nn.Parameter(torch.ones(output_dim) * -3.0) 

    def forward(self, branch_x: torch.Tensor, trunk_x: torch.Tensor) -> tuple:
        # Ensure 3D dimensionality for branch_x (S, B, D) if user passes (B, D)
        if branch_x.dim() == 2:
            branch_x = branch_x.unsqueeze(0)
            
        # 1. Pass through shared bases
        b_features = self.branch_base(branch_x)
        t_features = self.trunk_base(trunk_x)

        # 2. Pass through dedicated heads
        b_mean = self.branch_mean_head(b_features)
        b_var = self.branch_var_head(b_features)
        
        t_mean = self.trunk_mean_head(t_features)
        t_var = self.trunk_var_head(t_features)

        # 3. Reshape to separate output_dim and latent projection p
        # branch shape becomes: (num_samples, batch_size, output_dim, p)
        b_mean = b_mean.view(*branch_x.shape[:-1], self.output_dim, self.p)
        b_var = b_var.view(*branch_x.shape[:-1], self.output_dim, self.p)

        # trunk shape becomes: (batch_size, num_target, output_dim, p)
        t_mean = t_mean.view(*trunk_x.shape[:-1], self.output_dim, self.p)
        t_var = t_var.view(*trunk_x.shape[:-1], self.output_dim, self.p)

        # 4. Add broadcasting dimensions
        # Branch adds 'target' dim: (num_samples, batch, 1, output_dim, p)
        b_mean = b_mean.unsqueeze(2)
        b_var = b_var.unsqueeze(2)

        # Trunk adds 'samples' dim: (1, batch, target, output_dim, p)
        t_mean = t_mean.unsqueeze(0)
        t_var = t_var.unsqueeze(0)

        # 5. Scaled Dot Product over the 'p' dimension
        mean_raw = torch.sum(b_mean * t_mean, dim=-1) / math.sqrt(self.p)
        var_raw = torch.sum(b_var * t_var, dim=-1) / math.sqrt(self.p)

        # 6. Apply Biases
        mean = mean_raw + self.mean_bias
        var = var_raw + self.var_bias

        return mean, var