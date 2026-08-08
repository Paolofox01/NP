from typing import Type

import warnings

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

warnings.filterwarnings("ignore")

__all__ = ["Self_Attn"]

class Self_Attn(nn.Module):
    """
    Self-Attention mechanism for Neural Processes.
    
    Implements multi-head self-attention to allow context points to attend to each other,
    creating more expressive representations than simple mean pooling.
    
    Args:
        input_dim: Dimension of input features (x_dim + y_dim for context encoding)
        output_dim: Dimension of output representation
        num_heads: Number of attention heads. Default: 8
        dropout: Dropout probability. Default: 0.0
    
    Example:
        >>> attn = Self_Attn(input_dim=2, output_dim=128, num_heads=8)
        >>> x = torch.randn(16, 10, 2)  # (batch_size, num_points, input_dim)
        >>> output = attn(x)  # shape: (16, 10, 128)
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        
        assert output_dim % num_heads == 0, f"output_dim ({output_dim}) must be divisible by num_heads ({num_heads})"
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(input_dim, output_dim)
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        
        # Output projection
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for all linear layers."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, num_points, _ = x.shape

        Q = self.q_proj(x).view(batch_size, num_points, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, num_points, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, num_points, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            attn_mask = (~mask).unsqueeze(1).unsqueeze(2)  # same convention as Cross_Attn

        # No N×N matrix ever stored in VRAM
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_points, self.output_dim)
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        if self.input_dim == self.output_dim:
            output = self.norm(output + x)
        else:
            output = self.norm(output)

        return output