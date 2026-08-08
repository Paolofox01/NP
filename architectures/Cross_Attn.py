from typing import Type, Optional
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ["Cross_Attn"]

class Cross_Attn(nn.Module):
    """
    Cross-Attention mechanism for Neural Processes.
    
    Implements multi-head cross-attention using PyTorch's native scaled_dot_product_attention 
    for FlashAttention support, drastically reducing VRAM usage and speeding up training.
    """
    def __init__(self,
                 query_dim: int,
                 context_dim: int,
                 value_dim: int,
                 output_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        
        assert output_dim % num_heads == 0, f"output_dim ({output_dim}) must be divisible by num_heads ({num_heads})"
        
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.value_dim = value_dim  # In molti casi, value_dim è uguale a context_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(query_dim, output_dim)      # Queries from target
        self.k_proj = nn.Linear(context_dim, output_dim)    # Keys from context
        self.v_proj = nn.Linear(value_dim, output_dim)    # Values from context
        
        # Output projection
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        # Dropout
        self.dropout_p = dropout # Salviamo la probabilità per passarla a SDPA
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
    
    def forward(self, 
                query: torch.Tensor, 
                context: torch.Tensor,
                value : torch.Tensor, 
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size = query.shape[0]
        num_queries = query.shape[1]
        num_context = context.shape[1]
        num_values = value.shape[1]
        
        # Project and reshape in one go
        # (batch, num_points, output_dim) -> (batch, num_heads, num_points, head_dim)
        Q = self.q_proj(query).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(batch_size, num_context, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, num_values, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Gestione della maschera per SDPA
        attn_mask = None
        if context_mask is not None:
            # Il tuo context_mask originale usava True per i valori da MASCHERARE.
            # SDPA con maschere booleane richiede che True significhi "MANTIENI" e False "MASCHERA".
            # Quindi la invertiamo con `~`
            attn_mask = (~context_mask).unsqueeze(1).unsqueeze(2)
        
        # FLASH ATTENTION - Il cuore dell'ottimizzazione
        # Risolve il problema OOM scartando la matrice delle attenzioni dalla VRAM
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.output_dim)
        
        # Final projection & Layer Norm
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = self.norm(output)
        
        return output
    
    def get_attention_weights(self, 
                             query: torch.Tensor, 
                             context: torch.Tensor,
                             context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ATTENZIONE: Questa funzione forza il calcolo esplicito della matrice O(N^2).
        Usala solo per visualizzazioni (inferenza/debug), non nel training loop.
        """
        batch_size = query.shape[0]
        num_queries = query.shape[1]
        num_context = context.shape[1]
        
        # Project to Q, K
        Q = self.q_proj(query).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(batch_size, num_context, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute explicitly (Memory Intensive!)
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        
        if context_mask is not None:
            context_mask_expanded = context_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(context_mask_expanded, float('-inf'))
            
        return torch.softmax(attn_scores, dim=-1)