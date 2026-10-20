"""Pre-norm multi-head attention, for the self- and cross-attention of the AttentiveNP."""

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["MultiheadAttention"]


class MultiheadAttention(nn.Module):
    """Pre-norm multi-head attention: query + attention(norm(query), norm(key), norm(value)).

    Self-attention:  attn(x, x, x)
    Cross-attention: attn(query, key, value), output has the query's shape.

    The residual path is never normalized, so stacks of these layers train stably.
    A stack should end with one LayerNorm (see AttentiveNP).

    Args:
        embed_dim: Size of the query/key/value embeddings and of the output.
        num_heads: Number of attention heads (must divide embed_dim).
        dropout: Dropout on the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"num_heads ({num_heads}) must divide embed_dim ({embed_dim})")
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_key = nn.LayerNorm(embed_dim)
        self.norm_value = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        query: torch.Tensor,                       # (B, N_q, D)
        key: torch.Tensor,                         # (B, N_k, D)
        value: torch.Tensor,                       # (B, N_k, D)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, N_k), True = ignore
        return_weights: bool = False,
    ):
        out, weights = self.attention(
            self.norm_query(query), self.norm_key(key), self.norm_value(value),
            key_padding_mask=key_padding_mask,
            need_weights=return_weights,
            average_attn_weights=False,            # per-head weights when requested
        )
        out = query + out
        return (out, weights) if return_weights else out
