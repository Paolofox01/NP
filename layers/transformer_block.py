"""Transformer block: pre-norm attention followed by a pre-norm feed-forward network."""

from typing import Optional, Type

import torch
import torch.nn as nn

from layers.attention import MultiheadAttention
from layers.mlp import MLP

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    """h = query + attention(query, key, value);  out = h + FFN(norm(h)).

    Self-attention: block(x, x, x). Cross-attention: block(query, key, value), output shaped
    like the query. Both residual paths are left unnormalized (pre-norm), so a stack of blocks
    should end with one LayerNorm.

    Args:
        dim: Size of the tokens.
        num_heads: Attention heads (must divide dim).
        dropout: Dropout on the attention weights and in the feed-forward network.
        ffn_multiplier: Hidden width of the feed-forward network, in multiples of dim
            (4 in the original Transformer; 2 is enough for small models).
        activation: Activation of the feed-forward network.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, ffn_multiplier: int = 2,
                 activation: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.attention = MultiheadAttention(dim, num_heads, dropout)
        # LayerNorm -> Linear(dim, k*dim) -> activation -> [dropout] -> Linear(k*dim, dim)
        self.ffn = MLP(dim, dim, hidden_dim=ffn_multiplier * dim, n_hidden=1, activation=activation,
                       dropout=dropout, norm=True, norm_position="pre")

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.attention(query, key, value, key_padding_mask=key_padding_mask)
        return h + self.ffn(h)
