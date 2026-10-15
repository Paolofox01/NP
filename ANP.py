"""Attentive Neural Process (Kim et al., 2019)."""

import torch.nn as nn
from torch import Tensor

from layers import MultiheadAttention
from base import Mask, NeuralProcess, NPConfig

__all__ = ["AttentiveNP"]


class AttentiveNP(NeuralProcess):
    """Neural Process with attention in both the encoder and the deterministic path.

    - Self-attention: the context points exchange information before being pooled
      or attended to, so each representation knows about its neighbours.
    - Cross-attention: each target point builds its own deterministic representation
      by attending to the context, with e(x_target) as query, e(x_context) as keys and
      the context representations as values. Nearby context points get more weight,
      which is what lets the ANP fit the context far better than a mean-pooled NP.

    The same self-attended representations feed the latent path (pooled in the base class)
    and the cross-attention.
    """

    def __init__(self, cfg: NPConfig):
        super().__init__(cfg)
        self.self_attention = nn.ModuleList(
            MultiheadAttention(cfg.r_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.n_self_attention)
        )
        # Pre-norm layers leave the residual stream unnormalized: close the stack with one norm.
        self.self_attention_norm = (nn.LayerNorm(cfg.r_dim) if cfg.n_self_attention > 0
                                    else nn.Identity())
        self.cross_attention = MultiheadAttention(cfg.r_dim, cfg.num_heads, cfg.dropout)

    def encode_points(self, ex: Tensor, y: Tensor, mask: Mask = None) -> Tensor:
        s = super().encode_points(ex, y, mask)
        for layer in self.self_attention:
            s = layer(s, s, s, key_padding_mask=mask)
        return self.self_attention_norm(s)

    def deterministic_path(self, ex_context: Tensor, s_context: Tensor, ex_target: Tensor,
                           context_mask: Mask = None) -> Tensor:
        return self.cross_attention(query=ex_target, key=ex_context, value=s_context,
                                    key_padding_mask=context_mask)