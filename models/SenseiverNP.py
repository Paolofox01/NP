"""Senseiver Neural Process: a Perceiver encoder over the context (Santos et al., 2023; the
same structure as the Latent Bottlenecked ANP of Feng et al., 2023).

- Encoder: an array of n_latents learned vectors reads the context with one cross-attention
  block, then is processed by n_blocks self-attention blocks among the latents only. Any number
  of context points becomes a fixed-size set of latents; the cost is linear in the context size.
- Targets: each e(x_target) cross-attends to the latents (one block), and never to the other
  targets.
- Latent path (use_latent=True): z from the latents, as set by cfg.pooling (their mean, a
  learned-query attention, or a Bayesian aggregation); for the posterior the latents read context
  and targets together, again in linear time.

With use_latent=False this is the conditional model of the papers (the original Senseiver is
also trained on the MSE, i.e. without a predicted variance).
"""

import torch
import torch.nn as nn
from torch import Tensor

from layers import TransformerBlock
from models.base import Mask, NeuralProcess, NPConfig

__all__ = ["SenseiverNP"]


class SenseiverNP(NeuralProcess):

    def __init__(self, cfg: NPConfig):
        super().__init__(cfg)
        block = lambda: TransformerBlock(cfg.r_dim, cfg.num_heads, cfg.dropout, cfg.ffn_multiplier,
                                         cfg.activation)
        self.latents = nn.Parameter(torch.randn(cfg.n_latents, cfg.r_dim) * 0.02)
        self.read = block()                                       # latents <- context points
        self.process = nn.ModuleList(block() for _ in range(cfg.n_blocks))  # latents <-> latents
        self.latent_norm = nn.LayerNorm(cfg.r_dim)
        self.query = block()                                      # targets <- latents
        self.target_norm = nn.LayerNorm(cfg.r_dim)

    def summarize(self, ex: Tensor, y: Tensor, mask: Mask = None):
        """The processed latents (B, n_latents, r_dim); no padding among them."""
        points = self.encode_points(ex, y, mask)
        latents = self.latents.expand(points.shape[0], -1, -1)
        latents = self.read(latents, points, points, key_padding_mask=mask)
        for block in self.process:
            latents = block(latents, latents, latents)
        return self.latent_norm(latents), None

    def deterministic_path(self, ex_context: Tensor, s_context: Tensor, ex_target: Tensor,
                           context_mask: Mask = None) -> Tensor:
        return self.target_norm(self.query(ex_target, s_context, s_context))
