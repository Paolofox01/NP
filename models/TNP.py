"""Transformer Neural Process (Nguyen & Grover, 2022), and its Set Transformer version.

Context and target tokens go through the same stack of Transformer blocks (shared weights).
In every block the context tokens attend to the context, and the target tokens attend to the
context of that same block: the targets refine their query layer by layer, but never see each
other, so each prediction depends only on its own x and on the context (as with the attention
mask of the original TNP-D, here written without a mask: queries [context; targets], keys and
values = the context only).

- Context tokens: the (e(x), y) encodings of the base class. Target tokens: e(x_target), no y.
- n_inducing_points set (Set Transformer NP): every block reads the context through M learned
  inducing points (ISAB, Lee et al., 2019): H = attn(I, context), then context and targets
  attend to H. The cost per block is M * (N + T) instead of N * (N + T), and the targets read
  a summary of the context rather than the context points themselves.

With use_latent=True a global z is added as in the other models (pooled from the last context
layer); with use_latent=False it is the conditional TNP-D of the paper.
"""

import torch
import torch.nn as nn
from torch import Tensor

from layers import TransformerBlock
from models.base import Mask, NeuralProcess, NPConfig, masked_mean

__all__ = ["TransformerNP"]


class TransformerNP(NeuralProcess):

    def __init__(self, cfg: NPConfig):
        super().__init__(cfg)
        block = lambda: TransformerBlock(cfg.r_dim, cfg.num_heads, cfg.dropout, cfg.ffn_multiplier,
                                         cfg.activation)
        self.blocks = nn.ModuleList(block() for _ in range(cfg.n_blocks))
        self.n_inducing = cfg.n_inducing_points
        if self.n_inducing:
            self.inducing_points = nn.Parameter(torch.randn(cfg.n_blocks, self.n_inducing, cfg.r_dim) * 0.02)
            self.induce = nn.ModuleList(block() for _ in range(cfg.n_blocks))  # H = attn(I, context)
        self.norm = nn.LayerNorm(cfg.r_dim)  # pre-norm stack: one norm at the end

    def _keys(self, layer: int, context: Tensor, mask: Mask):
        """What the tokens of block `layer` attend to: the context, or its inducing summary H."""
        if not self.n_inducing:
            return context, mask
        inducing = self.inducing_points[layer].expand(context.shape[0], -1, -1)
        return self.induce[layer](inducing, context, context, key_padding_mask=mask), None

    def summarize(self, ex: Tensor, y: Tensor, mask: Mask = None):
        """The context tokens before every block and after the last: (B, N, n_blocks + 1, r_dim).

        The context stream never depends on the targets, so it is computed once here and read
        by the targets in deterministic_path."""
        context = self.encode_points(ex, y, mask)
        layers = [context]
        for i, block in enumerate(self.blocks):
            keys, key_mask = self._keys(i, context, mask)
            context = block(context, keys, keys, key_padding_mask=key_mask)
            layers.append(context)
        return torch.stack(layers, dim=2), mask

    def pool(self, summary: Tensor, mask: Mask = None) -> Tensor:
        return masked_mean(self.norm(summary[:, :, -1]), mask)

    def deterministic_path(self, ex_context: Tensor, s_context: Tensor, ex_target: Tensor,
                           context_mask: Mask = None) -> Tensor:
        target = ex_target
        for i, block in enumerate(self.blocks):
            keys, key_mask = self._keys(i, s_context[:, :, i], context_mask)
            target = block(target, keys, keys, key_padding_mask=key_mask)
        return self.norm(target)
