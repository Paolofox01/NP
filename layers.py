"""Building blocks of the Neural Process models: MLP, attention, Fourier features, decoder."""

import math
from typing import Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MLP", "MultiheadAttention", "FourierFeatures", "GaussianDecoder", "inverse_softplus"]


class MLP(nn.Module):
    """Stack of [norm] -> Linear -> activation -> [norm] -> [dropout] blocks.

    Acts on the last dimension, so inputs can have any number of leading
    dimensions: (..., in_dim) -> (..., out_dim).

    Args:
        in_dim: Size of the input features.
        out_dim: Size of the output. If None, no output layer is added and the
            module returns the last hidden representation (useful as a trunk).
        hidden_dim: Width of each hidden block.
        n_hidden: Number of hidden blocks.
        activation: Activation class, instantiated once per block.
        dropout: Dropout probability after each block (0 disables it).
        norm: Whether to apply LayerNorm in each block.
        norm_position: "pre" (before the Linear) or "post" (after the activation).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int],
        hidden_dim: int = 64,
        n_hidden: int = 1,
        activation: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        norm: bool = True,
        norm_position: str = "pre",
    ):
        super().__init__()
        if norm_position not in ("pre", "post"):
            raise ValueError(f"norm_position must be 'pre' or 'post', got {norm_position!r}")

        layers = []
        curr_dim = in_dim
        for _ in range(n_hidden):
            if norm and norm_position == "pre":
                layers.append(nn.LayerNorm(curr_dim))
            layers += [nn.Linear(curr_dim, hidden_dim), activation()]
            if norm and norm_position == "post":
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(curr_dim, out_dim) if out_dim is not None else nn.Identity()
        self.out_dim = out_dim if out_dim is not None else curr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


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


class FourierFeatures(nn.Module):
    """Random Fourier feature mapping: x -> [sin(2π xB), cos(2π xB)] (Tancik et al., 2020).

    Expects inputs normalized to a range of order 1; `scale` sets the frequency
    bandwidth relative to that range.

    Args:
        in_dim: Size of the input coordinates.
        num_frequencies: Number of frequencies; the output has 2 * num_frequencies features
            (plus in_dim if include_input).
        scale: Standard deviation of the Gaussian frequency matrix B.
        learnable: Train B instead of keeping it fixed.
        include_input: Concatenate the raw input to the features.
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int = 16,
        scale: float = 1.0,
        learnable: bool = False,
        include_input: bool = True,
    ):
        super().__init__()
        B = torch.randn(in_dim, num_frequencies) * scale
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
        self.include_input = include_input
        self.out_dim = 2 * num_frequencies + (in_dim if include_input else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (..., in_dim) -> (..., out_dim)
        proj = 2 * torch.pi * x @ self.B
        feats = [torch.sin(proj), torch.cos(proj)]
        if self.include_input:
            feats.insert(0, x)
        return torch.cat(feats, dim=-1)


def inverse_softplus(y: float) -> float:
    """x such that softplus(x) = y."""
    return math.log(math.expm1(y))


class GaussianDecoder(nn.Module):
    """Shared MLP trunk followed by per-group branches, each with a mean and a variance head.

    Without output_groups this is a standard MLP decoder. With output_groups, e.g. [1, 2]
    for (height, velocity), each group gets its own n_group_hidden layers on top of the
    shared trunk, so the groups can learn different features.

    Note: n_group_hidden=0 with several groups is equivalent to a single group, since
    separate linear heads on the same features are the same as one linear layer.

    Args:
        in_dim: Size of the input features.
        out_dim: Size of the output (y_dim).
        output_groups: Sizes of the output groups; must sum to out_dim. None means one group.
        hidden_dim: Width of all hidden layers.
        n_hidden: Number of shared trunk layers.
        n_group_hidden: Number of extra hidden layers per group branch.
        activation, dropout, norm, norm_position: Passed to the MLPs.
        min_var: Lower bound on the predicted variance.
        init_var: Approximate predicted variance at initialization.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        output_groups: Optional[Sequence[int]] = None,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        n_group_hidden: int = 1,
        activation: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        norm: bool = True,
        norm_position: str = "pre",
        min_var: float = 1e-6,
        init_var: float = 0.1,
    ):
        super().__init__()
        groups = list(output_groups) if output_groups is not None else [out_dim]
        if sum(groups) != out_dim:
            raise ValueError(f"output_groups {groups} must sum to out_dim ({out_dim})")
        if init_var <= min_var:
            raise ValueError("init_var must be larger than min_var")
        if len(groups) == 1:
            n_group_hidden = 0  # a single branch is just more trunk

        mlp_kwargs = dict(hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                          norm=norm, norm_position=norm_position)

        self.trunk = MLP(in_dim, None, n_hidden=n_hidden, **mlp_kwargs)
        self.branches = nn.ModuleList(
            MLP(self.trunk.out_dim, None, n_hidden=n_group_hidden, **mlp_kwargs) for _ in groups
        )
        self.mean_heads = nn.ModuleList(nn.Linear(b.out_dim, g) for b, g in zip(self.branches, groups))
        self.var_heads = nn.ModuleList(nn.Linear(b.out_dim, g) for b, g in zip(self.branches, groups))

        self.min_var = min_var
        with torch.no_grad():
            for head in self.var_heads:
                head.weight.mul_(0.1)  # start with nearly input-independent variance
                head.bias.fill_(inverse_softplus(init_var - min_var))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(..., in_dim) -> mean, var, each of shape (..., out_dim)."""
        h = self.trunk(x)
        means, raw_vars = [], []
        for branch, mean_head, var_head in zip(self.branches, self.mean_heads, self.var_heads):
            hb = branch(h)
            means.append(mean_head(hb))
            raw_vars.append(var_head(hb))
        mean = torch.cat(means, dim=-1)
        var = self.min_var + F.softplus(torch.cat(raw_vars, dim=-1))
        return mean, var