"""Gaussian decoder: MLP trunk with per-group mean and variance heads."""

from typing import Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mlp import MLP
from layers.utils import inverse_softplus

__all__ = ["GaussianDecoder"]


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
