"""Multilayer perceptron, the building block of every encoder and decoder."""

from typing import Optional, Type

import torch
import torch.nn as nn

__all__ = ["MLP"]


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
