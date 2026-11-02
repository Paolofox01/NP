"""Fourier feature mapping of input coordinates."""

import torch
import torch.nn as nn

__all__ = ["FourierFeatures"]


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
