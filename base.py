"""Parts shared by every latent Neural Process model."""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from layers import MLP, FourierFeatures, GaussianDecoder, inverse_softplus

__all__ = ["NPConfig", "NPOutput", "NeuralProcess", "InputScaler", "Mask", "masked_mean"]

Mask = Optional[Tensor]  # (B, N) bool, True = padding point to ignore


# ----------------------------------------------------------------------------- config / output

@dataclass
class NPConfig:
    """All hyperparameters of a Neural Process model, in one place."""

    x_dim: int
    y_dim: int
    # Domain of x, mapped to [-1, 1] inside the model. A number applies to every dimension;
    # a list gives one bound per dimension. The defaults mean "x is already in [-1, 1]".
    x_low: Union[float, Sequence[float]] = -1.0
    x_high: Union[float, Sequence[float]] = 1.0
    r_dim: int = 128
    z_dim: int = 64
    hidden_dim: int = 128
    n_hidden: int = 2                  # hidden layers of the encoders
    activation: Type[nn.Module] = nn.ReLU
    dropout: float = 0.0
    norm: bool = True

    # Fourier features on the last `fourier_dims` columns of x (None = all columns)
    use_fourier: bool = False
    fourier_dims: Optional[int] = None
    num_frequencies: int = 64
    fourier_scale: float = 10.0
    learnable_fourier: bool = False
    fourier_include_input: bool = True

    # Decoder
    n_decoder_hidden: int = 3
    output_groups: Optional[Sequence[int]] = None
    n_group_hidden: int = 1
    min_var: float = 1e-4
    init_var: float = 0.1

    # Latent variable
    posterior_max_targets: Optional[int] = None
    min_latent_var: float = 1e-4

    # Attention (AttentiveNP only)
    num_heads: int = 8
    n_self_attention: int = 1          # self-attention layers in the encoder

    # Extras
    target_includes_context: bool = True  # are the context points also in the target set?


@dataclass
class NPOutput:
    """What every model returns, in every mode. S = samples, B = batch, T = target points."""

    y: Normal                          # predictive distribution, (S, B, T, y_dim)
    z_prior: Normal                    # q(z | context), (B, z_dim)
    z_posterior: Optional[Normal]      # q(z | context ∪ target), None if no y_target


# ----------------------------------------------------------------------------- helpers

def masked_mean(x: Tensor, mask: Mask = None) -> Tensor:
    """Mean over the points dimension: (B, N, D) -> (B, D), ignoring padded points."""
    if mask is None:
        return x.mean(dim=1)
    keep = (~mask).unsqueeze(-1).to(x.dtype)
    return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)


def _concat_sets(a: Tensor, b: Tensor, mask_a: Mask, mask_b: Mask) -> tuple[Tensor, Mask]:
    """Concatenate two point sets along dim 1, together with their padding masks."""
    if mask_a is None and mask_b is None:
        return torch.cat([a, b], dim=1), None
    if mask_a is None:
        mask_a = torch.zeros(a.shape[:2], dtype=torch.bool, device=a.device)
    if mask_b is None:
        mask_b = torch.zeros(b.shape[:2], dtype=torch.bool, device=b.device)
    return torch.cat([a, b], dim=1), torch.cat([mask_a, mask_b], dim=1)


Bounds = Union[float, Sequence[float], Tensor]


class InputScaler(nn.Module):
    """Maps x from the box [low, high] (per dimension) to [-1, 1].

    The bounds are buffers: not trained, but saved in the state_dict, so a loaded model
    normalizes exactly as it did in training. Use the physical domain bounds (box size,
    time span), not statistics of a data sample.

    Warns once if a scaled value falls outside [-tolerance, tolerance], which means either
    the bounds were never set or the input lies outside the training domain.
    """

    def __init__(self, dim: int, low: Bounds = -1.0, high: Bounds = 1.0, tolerance: float = 1.5):
        super().__init__()
        self.register_buffer("low", torch.empty(dim))
        self.register_buffer("high", torch.empty(dim))
        self.tolerance = tolerance
        self._warned = False
        self.set_range(low, high)

    def set_range(self, low: Bounds, high: Bounds) -> None:
        low = torch.as_tensor(low, dtype=self.low.dtype).expand_as(self.low)
        high = torch.as_tensor(high, dtype=self.high.dtype).expand_as(self.high)
        if not (high > low).all():
            raise ValueError(f"every high must be larger than its low, got low={low}, high={high}")
        self.low.copy_(low)
        self.high.copy_(high)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """x: (B, N, dim). mask: (B, N), True = padding, excluded from the range check."""
        x = 2 * (x - self.low) / (self.high - self.low) - 1
        if not self._warned:
            self._check_range(x if mask is None else x[~mask])
        return x

    def _check_range(self, x: Tensor) -> None:
        peak = x.detach().abs().max() if x.numel() else x.new_zeros(())
        if peak > self.tolerance:
            self._warned = True
            warnings.warn(
                f"Scaled inputs reach {peak.item():.2f}, outside "
                f"[-{self.tolerance}, {self.tolerance}]. Check the bounds given to the model "
                f"(low={self.low.tolist()}, high={self.high.tolist()}), or whether these inputs "
                "lie outside the training domain."
            )

    def extra_repr(self) -> str:
        return f"low={self.low.tolist()}, high={self.high.tolist()}"


class XFeaturizer(nn.Module):
    """Optionally applies Fourier features to the last `fourier_dims` columns of x."""

    def __init__(self, cfg: NPConfig):
        super().__init__()
        if not cfg.use_fourier:
            self.fourier, self.n_plain, self.out_dim = None, cfg.x_dim, cfg.x_dim
            return
        k = cfg.x_dim if cfg.fourier_dims is None else cfg.fourier_dims
        if not 1 <= k <= cfg.x_dim:
            raise ValueError(f"fourier_dims must be in [1, {cfg.x_dim}], got {k}")
        self.n_plain = cfg.x_dim - k
        self.fourier = FourierFeatures(k, cfg.num_frequencies, cfg.fourier_scale,
                                       learnable=cfg.learnable_fourier,
                                       include_input=cfg.fourier_include_input)
        self.out_dim = self.n_plain + self.fourier.out_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.fourier is None:
            return x
        return torch.cat([x[..., :self.n_plain], self.fourier(x[..., self.n_plain:])], dim=-1)


class XEncoder(nn.Module):
    """Shared coordinate embedding e(x): [Fourier features] -> MLP, (..., x_dim) -> (..., r_dim).

    Expects x already scaled to [-1, 1] (done by the model's x_scaler).

    Computed once per point and reused by the (x, y) encoder, the attention and the decoder.
    """

    def __init__(self, cfg: NPConfig):
        super().__init__()
        if cfg.hidden_dim < cfg.r_dim:
            raise ValueError(f"hidden_dim ({cfg.hidden_dim}) should be >= r_dim ({cfg.r_dim}), "
                             "or the coordinate embedding becomes a bottleneck")
        self.featurizer = XFeaturizer(cfg)
        self.mlp = MLP(self.featurizer.out_dim, cfg.r_dim, hidden_dim=cfg.hidden_dim,
                       n_hidden=cfg.n_hidden, activation=cfg.activation, norm=cfg.norm,
                       norm_position="post")  # post-norm: LayerNorm on raw coordinates is meaningless

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.featurizer(x))


class LatentEncoder(nn.Module):
    """Pooled representation (B, r_dim) -> Normal over z, (B, z_dim)."""

    def __init__(self, cfg: NPConfig):
        super().__init__()
        self.net = MLP(cfg.r_dim, 2 * cfg.z_dim, hidden_dim=cfg.hidden_dim, n_hidden=cfg.n_hidden,
                       activation=cfg.activation, norm=cfg.norm)
        self.min_var = cfg.min_latent_var
        with torch.no_grad():  # start with unit variance
            self.net.head.weight[cfg.z_dim:].mul_(0.1)
            self.net.head.bias[cfg.z_dim:].fill_(inverse_softplus(1.0 - self.min_var))

    def forward(self, r: Tensor) -> Normal:
        mu, raw = self.net(r).chunk(2, dim=-1)
        return Normal(mu, (self.min_var + F.softplus(raw)).sqrt())


# ----------------------------------------------------------------------------- base model

class NeuralProcess(nn.Module, ABC):
    """Latent Neural Process skeleton.

    Pipeline (x is given in physical units and scaled to [-1, 1] first):
        x                       -> x_encoder           -> e(x)  (B, N, r_dim)
        (e(x), y) per point     -> encode_points       -> s_i   (B, N, r_dim)
        mean of s_i             -> latent encoder      -> q(z)
        context, e(x_target)    -> deterministic_path  -> r     (B, T, r_dim)
        [z, r, e(x_t), x_t]     -> decoder             -> p(y)

    Subclasses implement `deterministic_path` and may extend `encode_points`.
    """

    def __init__(self, cfg: NPConfig):
        super().__init__()
        self.cfg = cfg
        mlp_kwargs = dict(hidden_dim=cfg.hidden_dim, activation=cfg.activation, norm=cfg.norm)

        self.x_scaler = InputScaler(cfg.x_dim, cfg.x_low, cfg.x_high)
        self.x_encoder = XEncoder(cfg)
        self.y_embed = nn.Linear(cfg.y_dim, cfg.r_dim)  # so y is not drowned out by e(x)
        self.xy_encoder = MLP(2 * cfg.r_dim, cfg.r_dim, n_hidden=cfg.n_hidden, **mlp_kwargs)
        self.latent_encoder = LatentEncoder(cfg)
        self.decoder = GaussianDecoder(
            cfg.z_dim + 2 * cfg.r_dim + cfg.x_dim, cfg.y_dim, output_groups=cfg.output_groups,
            n_hidden=cfg.n_decoder_hidden, n_group_hidden=cfg.n_group_hidden,
            dropout=cfg.dropout, min_var=cfg.min_var, init_var=cfg.init_var, **mlp_kwargs,
        )

    # --- hooks for subclasses -------------------------------------------------

    def encode_points(self, ex: Tensor, y: Tensor, mask: Mask = None) -> Tensor:
        """Per-point representations of (e(x), y) pairs: (B, N, r_dim)."""
        return self.xy_encoder(torch.cat([ex, self.y_embed(y)], dim=-1))

    @abstractmethod
    def deterministic_path(self, ex_context: Tensor, s_context: Tensor, ex_target: Tensor,
                           context_mask: Mask = None) -> Tensor:
        """Deterministic representation for each target point: (B, T, r_dim)."""

    # --- forward --------------------------------------------------------------

    def forward(
        self,
        x_context: Tensor,                 # (B, N, x_dim)
        y_context: Tensor,                 # (B, N, y_dim)
        x_target: Tensor,                  # (B, T, x_dim)
        y_target: Optional[Tensor] = None, # (B, T, y_dim)
        num_samples: int = 1,
        context_mask: Mask = None,
        target_mask: Mask = None,
        use_posterior: Optional[bool] = None,
    ) -> NPOutput:
        """Predict y at x_target given the context.

        use_posterior: sample z from q(z | context ∪ target) instead of q(z | context).
            None (default) means "in training mode, if y_target is given", which is what
            the ELBO needs and keeps validation from ever seeing y_target.
        """
        x_context = self.x_scaler(x_context, context_mask)
        x_target = self.x_scaler(x_target, target_mask)
        ex_c, ex_t = self.x_encoder(x_context), self.x_encoder(x_target)

        s_c = self.encode_points(ex_c, y_context, context_mask)
        pooled_c = masked_mean(s_c, context_mask)
        z_prior = self.latent_encoder(pooled_c)
        z_posterior = (self._posterior(ex_c, y_context, context_mask, ex_t, y_target, target_mask)
                       if y_target is not None else None)

        if use_posterior is None:
            use_posterior = self.training and z_posterior is not None
        if use_posterior and z_posterior is None:
            raise ValueError("use_posterior=True needs y_target")
        z = (z_posterior if use_posterior else z_prior).rsample((num_samples,))  # (S, B, z_dim)

        r_t = self.deterministic_path(ex_c, s_c, ex_t, context_mask)
        y_dist = self._decode(z, r_t, ex_t, x_target)
        return NPOutput(y_dist, z_prior, z_posterior)

    # --- internals ------------------------------------------------------------

    def _posterior(self, ex_c: Tensor, y_c: Tensor, mask_c: Mask,
                   ex_t: Tensor, y_t: Tensor, mask_t: Mask) -> Normal:
        """q(z | context ∪ target)."""
        k = self.cfg.posterior_max_targets               # e.g. 512; None = use all targets
        if k is not None and ex_t.shape[1] > k:
            keep = torch.randperm(ex_t.shape[1], device=ex_t.device)[:k]
            ex_t, y_t = ex_t[:, keep], y_t[:, keep]
            mask_t = mask_t[:, keep] if mask_t is not None else None
        if self.cfg.target_includes_context:
            ex, y, mask = ex_t, y_t, mask_t
        else:
            ex, mask = _concat_sets(ex_c, ex_t, mask_c, mask_t)
            y = torch.cat([y_c, y_t], dim=1)
        return self.latent_encoder(masked_mean(self.encode_points(ex, y, mask), mask))

    def _decode(self, z: Tensor, r_t: Tensor, ex_t: Tensor, x_t: Tensor) -> Normal:
        """z (S, B, z_dim), r_t and ex_t (B, T, r_dim), x_t (B, T, x_dim) -> p(y), (S, B, T, y_dim).

        The raw x_t is passed too, so the exact position is never lost in the embedding.
        """
        S, (B, T) = z.shape[0], x_t.shape[:2]
        per_point = torch.cat([r_t, ex_t, x_t], dim=-1).unsqueeze(0).expand(S, -1, -1, -1)
        decoder_in = torch.cat([z.unsqueeze(2).expand(S, B, T, -1), per_point], dim=-1)
        mean, var = self.decoder(decoder_in)
        return Normal(mean, var.sqrt())