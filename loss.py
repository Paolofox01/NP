"""Training objective and evaluation metrics for the Neural Process models.

All terms are normalized per predicted value (per target point and output dimension),
so their size does not depend on how many target points or outputs a task has.
"""

from dataclasses import dataclass
import torch
from torch import Tensor
from torch.distributions import Normal, kl_divergence

from base import Mask, NPOutput

__all__ = ["LossTerms", "np_loss", "gaussian_nll", "masked_mse"]


@dataclass
class LossTerms:
    """The loss and its parts. Backpropagate `total`; log the others."""

    total: Tensor  # nll + kl_weight * kl
    nll: Tensor    # -E_z log p(y_target | z), per value
    kl: Tensor     # KL(q(z | C ∪ T) || q(z | C)), per value


def _float32(dist: Normal) -> Normal:
    """Same distribution in float32, so the loss is exact even under mixed precision."""
    return Normal(dist.mean.float(), dist.stddev.float())


def _values_per_task(target_mask: Mask, shape: torch.Size) -> Tensor:
    """Number of real predicted values in each task: (B,)."""
    B, T, D = shape
    if target_mask is None:
        return torch.full((B,), float(T * D))
    return ((~target_mask).sum(-1) * D).clamp(min=1).float()


def np_loss(
    out: NPOutput,
    y_target: Tensor,                      # (B, T, y_dim)
    target_mask: Mask = None,              # (B, T), True = padding
    kl_weight: float = 1.0,
) -> LossTerms:
    """Negative ELBO, normalized per predicted value.

    For each task:  [ -Σ log p(y_t | z)  +  kl_weight * KL(q(z | C ∪ T) || q(z | C)) ] / n_values
    then averaged over the batch. The model must have been called with y_target.
    """
    if out.z_posterior is None:
        raise ValueError("np_loss needs the posterior: call the model with y_target")

    log_lik = _float32(out.y).log_prob(y_target.float().unsqueeze(0)).mean(0).sum(-1)  # (B, T)
    if target_mask is not None:
        log_lik = log_lik.masked_fill(target_mask, 0.0)
    n_values = _values_per_task(target_mask, y_target.shape).to(log_lik.device)

    nll = (-log_lik.sum(-1) / n_values).mean()
    kl = (kl_divergence(_float32(out.z_posterior), _float32(out.z_prior)).sum(-1) / n_values).mean()
    return LossTerms(nll + kl_weight * kl, nll, kl)


def gaussian_nll(dist: Normal, y: Tensor, mask: Mask = None) -> Tensor:
    """Predictive NLL per value of y (B, T, D) under dist (S, B, T, D), ignoring padding.

    The validation metric: it measures the predictions themselves, with no KL and no
    dependence on the KL weight, so it is comparable across epochs and runs.
    """
    nll = -_float32(dist).log_prob(y.float().unsqueeze(0)).mean(0)  # (B, T, D)
    if mask is None:
        return nll.mean()
    keep = (~mask).unsqueeze(-1).float()
    return (nll * keep).sum() / (keep.sum() * nll.shape[-1]).clamp(min=1)


def masked_mse(dist: Normal, y: Tensor, mask: Mask = None) -> Tensor:
    """Squared error of the predictive mean (averaged over samples) per value, ignoring padding."""
    err = (dist.mean.float().mean(0) - y.float()) ** 2  # (B, T, D)
    if mask is None:
        return err.mean()
    keep = (~mask).unsqueeze(-1).float()
    return (err * keep).sum() / (keep.sum() * err.shape[-1]).clamp(min=1)