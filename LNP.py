"""Latent Neural Process with a global deterministic path (Garnelo et al., 2018)."""

from torch import Tensor

from layers import MLP
from base import Mask, NeuralProcess, NPConfig, masked_mean

__all__ = ["LatentNP"]


class LatentNP(NeuralProcess):
    """Neural Process whose deterministic representation is the same for every target point.

    The context is summarized by the mean of its point representations, which feeds both
    the latent path (in the base class) and a deterministic MLP here. Every target point
    then receives that one vector r, and differs only through its own coordinates.
    """

    def __init__(self, cfg: NPConfig):
        super().__init__(cfg)
        self.deterministic_encoder = MLP(cfg.r_dim, cfg.r_dim, hidden_dim=cfg.hidden_dim,
                                         n_hidden=1, activation=cfg.activation,
                                         dropout=cfg.dropout, norm=cfg.norm)

    def deterministic_path(self, ex_context: Tensor, s_context: Tensor, ex_target: Tensor,
                           context_mask: Mask = None) -> Tensor:
        r = self.deterministic_encoder(masked_mean(s_context, context_mask))  # (B, r_dim)
        return r.unsqueeze(1).expand(-1, ex_target.shape[1], -1)              # (B, T, r_dim)