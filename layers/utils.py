"""Small helpers shared by the layers and the models."""

import math

__all__ = ["inverse_softplus"]


def inverse_softplus(y: float) -> float:
    """x such that softplus(x) = y."""
    return math.log(math.expm1(y))
