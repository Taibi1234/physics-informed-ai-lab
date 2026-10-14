"""Equations and reference functions for the harmonic oscillator."""

from __future__ import annotations

import numpy as np
import torch


def exact_position(t: np.ndarray | torch.Tensor, omega: float = 1.0):
    """Return the analytical position for x'' + omega^2*x = 0, x(0)=1, x'(0)=0."""
    if omega <= 0:
        raise ValueError("omega must be positive")
    if isinstance(t, torch.Tensor):
        return torch.cos(omega * t)
    return np.cos(omega * np.asarray(t, dtype=float))


def oscillator_residual(
    t: torch.Tensor,
    x: torch.Tensor,
    omega: float = 1.0,
) -> torch.Tensor:
    """Compute x_tt + omega^2*x using automatic differentiation.

    `t` must have requires_grad=True, and `x` must be computed from `t` by a
    differentiable model. The returned tensor is suitable for a PINN loss.
    """
    if omega <= 0:
        raise ValueError("omega must be positive")
    if not t.requires_grad:
        raise ValueError("t must have requires_grad=True")
    first_derivative = torch.autograd.grad(
        outputs=x.sum(), inputs=t, create_graph=True, retain_graph=True
    )[0]
    second_derivative = torch.autograd.grad(
        outputs=first_derivative.sum(), inputs=t, create_graph=True, retain_graph=True
    )[0]
    return second_derivative + (omega**2) * x
