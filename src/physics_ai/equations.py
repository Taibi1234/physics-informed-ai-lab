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


def damped_oscillator_residual(
    t: torch.Tensor,
    x: torch.Tensor,
    omega: float = 1.0,
    zeta: float = 0.2,
) -> torch.Tensor:
    """Compute x'' + 2*zeta*omega*x' + omega^2*x (damped oscillator)."""
    if omega <= 0:
        raise ValueError("omega must be positive")
    if not (0 < zeta < 1):
        raise ValueError("zeta must be in (0, 1) for underdamped case")
    if not t.requires_grad:
        raise ValueError("t must have requires_grad=True")
    first_derivative = torch.autograd.grad(
        outputs=x.sum(), inputs=t, create_graph=True, retain_graph=True
    )[0]
    second_derivative = torch.autograd.grad(
        outputs=first_derivative.sum(), inputs=t, create_graph=True, retain_graph=True
    )[0]
    return second_derivative + (2 * zeta * omega) * first_derivative + (omega**2) * x


def exact_damped_position(
    t: np.ndarray | torch.Tensor, omega: float = 1.0, zeta: float = 0.2
):
    """Analytical solution for underdamped harmonic oscillator."""
    if omega <= 0:
        raise ValueError("omega must be positive")
    if not (0 < zeta < 1):
        raise ValueError("zeta must be in (0, 1) for underdamped case")
    t_arr = np.asarray(t, dtype=float)
    omega_d = omega * np.sqrt(1 - zeta**2)
    exp_term = np.exp(-zeta * omega * t_arr)
    cos_term = np.cos(omega_d * t_arr)
    sin_term = np.sin(omega_d * t_arr)
    return exp_term * (cos_term + (zeta * omega / omega_d) * sin_term)


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
