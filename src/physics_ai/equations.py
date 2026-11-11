"""Equations and analytical references for harmonic oscillators."""

from __future__ import annotations

import numpy as np
import torch


def exact_position(t: np.ndarray | torch.Tensor, omega: float = 1.0):
    """Solve x'' + omega^2*x = 0 with x(0)=1 and x'(0)=0."""
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
    """Compute x'' + omega^2*x using automatic differentiation."""
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


def _validate_damped_parameters(omega: float, zeta: float) -> None:
    if omega <= 0:
        raise ValueError("omega must be positive")
    if not 0 < zeta < 1:
        raise ValueError("zeta must be in (0, 1) for the underdamped case")


def exact_damped_position(
    t: np.ndarray | torch.Tensor,
    omega: float = 1.0,
    zeta: float = 0.2,
):
    """Solve the underdamped equation with x(0)=1 and x'(0)=0.

    The solution is exp(-zeta*omega*t) * [cos(omega_d*t)
    + zeta*omega/omega_d * sin(omega_d*t)], where
    omega_d = omega*sqrt(1-zeta^2).
    """
    _validate_damped_parameters(omega, zeta)
    omega_d = omega * np.sqrt(1.0 - zeta**2)
    coefficient = zeta * omega / omega_d

    if isinstance(t, torch.Tensor):
        envelope = torch.exp(-zeta * omega * t)
        return envelope * (
            torch.cos(omega_d * t)
            + coefficient * torch.sin(omega_d * t)
        )

    values = np.asarray(t, dtype=float)
    envelope = np.exp(-zeta * omega * values)
    return envelope * (
        np.cos(omega_d * values)
        + coefficient * np.sin(omega_d * values)
    )


def damped_oscillator_residual(
    t: torch.Tensor,
    x: torch.Tensor,
    omega: float = 1.0,
    zeta: float = 0.2,
) -> torch.Tensor:
    """Compute x'' + 2*zeta*omega*x' + omega^2*x."""
    _validate_damped_parameters(omega, zeta)
    if not t.requires_grad:
        raise ValueError("t must have requires_grad=True")
    first_derivative = torch.autograd.grad(
        outputs=x.sum(), inputs=t, create_graph=True, retain_graph=True
    )[0]
    second_derivative = torch.autograd.grad(
        outputs=first_derivative.sum(), inputs=t, create_graph=True, retain_graph=True
    )[0]
    return (
        second_derivative
        + (2 * zeta * omega) * first_derivative
        + (omega**2) * x
    )