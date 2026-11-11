"""Numerical reference solvers for harmonic oscillators."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def _validate_time_grid(t_eval: np.ndarray) -> np.ndarray:
    times = np.asarray(t_eval, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError(
            "t_eval must be a one-dimensional array with at least two points"
        )
    if not np.all(np.isfinite(times)):
        raise ValueError("t_eval must contain only finite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("t_eval must be strictly increasing")
    return times


def _solve_rhs(rhs, times: np.ndarray, x0: float, v0: float):
    solution = solve_ivp(
        rhs,
        t_span=(float(times[0]), float(times[-1])),
        y0=(x0, v0),
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"Numerical solver failed: {solution.message}")
    return solution.t, solution.y[0]


def solve_oscillator(
    t_eval: np.ndarray,
    omega: float = 1.0,
    x0: float = 1.0,
    v0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically solve the undamped harmonic oscillator."""
    times = _validate_time_grid(t_eval)
    if omega <= 0:
        raise ValueError("omega must be positive")

    def rhs(_t: float, state: np.ndarray) -> tuple[float, float]:
        position, velocity = state
        return velocity, -(omega**2) * position

    return _solve_rhs(rhs, times, x0, v0)


def solve_damped_oscillator(
    t_eval: np.ndarray,
    omega: float = 1.0,
    zeta: float = 0.2,
    x0: float = 1.0,
    v0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically solve the underdamped harmonic oscillator."""
    times = _validate_time_grid(t_eval)
    if omega <= 0:
        raise ValueError("omega must be positive")
    if not 0 < zeta < 1:
        raise ValueError("zeta must be in (0, 1) for the underdamped case")

    def rhs(_t: float, state: np.ndarray) -> tuple[float, float]:
        position, velocity = state
        acceleration = (
            -2.0 * zeta * omega * velocity
            - (omega**2) * position
        )
        return velocity, acceleration

    return _solve_rhs(rhs, times, x0, v0)