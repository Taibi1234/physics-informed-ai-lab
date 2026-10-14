"""Numerical reference solver for the harmonic oscillator."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def solve_oscillator(
    t_eval: np.ndarray,
    omega: float = 1.0,
    x0: float = 1.0,
    v0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, position) from SciPy's solve_ivp.

    The time grid must be one-dimensional, contain at least two points, and be
    strictly increasing. The default initial conditions match exact_position.
    """
    times = np.asarray(t_eval, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("t_eval must be a one-dimensional array with at least two points")
    if not np.all(np.isfinite(times)):
        raise ValueError("t_eval must contain only finite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("t_eval must be strictly increasing")
    if omega <= 0:
        raise ValueError("omega must be positive")

    def rhs(_t: float, state: np.ndarray) -> tuple[float, float]:
        position, velocity = state
        return velocity, -(omega**2) * position

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
