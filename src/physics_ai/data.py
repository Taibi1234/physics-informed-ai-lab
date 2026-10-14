"""Synthetic observations for controlled experiments."""

from __future__ import annotations

import numpy as np

from physics_ai.equations import exact_position


def sample_observations(
    n_observations: int = 25,
    noise_std: float = 0.05,
    seed: int = 42,
    t_min: float = 0.0,
    t_max: float = 2.0 * np.pi,
    omega: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample times uniformly and return times, noisy positions, true positions."""
    if n_observations < 2:
        raise ValueError("n_observations must be at least 2")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")
    if not t_max > t_min:
        raise ValueError("t_max must be greater than t_min")
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(t_min, t_max, size=n_observations))
    clean = exact_position(t, omega=omega)
    observed = clean + rng.normal(loc=0.0, scale=noise_std, size=n_observations)
    return t, observed, clean
