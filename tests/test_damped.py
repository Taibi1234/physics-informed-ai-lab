"""Regression tests for the underdamped harmonic oscillator experiment."""

import numpy as np
import pytest
import torch

from physics_ai.data import sample_damped_observations
from physics_ai.equations import (
    damped_oscillator_residual,
    exact_damped_position,
)
from physics_ai.numerical_solver import solve_damped_oscillator


def test_damped_solution_satisfies_initial_conditions():
    t = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    position = exact_damped_position(t, omega=1.2, zeta=0.15)
    velocity = torch.autograd.grad(position.sum(), t)[0]

    assert position.item() == pytest.approx(1.0, abs=1e-12)
    assert velocity.item() == pytest.approx(0.0, abs=1e-12)


def test_damped_residual_for_exact_solution_is_small():
    t = torch.linspace(
        0.0, 4.0 * np.pi, 200, dtype=torch.float64
    ).reshape(-1, 1).requires_grad_(True)
    position = exact_damped_position(t, omega=1.2, zeta=0.15)
    residual = damped_oscillator_residual(
        t, position, omega=1.2, zeta=0.15
    )

    assert torch.max(torch.abs(residual)).item() < 1e-8


def test_damped_analytical_and_numerical_solutions_agree():
    times = np.linspace(0.0, 4.0 * np.pi, 500)
    _, numerical = solve_damped_oscillator(
        times, omega=1.2, zeta=0.15
    )
    analytical = exact_damped_position(
        times, omega=1.2, zeta=0.15
    )

    assert np.max(np.abs(numerical - analytical)) < 1e-6


def test_damped_observations_are_reproducible_and_use_damped_solution():
    first = sample_damped_observations(
        n_observations=10,
        seed=42,
        omega=1.2,
        zeta=0.15,
    )
    second = sample_damped_observations(
        n_observations=10,
        seed=42,
        omega=1.2,
        zeta=0.15,
    )

    for first_values, second_values in zip(first, second):
        np.testing.assert_allclose(first_values, second_values)

    times, _, clean = first
    expected = exact_damped_position(times, omega=1.2, zeta=0.15)
    np.testing.assert_allclose(clean, expected)


def test_damped_functions_reject_invalid_damping():
    times = np.linspace(0.0, 1.0, 10)

    with pytest.raises(ValueError, match="zeta"):
        exact_damped_position(times, zeta=1.0)
    with pytest.raises(ValueError, match="zeta"):
        solve_damped_oscillator(times, zeta=0.0)
    with pytest.raises(ValueError, match="zeta"):
        sample_damped_observations(zeta=-0.1)