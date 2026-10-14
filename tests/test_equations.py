import numpy as np
import pytest
import torch

from physics_ai.equations import exact_position, oscillator_residual
from physics_ai.numerical_solver import solve_oscillator


def test_exact_position_at_initial_time():
    assert float(exact_position(np.array([0.0]))[0]) == pytest.approx(1.0)


def test_analytical_and_numerical_solutions_agree():
    times = np.linspace(0.0, 2.0 * np.pi, 200)
    _, numerical = solve_oscillator(times)
    analytical = exact_position(times)
    assert np.max(np.abs(numerical - analytical)) < 1e-6


def test_oscillator_residual_for_exact_solution_is_small():
    t = torch.linspace(0.0, 2.0 * np.pi, 100).reshape(-1, 1).requires_grad_(True)
    x = torch.cos(t)
    residual = oscillator_residual(t, x, omega=1.0)
    assert torch.max(torch.abs(residual)).item() < 1e-5


def test_solver_rejects_unsorted_time_grid():
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_oscillator(np.array([0.0, 1.0, 0.5]))
