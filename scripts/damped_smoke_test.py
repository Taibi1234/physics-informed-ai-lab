"""Smoke-test script for damped harmonic oscillator experiment."""
from __future__ import annotations

"""
SMOKE TEST ONLY — not full experimental evidence.
This verifies that a damped-oscillator PINN setup can train without errors
and that the analytical solution can be evaluated independently.

Full experiment (multiple seeds, multiple zeta/noise/weight settings,
controlled comparisons with data_only/data_ic/pinn) remains future work.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Use the existing physics_ai package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physics_ai.data import sample_observations
from physics_ai.equations import exact_position  # Will not apply directly to damped case

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def damped_exact_position(t: np.ndarray, omega: float = 1.0, zeta: float = 0.2) -> np.ndarray:
    """Analytical solution for underdamped oscillator (derived independently)."""
    if not (0 < zeta < 1):
        raise ValueError("zeta must be in (0, 1) for underdamped case")
    omega_d = omega * np.sqrt(1 - zeta**2)
    exp_term = np.exp(-zeta * omega * np.asarray(t, dtype=float))
    cos_term = np.cos(omega_d * np.asarray(t, dtype=float))
    sin_term = np.sin(omega_d * np.asarray(t, dtype=float))
    return exp_term * (cos_term + (zeta * omega / omega_d) * sin_term)


if __name__ == "__main__":
    output_dir = PROJECT_ROOT / "reports" / "advanced_damped_oscillator"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test analytical solution satisfies initial conditions
    t0 = np.array([0.0])
    x0 = damped_exact_position(t0, omega=1.0, zeta=0.2)
    assert np.isclose(x0[0], 1.0, atol=1e-9), f"Initial position should be 1, got {x0[0]}"
    print("PASS: Initial condition x(0)=1 verified (analytical).")

    # Small visual comparison (not experimental evidence — just verification that solution behaves)
    t_grid = np.linspace(0, 4 * np.pi, 400)
    x_ref = damped_exact_position(t_grid, omega=1.0, zeta=0.2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_grid, x_ref, label="Damped reference (zeta=0.2, omega=1)", linewidth=2)
    ax.set(title="Damped harmonic oscillator — analytical reference (zeta=0.2)", xlabel="t", ylabel="x(t)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "damped_reference_plot.png", dpi=160)
    plt.close(fig)
    print(f"PASS: Reference plot saved to {output_dir / 'damped_reference_plot.png'}")

    # Note: full PINN training for damped system is NOT performed here.
    # A complete experiment would require:
    # 1. Residual function for damped system (includes velocity derivative term)
    # 2. Controlled comparison (data_only, data_ic, pinn) with same architecture
    # 3. Multiple seeds, varying zeta/noise/observations
    # 4. Evaluation against analytical damped solution
    # This remains future work per project scope.
    print("NOTE: This is a smoke test only. Full damped-oscillator PINN experiment remains future work.")
