"""Run a reproducibility experiment with multiple seeds."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse the core experiment logic
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physics_ai.data import sample_observations
from physics_ai.equations import exact_position, oscillator_residual
from physics_ai.models import MLP
from physics_ai.numerical_solver import solve_oscillator

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_single(
    mode: str,
    seed: int,
    physics_weight: float = 1.0,
    ic_weight: float = 10.0,
    epochs: int = 1000,
    n_observations: int = 25,
    noise_std: float = 0.05,
) -> dict:
    """Simplified single-run extraction from run_experiment logic."""
    # Import training logic inline to avoid duplicating the full script
    from scripts.run_experiment import train_model, residual_rmse
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    t_max = 2.0 * np.pi
    t_obs, x_obs, _ = sample_observations(
        n_observations=n_observations,
        noise_std=noise_std,
        seed=seed,
        t_min=0.0,
        t_max=t_max,
        omega=1.0,
    )
    t_grid = np.linspace(0.0, t_max, 1000)
    x_exact = exact_position(t_grid, omega=1.0)

    model, history = train_model(
        mode=mode,
        t_obs=t_obs,
        x_obs=x_obs,
        t_max=t_max,
        omega=1.0,
        epochs=epochs,
        seed=seed,
        physics_weight=physics_weight,
        ic_weight=ic_weight,
    )
    with torch.no_grad():
        t_eval = torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)
        predicted = model(t_eval).cpu().numpy().reshape(-1)
    rmse = float(np.sqrt(np.mean((predicted - x_exact)**2)))
    mae = float(np.mean(np.abs(predicted - x_exact)))
    phys_rmse = residual_rmse(model, t_grid, omega=1.0)
    return {
        "mode": mode,
        "seed": seed,
        "physics_weight": physics_weight,
        "ic_weight": ic_weight,
        "position_rmse": rmse,
        "position_mae": mae,
        "physics_residual_rmse": phys_rmse,
        "epochs": epochs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 21, 42, 123])
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="reports/reproducibility_seeds.csv")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for mode in ["pinn"]:
            print(f"Running {mode} seed={seed} weight={args.physics_weight} ...")
            result = run_single(
                mode=mode, seed=seed, physics_weight=args.physics_weight
            )
            rows.append(result)
            print(f"  RMSE={result['position_rmse']:.6f} Phys={result['physics_residual_rmse']:.6f}")

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
