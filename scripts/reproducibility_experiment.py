"""Run the undamped PINN experiment across multiple seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physics_ai.data import sample_observations
from physics_ai.equations import exact_position
from scripts.run_experiment import residual_rmse, train_model


def run_single(
    mode: str,
    seed: int,
    physics_weight: float = 1.0,
    ic_weight: float = 10.0,
    epochs: int = 1000,
    n_observations: int = 25,
    noise_std: float = 0.05,
) -> dict:
    """Train and evaluate one model using the main experiment implementation."""
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

    model, _ = train_model(
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
        t_eval = torch.tensor(
            t_grid,
            dtype=torch.float32,
        ).reshape(-1, 1)
        predicted = model(t_eval).cpu().numpy().reshape(-1)

    error = predicted - x_exact

    return {
        "mode": mode,
        "seed": seed,
        "physics_weight": physics_weight,
        "ic_weight": ic_weight,
        "position_rmse": float(np.sqrt(np.mean(error**2))),
        "position_mae": float(np.mean(np.abs(error))),
        "physics_residual_rmse": residual_rmse(
            model,
            t_grid,
            omega=1.0,
        ),
        "epochs": epochs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the undamped PINN experiment across seeds."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 7, 21, 42, 123],
    )
    parser.add_argument(
        "--physics-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/reproducibility_seeds.csv"),
    )
    args = parser.parse_args()

    if not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be non-negative integers")
    if args.physics_weight < 0:
        parser.error("physics-weight cannot be negative")

    rows = []

    for seed in args.seeds:
        print(
            f"Running pinn seed={seed} "
            f"weight={args.physics_weight} ..."
        )
        result = run_single(
            mode="pinn",
            seed=seed,
            physics_weight=args.physics_weight,
        )
        rows.append(result)
        print(
            f"  RMSE={result['position_rmse']:.6f} "
            f"Phys={result['physics_residual_rmse']:.6f}"
        )

    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()