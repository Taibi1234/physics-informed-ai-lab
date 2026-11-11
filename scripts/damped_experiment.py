"""Controlled multi-seed comparison on the underdamped harmonic oscillator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from physics_ai.data import sample_damped_observations
from physics_ai.equations import (
    damped_oscillator_residual,
    exact_damped_position,
)
from physics_ai.models import MLP
from physics_ai.numerical_solver import solve_damped_oscillator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("data_only", "data_ic", "pinn")


def train_damped_model(
    mode: str,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    t_max: float,
    omega: float,
    zeta: float,
    epochs: int,
    seed: int,
    hidden_width: int = 64,
    hidden_layers: int = 3,
    n_collocation: int = 256,
    physics_weight: float = 1.0,
    ic_weight: float = 10.0,
) -> tuple[MLP, list[float]]:
    """Train one model variant using shared data and initialization seeds."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = MLP(hidden_width=hidden_width, hidden_layers=hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    t_train = torch.tensor(t_obs, dtype=torch.float32).reshape(-1, 1)
    x_train = torch.tensor(x_obs, dtype=torch.float32).reshape(-1, 1)
    history: list[float] = []
    rng = np.random.default_rng(seed + 1000)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(t_train)
        data_loss = mse(prediction, x_train)
        loss = data_loss

        if mode in {"data_ic", "pinn"}:
            t0 = torch.zeros(
                (1, 1), dtype=torch.float32, requires_grad=True
            )
            x0_pred = model(t0)
            v0_pred = torch.autograd.grad(
                x0_pred.sum(), t0, create_graph=True
            )[0]
            ic_loss = (
                (x0_pred - 1.0).square().mean()
                + v0_pred.square().mean()
            )
            loss = loss + ic_weight * ic_loss

        if mode == "pinn":
            collocation = rng.uniform(
                0.0, t_max, size=n_collocation
            ).astype(np.float32)
            t_coll = torch.tensor(collocation).reshape(-1, 1)
            t_coll.requires_grad_(True)
            x_coll = model(t_coll)
            residual = damped_oscillator_residual(
                t_coll, x_coll, omega=omega, zeta=zeta
            )
            loss = loss + physics_weight * residual.square().mean()

        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu().item()))

    return model, history


def evaluate_damped_model(
    model: MLP,
    t_grid: np.ndarray,
    omega: float,
    zeta: float,
) -> dict[str, float | np.ndarray]:
    """Evaluate position predictions and equation residual on a fixed grid."""
    exact = exact_damped_position(t_grid, omega=omega, zeta=zeta)

    with torch.no_grad():
        predicted = model(
            torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)
        ).cpu().numpy().reshape(-1)

    t_residual = torch.tensor(
        t_grid, dtype=torch.float32
    ).reshape(-1, 1)
    t_residual.requires_grad_(True)
    x_residual = model(t_residual)
    residual = damped_oscillator_residual(
        t_residual, x_residual, omega=omega, zeta=zeta
    )

    error = predicted - exact
    residual_values = residual.detach().cpu().numpy().reshape(-1)

    return {
        "predicted": predicted,
        "exact": exact,
        "position_rmse": float(np.sqrt(np.mean(error**2))),
        "position_mae": float(np.mean(np.abs(error))),
        "physics_residual_rmse": float(
            np.sqrt(np.mean(residual_values**2))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare data-driven and physics-informed models "
            "on a damped oscillator."
        )
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-observations", type=int, default=25)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 7, 21, 42, 123]
    )
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=0.2)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--ic-weight", type=float, default=10.0)
    parser.add_argument("--n-collocation", type=int, default=256)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "reports" / "advanced_damped_oscillator",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.n_collocation < 1:
        parser.error("epochs and n-collocation must be positive")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be non-negative integers")
    if args.physics_weight < 0 or args.ic_weight < 0:
        parser.error("loss weights cannot be negative")

    t_max = 4.0 * np.pi
    t_grid = np.linspace(0.0, t_max, 1000)

    _, numerical_reference = solve_damped_oscillator(
        t_grid, omega=args.omega, zeta=args.zeta
    )
    exact_reference = exact_damped_position(
        t_grid, omega=args.omega, zeta=args.zeta
    )
    reference_error = float(
        np.max(np.abs(numerical_reference - exact_reference))
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for seed in args.seeds:
        t_obs, x_obs, _ = sample_damped_observations(
            n_observations=args.n_observations,
            noise_std=args.noise_std,
            seed=seed,
            t_min=0.0,
            t_max=t_max,
            omega=args.omega,
            zeta=args.zeta,
        )

        for mode in MODES:
            model, history = train_damped_model(
                mode=mode,
                t_obs=t_obs,
                x_obs=x_obs,
                t_max=t_max,
                omega=args.omega,
                zeta=args.zeta,
                epochs=args.epochs,
                seed=seed,
                n_collocation=args.n_collocation,
                physics_weight=args.physics_weight,
                ic_weight=args.ic_weight,
            )
            metrics = evaluate_damped_model(
                model, t_grid, omega=args.omega, zeta=args.zeta
            )

            result_rows.append({
                "seed": seed,
                "mode": mode,
                "position_rmse": metrics["position_rmse"],
                "position_mae": metrics["position_mae"],
                "physics_residual_rmse": (
                    metrics["physics_residual_rmse"]
                ),
                "final_total_loss": history[-1],
            })
            prediction_frames.append(pd.DataFrame({
                "seed": seed,
                "mode": mode,
                "time": t_grid,
                "exact_position": metrics["exact"],
                "predicted_position": metrics["predicted"],
            }))
            print(
                f"seed={seed} {mode}: "
                f"RMSE={metrics['position_rmse']:.6f}"
            )

    results = pd.DataFrame(result_rows)
    results.to_csv(
        args.output_directory / "damped_results.csv", index=False
    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        args.output_directory / "damped_predictions.csv", index=False
    )

    summary = results.groupby("mode").agg(
        runs=("seed", "count"),
        position_rmse_mean=("position_rmse", "mean"),
        position_rmse_std=(
            "position_rmse",
            lambda values: np.std(values, ddof=0),
        ),
        position_mae_mean=("position_mae", "mean"),
        physics_residual_rmse_mean=(
            "physics_residual_rmse",
            "mean",
        ),
        final_total_loss_mean=("final_total_loss", "mean"),
    ).reindex(MODES).reset_index()

    summary.to_csv(
        args.output_directory / "damped_summary.csv", index=False
    )

    configuration = {
        "equation": "x'' + 2*zeta*omega*x' + omega^2*x = 0",
        "initial_conditions": {"x0": 1.0, "v0": 0.0},
        "t_max": t_max,
        "reference_solver_max_abs_error": reference_error,
        "std_convention": "population standard deviation (ddof=0)",
        "settings": {
            key: value
            for key, value in vars(args).items()
            if key != "output_directory"
        },
    }
    (args.output_directory / "damped_config.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        summary["mode"],
        summary["position_rmse_mean"],
        yerr=summary["position_rmse_std"],
        capsize=4,
    )
    ax.set(
        xlabel="Model",
        ylabel="Position RMSE",
        title=(
            f"Damped oscillator: {len(args.seeds)}-seed comparison"
        ),
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        args.output_directory / "damped_model_comparison.png",
        dpi=160,
    )
    plt.close(fig)

    representative_seed = 42 if 42 in args.seeds else args.seeds[0]
    representative = predictions[
        predictions["seed"] == representative_seed
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    reference = representative[representative["mode"] == "data_only"]
    ax.plot(
        reference["time"],
        reference["exact_position"],
        "k--",
        label="Analytical",
    )

    for mode in MODES:
        subset = representative[representative["mode"] == mode]
        ax.plot(
            subset["time"],
            subset["predicted_position"],
            label=mode,
        )

    ax.set(
        xlabel="Time",
        ylabel="Position",
        title=(
            "Damped oscillator predictions "
            f"(seed={representative_seed})"
        ),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.output_directory / "damped_predictions.png",
        dpi=160,
    )
    plt.close(fig)

    print(
        "Reference solver maximum absolute error: "
        f"{reference_error:.3e}"
    )
    print(
        "Saved damped experiment artifacts to "
        f"{args.output_directory}"
    )


if __name__ == "__main__":
    main()