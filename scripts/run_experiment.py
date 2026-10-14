"""Train and compare data-only, data+initial-condition, and PINN models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from physics_ai.data import sample_observations
from physics_ai.equations import exact_position, oscillator_residual
from physics_ai.models import MLP
from physics_ai.numerical_solver import solve_oscillator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_model(
    mode: str,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    t_max: float,
    omega: float,
    epochs: int,
    seed: int,
    hidden_width: int = 64,
    hidden_layers: int = 3,
    n_collocation: int = 256,
    physics_weight: float = 1.0,
    ic_weight: float = 10.0,
) -> tuple[MLP, list[float]]:
    """Train one model variant; return model and total-loss history."""
    if mode not in {"data_only", "data_ic", "pinn"}:
        raise ValueError(f"Unknown mode: {mode}")
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # Keep the CPU baseline responsive on modest PCs.
    model = MLP(hidden_width=hidden_width, hidden_layers=hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    t_train = torch.tensor(t_obs, dtype=torch.float32).reshape(-1, 1)
    x_train = torch.tensor(x_obs, dtype=torch.float32).reshape(-1, 1)
    history: list[float] = []
    rng = np.random.default_rng(seed + 1000)

    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(t_train)
        data_loss = mse(prediction, x_train)
        loss = data_loss

        if mode in {"data_ic", "pinn"}:
            t0 = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
            x0_prediction = model(t0)
            v0_prediction = torch.autograd.grad(
                x0_prediction.sum(), t0, create_graph=True
            )[0]
            initial_loss = (x0_prediction - 1.0).square().mean() + v0_prediction.square().mean()
            loss = loss + ic_weight * initial_loss
        if mode == "pinn":
            # Resample collocation points per epoch so the model sees the whole domain.
            collocation_np = rng.uniform(0.0, t_max, size=n_collocation).astype(np.float32)
            t_collocation = torch.tensor(collocation_np).reshape(-1, 1).requires_grad_(True)
            x_collocation = model(t_collocation)
            residual = oscillator_residual(t_collocation, x_collocation, omega=omega)
            physics_loss = residual.square().mean()
            loss = loss + physics_weight * physics_loss

        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu().item()))

    return model, history


def residual_rmse(model: nn.Module, times: np.ndarray, omega: float) -> float:
    t = torch.tensor(times, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    x = model(t)
    residual = oscillator_residual(t, x, omega=omega)
    return float(torch.sqrt(torch.mean(residual.detach() ** 2)).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-observations", type=int, default=25)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--ic-weight", type=float, default=10.0)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.omega <= 0:
        parser.error("--omega must be positive")
    if args.physics_weight < 0 or args.ic_weight < 0:
        parser.error("Loss weights must be non-negative")

    t_max = 2.0 * np.pi / args.omega
    t_obs, x_obs, _x_clean = sample_observations(
        n_observations=args.n_observations,
        noise_std=args.noise_std,
        seed=args.seed,
        t_min=0.0,
        t_max=t_max,
        omega=args.omega,
    )
    t_grid = np.linspace(0.0, t_max, 1000)
    x_exact = exact_position(t_grid, omega=args.omega)
    t_reference, x_numerical = solve_oscillator(t_grid, omega=args.omega)
    solver_error = float(np.max(np.abs(x_exact - x_numerical)))

    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    modes = ["data_only", "data_ic", "pinn"]
    models: dict[str, MLP] = {}
    histories: dict[str, list[float]] = {}
    rows: list[dict[str, float | int | str]] = []
    predictions: dict[str, np.ndarray] = {
        "time": t_grid,
        "exact_position": x_exact,
        "numerical_position": x_numerical,
    }

    for mode in modes:
        print(f"Training {mode} model for {args.epochs} epochs...")
        start = time.perf_counter()
        model, history = train_model(
            mode=mode,
            t_obs=t_obs,
            x_obs=x_obs,
            t_max=t_max,
            omega=args.omega,
            epochs=args.epochs,
            seed=args.seed,
            physics_weight=args.physics_weight,
            ic_weight=args.ic_weight,
        )
        elapsed = time.perf_counter() - start
        with torch.no_grad():
            t_eval = torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)
            predicted = model(t_eval).cpu().numpy().reshape(-1)
        error = predicted - x_exact
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        phys_rmse = residual_rmse(model, t_grid, args.omega)
        models[mode] = model
        histories[mode] = history
        predictions[f"{mode}_prediction"] = predicted
        rows.append({
            "model": mode,
            "n_observations": args.n_observations,
            "noise_std": args.noise_std,
            "epochs": args.epochs,
            "seed": args.seed,
            "physics_weight": args.physics_weight,
            "ic_weight": args.ic_weight,
            "position_rmse": rmse,
            "position_mae": mae,
            "physics_residual_rmse": phys_rmse,
            "training_seconds": round(elapsed, 3),
            "reference_solver_max_abs_error": solver_error,
        })
        print(f"  position RMSE={rmse:.6f}; physics residual RMSE={phys_rmse:.6f}; {elapsed:.1f}s")

    pd.DataFrame(rows).to_csv(reports_dir / "results.csv", index=False)
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(reports_dir / "predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(t_grid, x_exact, label="Analytical reference", linewidth=2.5)
    ax.scatter(t_obs, x_obs, label="Noisy observations", s=28, alpha=0.8)
    for mode in modes:
        ax.plot(t_grid, predictions[f"{mode}_prediction"], label=mode.replace("_", " "), alpha=0.9)
    ax.set(title="Harmonic oscillator: model predictions", xlabel="Time t", ylabel="Position x(t)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "predictions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for mode, history in histories.items():
        ax.plot(history, label=mode.replace("_", " "))
    ax.set(title="Training loss by model", xlabel="Epoch", ylabel="Total training loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "training_losses.png", dpi=160)
    plt.close(fig)

    print("\nResults saved:")
    print(f"  {reports_dir / 'results.csv'}")
    print(f"  {reports_dir / 'predictions.csv'}")
    print(f"  {figures_dir / 'predictions.png'}")
    print(f"  {figures_dir / 'training_losses.png'}")
    print("\nInterpret model differences using repeated seeds and controlled experiments before drawing conclusions.")


if __name__ == "__main__":
    main()
