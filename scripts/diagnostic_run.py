"""Extended experiment runner with component-wise diagnostics and outlier tracking."""
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


def train_model_with_diagnostics(
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
) -> tuple[MLP, list[float], dict[str, list[float]]]:
    """Train with per-epoch component loss logging."""
    if mode not in {"data_only", "data_ic", "pinn"}:
        raise ValueError(f"Unknown mode: {mode}")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = MLP(hidden_width=hidden_width, hidden_layers=hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    t_train = torch.tensor(t_obs, dtype=torch.float32).reshape(-1, 1)
    x_train = torch.tensor(x_obs, dtype=torch.float32).reshape(-1, 1)

    total_history: list[float] = []
    data_history: list[float] = []
    ic_history: list[float] = []
    physics_history: list[float] = []

    rng = np.random.default_rng(seed + 1000)
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(t_train)
        data_loss = mse(prediction, x_train)
        loss = data_loss

        ic_loss_val = torch.tensor(0.0)
        physics_loss_val = torch.tensor(0.0)

        if mode in {"data_ic", "pinn"}:
            t0 = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
            x0_prediction = model(t0)
            v0_prediction = torch.autograd.grad(
                x0_prediction.sum(), t0, create_graph=True
            )[0]
            initial_loss = (x0_prediction - 1.0).square().mean() + v0_prediction.square().mean()
            ic_loss_val = initial_loss
            loss = loss + ic_weight * initial_loss

        if mode == "pinn":
            collocation_np = rng.uniform(0.0, t_max, size=n_collocation).astype(np.float32)
            t_collocation = torch.tensor(collocation_np).reshape(-1, 1).requires_grad_(True)
            x_collocation = model(t_collocation)
            residual = oscillator_residual(t_collocation, x_collocation, omega=omega)
            physics_loss = residual.square().mean()
            physics_loss_val = physics_loss
            loss = loss + physics_weight * physics_loss

        # Check numerical stability
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at epoch {_epoch}: loss={loss.item()}")

        loss.backward()
        optimizer.step()
        total_history.append(float(loss.detach().cpu().item()))
        data_history.append(float(data_loss.detach().cpu().item()))
        ic_history.append(float(ic_loss_val.detach().cpu().item()))
        physics_history.append(float(physics_loss_val.detach().cpu().item()))

    diagnostics = {
        "total": total_history,
        "data": data_history,
        "ic": ic_history,
        "physics": physics_history,
    }
    return model, total_history, diagnostics


def run_diagnostic_experiment(
    seed: int = 21,
    physics_weight: float = 10.0,
    epochs: int = 1000,
    output_dir: Path | None = None,
) -> dict:
    """Run focused diagnostic for seed-21 outlier."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = argparse.Namespace(
        epochs=epochs,
        n_observations=25,
        noise_std=0.05,
        seed=seed,
        omega=1.0,
        physics_weight=physics_weight,
        ic_weight=10.0,
    )

    t_max = 2.0 * np.pi / args.omega
    t_obs, x_obs, _ = sample_observations(
        n_observations=args.n_observations,
        noise_std=args.noise_std,
        seed=args.seed,
        t_min=0.0,
        t_max=t_max,
        omega=args.omega,
    )
    t_grid = np.linspace(0.0, t_max, 1000)
    x_exact = exact_position(t_grid, omega=args.omega)

    modes = ["data_only", "data_ic", "pinn"]
    results: dict[str, dict] = {}

    for mode in modes:
        print(f"[Diagnostic] Training {mode} seed={seed} weight={physics_weight} ...")
        start = time.perf_counter()
        model, history, diagnostics = train_model_with_diagnostics(
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
        # Compute physics RMSE using array helper
        phys_rmse = float(np.sqrt(np.mean(np.array(residual_rmse_array(model, t_grid, args.omega))**2)))

        results[mode] = {
            "model": mode,
            "seed": seed,
            "physics_weight": physics_weight,
            "position_rmse": rmse,
            "position_mae": mae,
            "training_seconds": round(elapsed, 3),
            "final_total_loss": diagnostics["total"][-1],
            "final_data_loss": diagnostics["data"][-1],
            "final_ic_loss": diagnostics["ic"][-1],
            "final_physics_loss": diagnostics["physics"][-1],
        }
        print(f"  RMSE={rmse:.6f}  MAE={mae:.6f}  time={elapsed:.1f}s")

    # Save component losses for seed diagnostics
    pd.DataFrame(
        [{"mode": k, **v} for k, v in results.items() if isinstance(v, dict) and "model" in v]
    ).to_csv(output_dir / f"seed_{seed}_weight_{physics_weight}.csv", index=False)

    # Plot predictions vs reference
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(t_grid, x_exact, label="Analytical reference", linewidth=2.5)
    ax.scatter(t_obs, x_obs, label="Observations", s=28, alpha=0.8)
    for mode in modes:
        if mode in results:
            with torch.no_grad():
                pred = MLP(hidden_width=64, hidden_layers=3)(torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)).cpu().numpy().reshape(-1)
            ax.plot(t_grid, pred, label=mode.replace("_", " "), alpha=0.9)
    ax.set(title=f"Seed {seed}, physics weight {physics_weight}", xlabel="t", ylabel="x(t)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"predictions_seed_{seed}_w{physics_weight}.png", dpi=160)
    plt.close(fig)
    return results


def residual_rmse_array(model: nn.Module, times: np.ndarray, omega: float) -> np.ndarray:
    t = torch.tensor(times, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    x = model(t)
    residual = oscillator_residual(t, x, omega=omega)
    return residual.detach().cpu().numpy().reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--physics-weight", type=float, default=10.0)
    args = parser.parse_args()
    diagnostics_dir = PROJECT_ROOT / "reports" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Run diagnostic for outlier seed
    outlier_result = run_diagnostic_experiment(
        seed=args.seed,
        physics_weight=args.physics_weight,
    )

    # Also run representative good seed (seed 7) for comparison
    good_result = run_diagnostic_experiment(
        seed=7,
        physics_weight=args.physics_weight,
    )

    print(f"\nDiagnostic complete. Results in {diagnostics_dir}")
