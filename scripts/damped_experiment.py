"""Full damped harmonic oscillator PINN experiment."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from physics_ai.data import sample_observations
from physics_ai.equations import exact_damped_position, damped_oscillator_residual
from physics_ai.models import MLP

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    if mode not in {"data_only", "data_ic", "pinn"}:
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
            t0 = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
            x0_pred = model(t0)
            v0_pred = torch.autograd.grad(x0_pred.sum(), t0, create_graph=True)[0]
            ic_loss = (x0_pred - 1.0).square().mean() + v0_pred.square().mean()
            loss = loss + ic_weight * ic_loss

        if mode == "pinn":
            collocation_np = rng.uniform(0.0, t_max, size=n_collocation).astype(np.float32)
            t_coll = torch.tensor(collocation_np).reshape(-1, 1).requires_grad_(True)
            x_coll = model(t_coll)
            residual = damped_oscillator_residual(t_coll, x_coll, omega=omega, zeta=zeta)
            physics_loss = residual.square().mean()
            loss = loss + physics_weight * physics_loss

        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu().item()))
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-observations", type=int, default=25)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=0.2)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    args = parser.parse_args()

    t_max = 4 * np.pi  # longer window for damped decay
    t_obs, x_obs, _ = sample_observations(
        n_observations=args.n_observations,
        noise_std=args.noise_std,
        seed=args.seed,
        t_min=0.0,
        t_max=t_max,
        omega=args.omega,
    )
    t_grid = np.linspace(0.0, t_max, 1000)
    x_exact = exact_damped_position(t_grid, omega=args.omega, zeta=args.zeta)

    reports_dir = PROJECT_ROOT / "reports" / "advanced_damped_oscillator"
    reports_dir.mkdir(parents=True, exist_ok=True)

    modes = ["data_only", "data_ic", "pinn"]
    rows = []
    for mode in modes:
        model, history = train_damped_model(
            mode, t_obs, x_obs, t_max, args.omega, args.zeta,
            args.epochs, args.seed,
            physics_weight=args.physics_weight,
        )
        with torch.no_grad():
            pred = model(torch.tensor(t_grid, dtype=torch.float32).reshape(-1, 1)).cpu().numpy().reshape(-1)
        rmse = float(np.sqrt(np.mean((pred - x_exact)**2)))
        rows.append({"mode": mode, "zeta": args.zeta, "rmse": rmse, "seed": args.seed})
        print(f"{mode}: RMSE={rmse:.6f}")

    pd.DataFrame(rows).to_csv(reports_dir / "damped_results.csv", index=False)
    print(f"Saved to {reports_dir / 'damped_results.csv'}")


if __name__ == "__main__":
    main()
