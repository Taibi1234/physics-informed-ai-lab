"""Generate comparison visualizations from verified weight-sweep results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "weight_sweep"
    / "pinn_weight_summary.csv"
)
FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures"

NUMERIC_COLUMNS = [
    "physics_weight",
    "seeds",
    "position_mean",
    "position_std",
    "position_median",
    "residual_mean",
    "residual_std",
    "residual_median",
]


def main() -> None:
    summary = pd.read_csv(SUMMARY_PATH)

    missing = [
        column
        for column in NUMERIC_COLUMNS
        if column not in summary.columns
    ]
    if missing:
        raise ValueError(f"Missing summary columns: {missing}")
    if summary.empty:
        raise ValueError("The weight-sweep summary contains no results")

    for column in NUMERIC_COLUMNS:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="raise",
        )

    values = summary[NUMERIC_COLUMNS].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Summary metrics must contain only finite numbers"
        )
    if (summary["physics_weight"] <= 0).any():
        raise ValueError("Physics weights must be positive")
    if (
        summary[["position_std", "residual_std"]] < 0
    ).any().any():
        raise ValueError("Standard deviations cannot be negative")

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    weights = summary["physics_weight"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        weights,
        summary["position_mean"],
        yerr=summary["position_std"],
        fmt="o-",
        capsize=4,
        color="#2c3e50",
        label="PINN position RMSE (mean ± population std)",
    )
    ax.set_xlabel("Physics loss weight")
    ax.set_ylabel("Position RMSE")
    ax.set_title(
        "PINN position accuracy vs physics weight (5 seeds)"
    )
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIRECTORY / "pinn_weight_sensitivity.png",
        dpi=160,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        weights,
        summary["residual_mean"],
        yerr=summary["residual_std"],
        fmt="s-",
        capsize=4,
        color="#c0392b",
        label="PINN physics residual RMSE (mean ± population std)",
    )
    ax.set_xlabel("Physics loss weight")
    ax.set_ylabel("Physics residual RMSE")
    ax.set_title(
        "PINN physical consistency vs physics weight (5 seeds)"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIRECTORY / "pinn_residual_sensitivity.png",
        dpi=160,
    )
    plt.close(fig)

    print(f"Figures saved to {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()