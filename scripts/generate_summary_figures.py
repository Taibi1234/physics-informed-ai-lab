"""Generate comparison visualizations from verified CSV results."""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load historical weight sweep summary
summary = pd.read_csv("reports/weight_sweep/pinn_weight_summary.csv")

# Figure 1: Position RMSE vs physics weight
fig, ax = plt.subplots(figsize=(8, 5))
weights = summary["physics_weight"]
ax.errorbar(weights, summary["position_mean"], yerr=summary["position_std"],
            fmt="o-", capsize=4, color="#2c3e50", label="PINN position RMSE (mean ± std)")
ax.set_xlabel("Physics loss weight")
ax.set_ylabel("Position RMSE")
ax.set_title("PINN position accuracy vs physics weight (5 seeds)")
ax.set_xscale("log")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig("reports/figures/pinn_weight_sensitivity.png", dpi=160)
plt.close(fig)

# Figure 2: Physics residual RMSE vs physics weight
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(weights, summary["residual_mean"], yerr=summary["residual_std"],
            fmt="s-", capsize=4, color="#c0392b", label="PINN physics residual RMSE")
ax.set_xlabel("Physics loss weight")
ax.set_ylabel("Physics residual RMSE")
ax.set_title("PINN physical consistency vs physics weight (5 seeds)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig("reports/figures/pinn_residual_sensitivity.png", dpi=160)
plt.close(fig)

print("Figures saved to reports/figures/")
