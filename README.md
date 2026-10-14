# Physics-Informed AI Lab

A reproducible educational research experiment comparing data-driven and physics-informed neural networks on the harmonic oscillator.

## Quick start (Windows PowerShell)

```powershell
# Activate or use directly:
.\.venv\Scripts\python.exe -m pytest tests/
.\.venv\Scripts\python.exe .\scripts\run_experiment.py --epochs 1000 --n-observations 25 --noise-std 0.05 --seed 42
```

## Mathematical problem

Undamped harmonic oscillator (`omega=1`): `x''(t) + x(t) = 0`, `x(0)=1`, `x'(0)=0`.
Analytical solution: `x(t) = cos(t)`. Numerical reference: `scipy.integrate.solve_ivp` (verified agreement `<1e-6`).

## Architecture

Single MLP (`hidden_width=64`, `hidden_layers=3`, `tanh`). Same architecture for all three model categories (`data_only`, `data_ic`, `pinn`).

## Directory structure

```
physics-informed-ai/
├── README.md                 # This file
├── pyproject.toml             # Dependencies and build config
├── src/physics_ai/            # Core modules (equations, data, models, numerical_solver)
├── scripts/
│   ├── run_experiment.py      # Main experiment entry point
│   ├── diagnostic_run.py      # Component-wise diagnostic for outlier seeds
│   ├── reproducibility_experiment.py  # Multi-seed reproducibility pipeline
│   └── generate_summary_figures.py
├── tests/                     # Automated tests (equations, models, extended)
├── configs/                   # Reproducible experiment configurations (JSON)
├── reports/
│   ├── results.csv
│   ├── predictions.csv
│   ├── figures/
│   ├── diagnostics/           # Seed-21 outlier investigation
│   ├── weight_sweep/          # Historical weight-sensitivity results (preserved)
│   └── experiment_logs/       # Backups of original CSV outputs
```

## Reproducing key experiments

**Baseline:**
```powershell
.\.venv\Scripts\python.exe .\scripts\run_experiment.py --epochs 1000 --seed 42
```

**Weight sweep (historical, preserved):**
Results already in `reports/weight_sweep/all_weight_results.csv`.

**Seed-21 outlier diagnostic:**
```powershell
.\.venv\Scripts\python.exe .\scripts\diagnostic_run.py --seed 21 --physics-weight 10.0
```

**Reproducibility across seeds:**
```powershell
.\.venv\Scripts\python.exe .\scripts\reproducibility_experiment.py --seeds 1 7 21 42 123 --physics-weight 10.0
```

**Tests:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

## Key verified results

- Governing equation verified; numerical reference agrees with analytical (`cos t`) within `~1e-9`.
- Existing baseline (`seed 42`) reproduced: `data_only RMSE=0.0258`, `data_ic RMSE=0.0445`, `pinn RMSE=0.0367`.
- Weight sweep (`0.1`, `1.0`, `10.0`) preserved; PINN mean position RMSE: `0.028`, `0.055`, `0.024` (mean ± std over 5 seeds).
- Seed 21 (`physics_weight=10.0`) is a genuine optimization outlier (`RMSE=0.080`); it is retained in all aggregates. Diagnostic reports and prediction plots are in `reports/diagnostics/`.

## Limitations

- Small synthetic problem (`n=25` observations) — conclusions do not generalize to complex real-world dynamics.
- Only 5 random seeds per condition; no formal statistical significance testing performed.
- CPU-only; no GPU determinism verification.
- No adaptive sampling or gradient clipping added; optimization stability relies solely on Adam (`lr=1e-3`).

## References

This is an educational portfolio project. No external publication claims are made. The governing equation and PINN formulation are standard in physics-informed machine learning literature (see, e.g., Raissi et al., 2019, for PINN concepts).
