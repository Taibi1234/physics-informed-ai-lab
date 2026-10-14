# Project Audit — Physics-Informed AI Lab

Date: 2026-10-14
Auditor: Claude (automated assistant)

## Repository status
- Not a Git repository (no `.git`). Previous uncommitted work noted.
- `.gitignore` present; `.venv/`, `.pytest_cache/`, `__pycache__/` excluded correctly.

## Source modules inspected
- `src/physics_ai/equations.py`: Governing equation `x'' + omega^2 x = 0`, analytical solution `cos(omega t)`. Residual uses `torch.autograd.grad` correctly with `create_graph=True`, `retain_graph=True`.
- `src/physics_ai/data.py`: Uniform observation sampling with controlled noise and reproducible `np.random.default_rng`.
- `src/physics_ai/models.py`: Small 3-layer MLP, `tanh` activation, scalar input/output.
- `src/physics_ai/numerical_solver.py`: `scipy.integrate.solve_ivp` with tight tolerances (`rtol=1e-9`, `atol=1e-11`). Matches analytical solution within ~1e-9 (verified by existing test).
- `scripts/run_experiment.py`: Trains three modes (`data_only`, `data_ic`, `pinn`). Uses same architecture, same observation set, same evaluation grid (`t_grid`), consistent seed propagation.
- `tests/`: 5 tests pass; coverage is minimal (no loss-composition, no metrics validation, no reproducibility tests, no file-writing tests).

## Existing results verified
- `reports/results.csv`: preserved (seed 123, physics_weight 10.0, ic_weight 10.0).
- `reports/weight_sweep/`: `all_weight_results.csv` (45 runs), `pinn_weight_summary.csv` (mean/std per weight), individual `physics_*_seed_*.csv` files preserved.
- `reports/figures/`: `predictions.png` and `training_losses.png` present.
- `reports/seed_sweep/seed_sweep.csv` exists (5 seeds).
- Historical results preserved; no destructive overwrite planned.

## Mathematical verification
- Governing equation: `x'' + omega^2 x = 0` with `x(0)=1`, `x'(0)=0`.
- Analytical: `cos(t)` for `omega=1`.
- Numerical solver verified against analytical (`max abs error < 1e-6`).
- Residual definition matches equation directly (`x_tt + omega^2 x`).
- Initial-condition loss uses `(x(0)-1)^2 + (v(0)-0)^2` with `ic_weight`.
- Physics loss uses `mean(residual^2)` with `physics_weight`.
- Loss components numerically comparable only if weights are considered; absolute magnitudes differ by scale.

## Implementation bugs / weaknesses
1. No `training.py`; training logic is embedded in `run_experiment.py`. Acceptable but less modular.
2. `run_experiment.py` records only `total_loss` history; component losses (data, IC, physics) are not logged per epoch. This limits diagnostic capability (critical for seed-21 outlier).
3. No gradient-clipping, no learning-rate scheduling; optimization stability relies entirely on Adam at fixed `lr=1e-3`.
4. No check for numerical divergence (NaN/Inf) during training.
5. No automated test for loss composition, metric calculations, or file-writing/reproducibility.
6. `models.py` predicts only position; velocity is derived via autograd only for IC and physics losses, not stored separately. This is fine, but it means `data_ic` and `data_only` share the same parameter space as PINN (same architecture) — fairness is maintained.
7. Random seed propagation: `torch.manual_seed(seed)` and `np.random.default_rng(seed + 1000)` for collocation. No `torch.cuda.manual_seed_all` needed (CPU-only).
8. No version pinning beyond `pyproject.toml` ranges; reproducibility note already present in README.

## Experimental design assessment
- Model categories (`data_only`, `data_ic`, `pinn`) are genuinely distinct and correctly implemented.
- Same MLP architecture used across all modes (fair comparison).
- Same evaluation grid (`1000` points over `[0, 2*pi]`) for RMSE and residual.
- Same observation count (`25`), noise (`0.05`), epochs (`1000`), seeds (`1, 7, 21, 42, 123`) across weight sweep.
- Weight sweep covers `0.1`, `1.0`, `10.0` (logarithmically spaced-ish; could include `0.01` or `100` for broader sensitivity).
- Seed 21 is a genuine outlier at `physics_weight=10.0` (position RMSE ~0.080 vs. mean ~0.024). It must not be removed.

## Recommendations for next phases
- Add component-wise loss logging.
- Add numerical-divergence checks.
- Add more automated tests (metrics, file I/O, reproducibility, loss composition).
- Investigate seed-21 outlier with diagnostics (training curves, predictions, residual plots).
- Preserve all historical CSVs; add new results to versioned files (`reports/diagnostics/`).
