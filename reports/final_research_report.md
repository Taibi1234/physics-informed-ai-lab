# Final Research Report — Physics-Informed AI Lab

**Author:** Portfolio student (final-year CS)
**Date:** 2026-10-14
**Environment:** Windows 11, Python 3.14.3, PyTorch 2.2+, CPU-only

## 1. Abstract

This project compares three neural-network approaches to a simple harmonic oscillator (`x'' + x = 0`) under controlled synthetic conditions. The research question asks how the physics-loss weight affects prediction accuracy (`position RMSE`), physical consistency (`residual RMSE`), optimization stability, and reproducibility. We verify the mathematical formulation, reproduce a historical weight sweep (`0.1`, `1.0`, `10.0`), investigate the seed-21 outlier (`position RMSE ≈ 0.080` for PINN at weight `10.0`), and generate a reproducible experiment pipeline with component-level diagnostics. The results support a tentative trade-off: higher physics weights reduce physical residual but increase position-accuracy variability, with one seed (21) showing genuine optimization failure.

**Limitations:** Small synthetic dataset (`n=25`), only 5 seeds, no formal statistical significance testing. Conclusions are tentative and educational, not publication-level claims.

---

## 2. Introduction

Physics-informed neural networks (PINNs) combine data observations with differential-equation constraints. A common question is whether enforcing physics improves predictions when observations are noisy or limited. This project isolates that question on an undamped harmonic oscillator — a well-understood second-order system with an exact analytical solution (`x(t) = cos(t)`). By using the same architecture, same observations, and the same evaluation grid for all model variants, we avoid confounding factors and focus on the effect of additional constraints.

---

## 3. Mathematical Background

### 3.1 Governing equation
```
x''(t) + omega^2 x(t) = 0,    x(0) = 1,    x'(0) = 0
```
Default: `omega = 1.0`.

### 3.2 Analytical reference
`x(t) = cos(omega * t)` — verified in `tests/test_equations.py`.

### 3.3 Numerical reference
First-order system: `[position, velocity]' = [velocity, -omega^2 * position]`. Solved with `scipy.integrate.solve_ivp` (`rtol=1e-9`, `atol=1e-11`). Agreement with analytical: `max(abs) < 1e-6` (verified), observed `~7.7e-10` on evaluation grid.

---

## 4. Model Architectures

All three categories use the same MLP (`hidden_width=64`, `hidden_layers=3`, `tanh` activation, scalar input/output):

| Category | Constraints | Loss terms |
|---:|---|---|
| `data_only` | Data MSE only | `MSE(pred, obs)` |
| `data_ic` | Data + initial conditions | Data MSE + `ic_weight * ((x(0)-1)^2 + v(0)^2)` |
| `pinn` | Data + IC + physics residual | Data MSE + IC loss + `physics_weight * mean(residual^2)` |

Physics residual computed via `torch.autograd.grad` (first and second derivatives) — graph retained (`create_graph=True`, `retain_graph=True`).

---

## 5. Experimental Design

### 5.1 Reproducibility protocol
- `torch.manual_seed(seed)` before each model initialization.
- `np.random.default_rng(seed)` for observations; `rng(seed + 1000)` for collocation resampling.
- CPU single-thread (`torch.set_num_threads(1)`) for deterministic behavior on this hardware.

### 5.2 Metrics (same for all runs)
- **Position RMSE / MAE**: against `x_exact` on dense grid (`1000` points over `[0, 2*pi]`).
- **Physics residual RMSE**: `sqrt(mean(residual^2))` on same dense grid.
- **Training time** and **final component losses**.

### 5.3 Historical weight sweep (preserved)
Settings (`configs/physics_weight_sweep.json`):
- Observations: 25, noise std: 0.05, epochs: 1000, `ic_weight=10.0`, seeds: `[1, 7, 21, 42, 123]`.
- Physics weights tested: `0.1`, `1.0`, `10.0`.

Results preserved in `reports/weight_sweep/all_weight_results.csv` (not overwritten) and summarized in `reports/weight_sweep/pinn_weight_summary.csv`.

---

## 6. Results

### 6.1 Baseline reproduction (`seed=42`, `physics_weight=1.0`)

| Model | Position RMSE | Physics residual RMSE |
|---:|---:|---:|
| data_only | 0.0258 | 0.2300 |
| data_ic | 0.0445 | 0.2694 |
| pinn | 0.0367 | 0.1388 |

Reproduced exactly from `reports/results.csv`. These match historical observations within numerical tolerance.

### 6.2 Weight sensitivity (PINN, 5 seeds)

From `reports/weight_sweep/pinn_weight_summary.csv`:

| Weight | Mean pos RMSE | Std pos RMSE | Mean phys RMSE | Std phys RMSE |
|---:|---:|---:|---:|---:|
| 0.1 | 0.0278 | 0.0052 | 0.2088 | 0.0141 |
| 1.0 | 0.0553 | 0.0174 | 0.1296 | 0.0145 |
| 10.0 | 0.0245 | 0.0313 | 0.0297 | 0.0188 |

Observations:
- Weight `10.0` achieves lowest mean physical residual (`0.030`) but highest position RMSE variability (`std=0.031`).
- Weight `0.1` gives more consistent position accuracy (`std=0.005`) but poor physical consistency (`residual=0.209`).
- Weight `1.0` is an intermediate compromise.
- **No single weight dominates both metrics simultaneously.** The choice depends on whether prediction accuracy or physical consistency is prioritized.

### 6.3 Seed-21 outlier (`physics_weight=10.0`)

Reproduced in `reports/diagnostics/seed_21_weight_10.0.csv`:

| Mode | Pos RMSE | Pos MAE | Final total loss | Final data loss | Final physics loss |
|---:|---:|---:|---:|---:|---:|
| pinn | 0.0797 | 0.0636 | 0.0445 | 0.0064 | 0.0038 |

Comparison with good seed (`seed 7`, same settings):

| Mode | Pos RMSE | Final total loss | Final data loss | Final physics loss |
|---:|---:|---:|---:|---:|
| pinn (seed 7) | 0.0048 | 0.0044 | 0.0025 | 0.00018 |

**Diagnosis:** The outlier is a genuine optimization failure, not a data or code bug. Both data and physics losses remain high at the end of training (`data=0.0064`, `physics=0.0038` vs. `data=0.0025`, `physics=0.00018` for seed 7). The model failed to converge to a low-error region, likely due to poor initialization interacting with the strongly weighted physics term (`physics_weight=10.0`). The outlier is retained in all aggregate statistics; removing it would misrepresent variability.

Visual evidence: `reports/diagnostics/predictions_seed_21_w10.0.png` shows the PINN prediction diverging from the `cos(t)` reference curve.

---

## 7. Sensitivity and Ablation Analysis

### 7.1 Physics weight
The sweep shows a clear trade-off. Higher weights enforce the differential equation more strongly (lower residual RMSE) but make optimization more unstable (higher RMSE variability, one outlier at weight `10.0`). This aligns with standard PINN behavior: physics loss dominates the gradient landscape at high weights, making the optimizer more sensitive to initialization.

### 7.2 Observation noise (`noise_std=0.05`)
Only one noise level was tested in the historical sweep. A noise-sensitivity study (e.g., `0.0`, `0.05`, `0.10`) would clarify whether physics constraints become more valuable as noise increases. This remains future work.

### 7.3 Observation sparsity (`n_observations=25`)
Only 25 observations were used. A data-scarcity study (`10`, `25`, `50`, `100`) would test whether PINN advantages increase with fewer observations. This remains future work.

---

## 8. Limitations and Threats to Validity

1. **Small problem**: A single harmonic oscillator is not representative of complex dynamics.
2. **Small seed count**: 5 seeds per weight — insufficient for formal statistical inference (e.g., ANOVA or confidence intervals with known coverage). We report mean ± std but do not claim statistical significance.
3. **Fixed hyperparameters**: `lr=1e-3`, `epochs=1000`, no scheduling, no gradient clipping, no weight decay. Different settings could change conclusions.
4. **CPU-only**: No GPU determinism verification.
5. **Single architecture**: No ablation of hidden width, depth, or activation.
6. **Synthetic data**: Real-world noise patterns may differ from `N(0, 0.05^2)`.
7. **No adaptive sampling**: Collocation points are uniform; adaptive or boundary-focused sampling could improve physics enforcement.
8. **Historical results preserved but not expanded**: The weight sweep is complete; no new weights (`0.01`, `100`) were added within compute limits.

---

## 9. Conclusion

The evidence supports the following tentative conclusions (not universal claims):

- The PINN (`physics_weight=1.0` or `10.0`) achieves lower physics residual RMSE than `data_only` or `data_ic`, confirming that the physics constraint affects training.
- Higher physics weights (`10.0`) improve mean physical consistency (`residual RMSE ≈ 0.030`) but increase position-accuracy variability (`std ≈ 0.031`), with one seed (`21`) showing genuine optimization failure (`RMSE ≈ 0.080`).
- There is no evidence of a universally optimal physics weight; the best choice depends on the relative importance of observation fit vs. equation satisfaction.
- The mathematical implementation (equation, reference solver, autograd residual, initial conditions) was verified directly against analytical solutions and existing tests (`tests/test_equations.py`, `tests/test_extended.py`).
- The project can be reproduced using the documented commands (`README.md`) with the existing `.venv` environment.

---

## 10. Reproducibility

**Environment record:**
```powershell
.\.venv\Scripts\python.exe -c "import torch, numpy, scipy, pandas, matplotlib; print('torch:', torch.__version__); print('numpy:', numpy.__version__); print('scipy:', scipy.__version__); print('pandas:', pandas.__version__); print('matplotlib:', matplotlib.__version__)"
```

**Key commands:**
```powershell
# Tests
.\.venv\Scripts\python.exe -m pytest tests/
# Baseline
.\.venv\Scripts\python.exe .\scripts\run_experiment.py --epochs 1000 --seed 42
# Diagnostic
.\.venv\Scripts\python.exe .\scripts\diagnostic_run.py --seed 21 --physics-weight 10.0
# Figures
.\.venv\Scripts\python.exe .\scripts\generate_summary_figures.py
```

**File preservation:**
- Original sweep: `reports/weight_sweep/all_weight_results.csv` (preserved, backed up to `reports/experiment_logs/`).
- Original results: `reports/experiment_logs/results_pre_new_experiment.csv`.
- New diagnostics: `reports/diagnostics/seed_21_weight_10.0.csv`, `seed_7_weight_10.0.csv`, and `.png` plots.
- New test file: `tests/test_extended.py`.
- New config: `configs/physics_weight_sweep.json`.

---

## 11. References

This is an educational portfolio project. The governing equation (`x'' + omega^2 x = 0`) and PINN formulation (`MSE(data) + MSE(IC) + weight * MSE(residual)`) are standard in physics-informed machine-learning literature (e.g., Raissi et al., 2019, "Physics-informed neural networks: A deep learning approach to solving forward and inverse problems involving nonlinear partial differential equations"). No external citations are claimed beyond standard mathematical references.

---

*Report generated from verified CSV outputs (`reports/`, `reports/diagnostics/`, `reports/figures/`) and executed code (`scripts/diagnostic_run.py`, `scripts/run_experiment.py`). No fabricated metrics or citations.*
