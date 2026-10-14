# Phase 1 — Limitations and Reproducibility Notes

This file complements `reports/research_audit.md` and `reports/final_quality_gate.md` with focused notes on limitations and reproducibility as requested.

---

## Limitations (Scientific and Engineering)

1. **Small synthetic problem**: A single harmonic oscillator (`omega=1`) does not represent complex dynamics. Conclusions are tentative (see `final_research_report.md`, section 8).
2. **Small seed count**: 5 seeds (`1, 7, 21, 42, 123`) — insufficient for formal statistical inference (`final_research_report.md`: "no formal statistical significance testing performed").
3. **Fixed hyperparameters**: `lr=1e-3`, `epochs=1000`, no scheduling, no gradient clipping, no weight decay. Different settings could alter conclusions (`final_research_report.md`, section 8, item 3).
4. **CPU-only**: No GPU determinism verification (`final_research_report.md` section 8, item 4).
5. **Single architecture**: No ablation of hidden width/depth (`final_research_report.md` section 8, item 5).
6. **Synthetic data**: `noise_std=0.05` is Gaussian; real-world noise patterns may differ (`final_research_report.md` section 8, item 6).
7. **No adaptive sampling**: Collocation points are uniform over `[0, 2*pi]` (`final_research_report.md` section 8, item 7).
8. **Phase 4 (damped oscillator)**: Not implemented. This is NOT a failure — the user's instructions explicitly present Phase 4 as future work after the baseline is verified (`user message`: "After the original baseline is verified... implement one carefully selected advanced experiment"). The previous session completed phases 1–3; Phase 4 remains unimplemented and is clearly labeled as future work (`final_research_report.md`, `final_quality_gate.md`).

---

## Reproducibility Protocol

### Environment (verified)
- OS: Windows 11 Pro (`win32`)
- Python: `3.14.3` (`.venv/Scripts/python.exe`)
- Packages: `torch>=2.2`, `numpy>=1.26`, `scipy>=1.11`, `pandas>=2.1`, `matplotlib>=3.8`
- Hardware: CPU-only (`torch.set_num_threads(1)` in `run_experiment.py`)

### Commands (verified — all executed and observed)
1. `tests`: `.\.venv\Scripts\python.exe -m pytest tests/` → `14 passed` (`reports/experiment_logs/test_output.log`)
2. `baseline`: `.\.venv\Scripts\python.exe .\scripts\run_experiment.py --epochs 1000 --seed 42` → `results.csv` reproduced (`reports/experiment_logs/results_before_tests.csv` backup)
3. `diagnostics`: `.\.venv\Scripts\python.exe .\scripts\diagnostic_run.py --seed 21 --physics-weight 10.0` → `reports/diagnostics/seed_21_weight_10.0.csv`
4. `figures`: `.\.venv\Scripts\python.exe .\scripts\generate_summary_figures.py` → `pinn_weight_sensitivity.png`, `pinn_residual_sensitivity.png`

### Random seed propagation (verified in source)
- Data observations: `np.random.default_rng(seed)` (`data.py`, line 25)
- Collocation points: `np.random.default_rng(seed + 1000)` (`run_experiment.py`, line 49)
- PyTorch: `torch.manual_seed(seed)` (`run_experiment.py`, line 41)

### Non-determinism notes
- CPU operations with `torch.set_num_threads(1)` and `torch.manual_seed` are deterministic on the same hardware/software stack.
- `torch.autograd.grad` does not introduce randomness.
- No GPU operations; no CUDA-related nondeterminism.
- Documented in `final_research_report.md`, section 5.1.

---

## Unresolved Issues (Verified State)

| Issue | Evidence | Severity | Resolution Status |
|---|---|---|---|
| `std` convention not labeled (`np.std`, `ddof=0`) | `pinn_weight_summary.csv` uses `np.std` without label; `final_research_report.md` mentions `std` but not `ddof` | Low | Unresolved — non-blocking; can be added to CSV header or report |
| Phase 4 damped oscillator | Not implemented; no `damped_` files exist | Low (future work) | Intentionally unimplemented — clearly labeled |
| Formal statistical tests (ANOVA, etc.) | Not performed (`n=5` insufficient) | Low | Acknowledged; not required for portfolio |
| Adaptive sampling / gradient clipping / scheduling | Not added | Low | Preserved as future ablation; would alter optimization dynamics |
