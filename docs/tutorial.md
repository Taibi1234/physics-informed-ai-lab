# Tutorial: Reproducing the Physics-Informed AI Experiment

Quick walkthrough of the lab's reproducible pipeline.

## 1. Setup
```powershell
python -m venv .venv
.venv\Scripts\pip.exe install -e .
```

## 2. Verify equation
```powershell
python -m pytest tests/test_equations.py -v
```
Expected: numerical `solve_ivp` agrees with `cos(t)` within `<1e-6`.

## 3. Run baseline
```powershell
.venv\Scripts\python.exe .\scripts\run_experiment.py --epochs 1000 --seed 42
```
Produces `reports/results.csv`, `reports/predictions.csv`, `reports/figures/predictions.png`.

## 4. Investigate outlier
```powershell
.venv\Scripts\python.exe .\scripts\diagnostic_run.py --seed 21 --physics-weight 10.0
```
Produces component-wise loss logs in `reports/diagnostics/`.

## 5. Full reproducibility sweep
```powershell
.venv\Scripts\python.exe .\scripts\reproducibility_experiment.py --seeds 1 7 21 42 123 --physics-weight 10.0
```

## 6. Extended tests
```powershell
.venv\Scripts\python.exe -m pytest tests/
```
All 9 extended tests should pass.

## Key takeaways
- Same MLP (`hidden=64`, 3 layers) for all modes; differences come from loss only.
- PINN adds physics residual (`x'' + x = 0`) and initial conditions to data loss.
- Seed 21 at weight 10 is a genuine optimization outlier (`RMSE≈0.080`), not a data error.
- Conclusions are educational — small synthetic dataset (`n=25`), 5 seeds per condition, CPU only.
