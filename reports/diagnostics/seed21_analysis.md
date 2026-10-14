# Seed-21 Outlier Investigation — PINN, physics weight 10.0

## Context
- Physics weight: 10.0
- Observations: 25, noise std 0.05
- Seed 21 produced position RMSE ~0.080 (much worse than other seeds: 0.008, 0.005, 0.012, 0.019).
- The outlier was reproduced exactly: `RMSE=0.079679`, `MAE=0.063556`.

## Component losses at end of training (seed 21)

| Mode | Final data loss | Final IC loss | Final physics loss | Final total loss |
|---:|---:|---:|---:|---:|
| data_only | 0.00122 | 0 | 0 | 0.00122 |
| data_ic | 0.00163 | ~1.5e-8 | 0 | 0.00163 |
| pinn | 0.00642 | ~3.7e-5 | 0.00377 | 0.04449 |

## Component losses at end of training (seed 7 — representative good seed)

| Mode | Final data loss | Final IC loss | Final physics loss | Final total loss |
|---:|---:|---:|---:|---:|
| pinn | 0.00254 | ~1.0e-6 | 0.00018 | 0.00438 |

## Interpretation
- For seed 21 PINN, the physics loss (`0.00377`) and data loss (`0.00642`) are both much larger than for seed 7 (`0.00018` and `0.00254`).
- The optimization did not converge to a low-residual, low-data-error state. The model failed to satisfy either data or physics well.
- This is consistent with optimization instability or a poor initialization landing in a region where Adam struggles with the heavily weighted physics loss (`physics_weight=10.0`).
- There is no evidence of a code bug (same script, same architecture, same data generation). The poor result is a genuine sensitivity of PINN training to initialization for this weight.

## Visual inspection
- `reports/diagnostics/predictions_seed_21_w10.0.png` shows PINN prediction diverging significantly from the reference `cos(t)` curve, while other seeds track closely.
- The prediction plot confirms the numerical RMSE.

## Conclusion
The seed-21 outlier is a real optimization failure / poor initialization effect, not a data or implementation error. It demonstrates that a high physics weight (`10.0`) can cause unstable optimization across seeds. Including it in aggregate statistics is scientifically necessary; excluding it would misrepresent variability.
