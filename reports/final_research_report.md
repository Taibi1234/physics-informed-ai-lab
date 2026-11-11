# Physics-Informed AI Lab: Final Research Report

**Author:** Nourddine Taibi  
**Updated:** 2026-11-11  
**Project type:** Reproducible scientific machine-learning study

## 1. Abstract

This project compares data-driven, initial-condition-constrained, and
physics-informed neural networks on controlled harmonic-oscillator problems.
The study combines analytical and numerical reference solutions, seeded
synthetic observations, repeated training runs, physics-weight sensitivity
analysis, optimization-outlier diagnostics, and an advanced underdamped
oscillator experiment.

In the undamped study, stronger physics weighting reduced equation residuals
but did not produce a single setting that dominated both prediction accuracy
and physical consistency. In the five-seed damped study, `data_only` achieved
the lowest mean position RMSE (`0.0762`), while `pinn` achieved the lowest
mean physics-residual RMSE (`0.0780`). PINN therefore improved equation
satisfaction without achieving the best position-prediction accuracy.

The results illustrate trade-offs between data fit, initial-condition
enforcement, equation residuals, and optimization variability. They are
small-scale descriptive findings, not evidence of universal PINN superiority
or statistical significance.

## 2. Research questions

1. Does adding initial-condition and differential-equation constraints improve
   prediction accuracy relative to data-only training?
2. How does physics-loss weighting affect position error, equation residual,
   and optimization variability?
3. Can the same controlled comparison be extended from an undamped oscillator
   to an underdamped oscillator with a velocity-dependent residual?
4. Are the experiments reproducible across documented seeds and preserved
   artifacts?

## 3. Mathematical formulation

### 3.1 Undamped oscillator

```text
x''(t) + omega^2*x(t) = 0
x(0) = 1
x'(0) = 0