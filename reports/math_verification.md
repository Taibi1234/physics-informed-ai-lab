# Mathematical Verification — Physics-Informed AI Lab

## Governing differential equation

The system is the undamped harmonic oscillator with unit mass:

```
x''(t) + omega^2 * x(t) = 0,    x(0) = 1,    x'(0) = 0
```

Default parameter: `omega = 1.0`.

## Analytical reference solution

For `omega = 1.0`: `x(t) = cos(t)`.
Verified in `exact_position()` using `np.cos` / `torch.cos`.

## Numerical reference

`scipy.integrate.solve_ivp` solves the first-order system:
- State: `[position, velocity]`
- RHS: `[velocity, -omega^2 * position]`
- Tolerances: `rtol=1e-9`, `atol=1e-11`

Verified: numerical and analytical agree within `1e-6` on 200-point grid (`test_analytical_and_numerical_solutions_agree`).
Actual solver error observed: `~7.7e-10` (from `results.csv`).

## Residual definition verification

`oscillator_residual(t, x, omega)` computes:
```
d/dt (first_derivative) via autograd + omega^2 * x
```
This is exactly `x_tt + omega^2 * x`.
Verified for exact `cos(t)`: `max(abs(residual)) < 1e-5` (`test_oscillator_residual_for_exact_solution_is_small`).

## Initial conditions

- Position at `t=0`: `x(0) = 1`
- Velocity at `t=0`: `x'(0) = 0`
Initial-condition loss: `(x_pred(0) - 1)^2 + (v_pred(0) - 0)^2`, scaled by `ic_weight` (`10.0` default).

## Data generation

Observations: 25 points, uniform over `[0, 2*pi]`, noise `N(0, 0.05^2)`.
Reproducible via `np.random.default_rng(seed)`.

## Architecture fairness

All three modes (`data_only`, `data_ic`, `pinn`) use identical `MLP(hidden_width=64, hidden_layers=3)`.
Output: scalar position predicted from scalar time.
Velocity is derived via autograd only for IC and physics losses; the network itself predicts only position (same for all modes). This ensures parameter-space fairness.
