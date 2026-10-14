# Model Fairness Verification

## Architecture
- All three modes (`data_only`, `data_ic`, `pinn`) use identical `MLP(hidden_width=64, hidden_layers=3)`.
- Output: scalar position `x(t)` predicted from scalar time `t`.
- Velocity is derived via autograd (`torch.autograd.grad`) for IC and physics calculations; the network never explicitly predicts velocity. This keeps parameter spaces identical across modes (fair comparison).

## Loss definitions
- `data_loss`: `MSE(pred, obs)` — same observations, same scale.
- `ic_loss`: `(pred(0)-1)^2.mean() + (pred'(0)-0)^2.mean()` — same for `data_ic` and `pinn`, scaled by `ic_weight=10.0`.
- `physics_loss`: `mean(residual^2)` — only for `pinn`, scaled by `physics_weight` (`1.0` or `10.0` in existing sweep).

## Scaling considerations
- Data loss operates on noisy observations (`noise_std=0.05`), magnitude ~`0.0025` (variance).
- Physics loss operates on 256 collocation points over full domain; residual magnitude for poor predictions can be `O(1)`, giving physics loss `~O(1)`.
- Thus `physics_weight=10` heavily amplifies physics relative to data; `physics_weight=1` gives more balanced contribution. This explains why weight `10.0` leads to very low residual but sometimes poor observation fit (e.g., seed 21 outlier at `RMSE=0.080`).
- The comparison is fair because all modes share architecture, data, evaluation grid, and seed protocol. Only the additional constraints (IC, physics) differ.
