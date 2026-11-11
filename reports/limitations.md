# Limitations and Reproducibility Notes

## Scientific limitations

1. **Synthetic problems:** The experiments cover undamped and underdamped
   harmonic oscillators with known analytical solutions. These controlled
   problems do not represent the complexity of real-world physical systems.

2. **Limited seed count:** Each aggregate uses five seeds. Means and standard
   deviations describe the recorded runs but do not establish statistical
   significance.

3. **Fixed architecture:** All variants use a three-layer, 64-unit Tanh MLP.
   Architecture changes could alter the results.

4. **Fixed optimization settings:** Experiments use Adam with learning rate
   0.001 and 1,000 epochs. Schedules, gradient clipping, other optimizers, and
   convergence tolerances were not studied.

5. **Synthetic noise:** Observations use independent Gaussian noise with
   standard deviation 0.05. Other noise distributions and correlated
   measurement errors were not evaluated.

6. **Uniform collocation sampling:** PINN collocation points are sampled
   uniformly. Adaptive or residual-guided sampling was not evaluated.

7. **Limited damped-study coverage:** The damped experiment uses one damping
   ratio, one observation count, one noise level, and one physics weight.

8. **Metric distinction:** A lower physics-residual RMSE indicates closer
   equation satisfaction on the evaluation grid, not necessarily lower
   position-prediction error.

9. **Loss comparability:** Final total losses contain different terms across
   model variants and therefore are not directly comparable measures of model
   quality.

10. **No quantum or real-hardware experiments:** All models are conventional
    neural networks running on classical computers.

## Engineering limitations

- Experiments were designed for CPU execution with one PyTorch thread.
- Identical seeds support reproducibility on the same software and hardware
  stack, but bit-for-bit reproducibility across platforms is not guaranteed.
- Dependency ranges are recorded in `pyproject.toml`; exact package versions
  should be recorded when reproducing results in a new environment.
- Runtime measurements can vary with machine load and hardware.

## Reproducibility protocol

### Random seeds

- Observations: `np.random.default_rng(seed)`
- Collocation sampling: `np.random.default_rng(seed + 1000)`
- Model initialization: `torch.manual_seed(seed)`
- Seed set: `1, 7, 21, 42, 123`

### Aggregate statistics

Reported standard deviations use NumPy's population convention:

```python
np.std(values, ddof=0)