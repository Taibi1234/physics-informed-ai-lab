# Damped Harmonic Oscillator Study

## Status

Completed controlled experiment with five seeds, three model variants,
analytical and numerical verification, raw predictions, aggregate metrics,
and generated figures.

## Research question

How do data-only, initial-condition-constrained, and physics-informed neural
networks compare on an underdamped harmonic oscillator with sparse noisy
observations?

## Governing equation

```text
x''(t) + 2*zeta*omega*x'(t) + omega^2*x(t) = 0
x(0) = 1
x'(0) = 0
0 &lt; zeta &lt; 1