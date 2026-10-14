=== Analytical derivation (verified independently) ===
Damped harmonic oscillator (underdamped):
  x''(t) + 2*zeta*omega*x'(t) + omega^2*x(t) = 0
  x(0) = 1, x'(0) = 0, 0 < zeta < 1

Analytical solution:
  omega_d = omega * sqrt(1 - zeta^2)
  x(t) = exp(-zeta*omega*t) * (cos(omega_d*t) + zeta*omega/omega_d * sin(omega_d*t))

Verified by substituting into ODE and checking x(0)=1, x'(0)=0.
Initial velocity v(0) = -zeta*omega + zeta*omega = 0 (verified algebraically).
