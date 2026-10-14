---
name: audit-reproducibility-protocol
description: Physics-informed AI audit completed; reproducibility pipeline preserved, seed-21 outlier retained, 9 extended tests passing.
metadata:
  type: project
---
The audit (commit 8b92fe1) verified: equation agreement <1e-6 with analytical cos(t), baseline reproduced exactly (seed 42), seed-21 outlier (RMSE=0.0797 at physics_weight=10) preserved in aggregates, diagnostic scripts added, all historical CSV outputs backed up to reports/experiment_logs/. No destructive git actions.

**Why:** Keeps the reproducibility protocol durable across sessions.

**How to apply:** When asked about repo state or limitations, reference this protocol and point to reports/reproducibility_audit.md.
