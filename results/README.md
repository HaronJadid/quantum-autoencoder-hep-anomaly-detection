# Results: start here

**Current study: [final-v2](final-v2/README.md).**

- [Main metrics](final-v2/metrics.json): 15 training seeds, fixed partition,
  event-weighted validation, 600-epoch training ceiling.
- [Detailed report](final-v2/report.md): all models and both signal topologies.
- [Separate 3,000-epoch classical diagnostic](final-v2/budget-3000/summary.json).

[legacy-v1/](legacy-v1/README.md) preserves the earlier protocol, figures,
reference selection and reproduction artifacts. Those measurements are not
the current headline results. There is intentionally no root `metrics.json`
alias that could silently select the wrong study.

`runs/` at the repository root is ignored local working storage, not a second
published results tree. Original evidence snapshots retain their historical
paths and bytes; current tools reference the explicit versioned directories.
