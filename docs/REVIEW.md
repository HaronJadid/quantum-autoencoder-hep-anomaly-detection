# Completion review — 2026-09-13

Historical legacy-v1 review. The current release is described in
`results/final-v2/README.md`, `FINAL_V2_AUDIT.md` and
`BUDGET_DIAGNOSTIC_AUDIT.md`. The numbers and cap status below refer to the
older study, not the final-v2 results now headlined in the README.

The saved study contains 15 seeds and both signal topologies. Recomputing the
summaries from the saved per-seed metrics reproduces both summary tables exactly.
All nine primary seed-0 AUCs equal the later saved Colab reproduction. The
recorded epoch-cap flag is false. These checks validate saved artifacts; this
review did not rerun the full training study.

The README interpretation now follows these artifacts rather than the handoff:
the pooled within-model correlation is -0.056, not the handoff's +0.357.
The three-prong RY result is promising within this protocol; the two-prong AUC
gain over the mass cut is unresolved, while rejection at efficiency 0.3 improves.

Corrections made:

- Replaced withdrawal notices with bounded interpretation from current results.
- Removed the independent-tests false-positive probability, since these tests
  are dependent; retained the Bonferroni correction and all computed p-values.
- Displayed small p-values in significant digits instead of rounding to zero.
- Restricted the compression statement to the sampled density matrix. Equal
  amplitude magnitudes do not imply orthogonality or maximal mixing.
- Corrected the complex density-matrix construction (it previously returned
  the conjugate matrix). This preserves eigenvalues mathematically and does
  not change training, anomaly scores or the spectral bound.
- Corrected the parameter-shift arithmetic and the claim that selection uses signal.
- Qualified parameter-count/capacity and platform-reproduction claims.
- Documented equal-batch validation weighting as part of the historical protocol.

Remaining limitations to explain in an interview:

- Hyperparameter selection is short-budget and uses a fixed absolute tolerance;
  that tolerance is a heuristic, not a measured statistical noise floor.
- Shared benchmark events and iterative project development mean the paired
  tests are conditional evidence, not an independent discovery confirmation.
- The classical comparison is sensitive to optimisation and environment. The
  observed portability spread is not a universal bound on future runs.
- Small score–mass correlation cannot rule out local mass sculpting.
- Validation weights the short final batch equally with full batches. Changing
  that rule would require a new versioned study; historical metrics are preserved.
- The notebook now calls the chunked Colab runner with inspect, frozen seed-0
  reproduction and fresh-study modes. Inspection executed locally and mocked
  orchestration/resume tests passed. An actual Colab GPU run through this updated
  notebook remains to be checked; the earlier reproduction predates these edits.

Run `python tools/audit_saved_results.py` to verify summaries, generated prose,
the optional local seed-0 comparison, and the complex density-matrix convention.
No commit or push was made during this review.
