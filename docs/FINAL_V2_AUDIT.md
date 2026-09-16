# Received final-v2 study: audit status

Audit date: 2026-09-14. Source: user-supplied `study-bundle.zip`, extracted
without replacing published results into `runs/received-final-v2/`.

Release update (2026-09-16): final-v2 is packaged at `results/final-v2/`.
The extended-budget diagnostic is complete; see `BUDGET_DIAGNOSTIC_AUDIT.md`.
Source hashes and timing-only differences between received study copies are
verified by `tools/audit_release.py`. Reporting now suppresses t-tests for
constant paired differences and identifies worker timing correctly. The
observations below retain the original audit context, not a request to rerun.

## Verified

- The execution log ends with exit status 0 and reports 113 minutes.
- Three shards cover seeds 0–4, 5–9 and 10–14. All record selection digest
  `f34c739c8982f99e`.
- All nine models have 15 records for each of the two signal evaluations.
  Recomputing all 18 summaries with the aggregation function matches exactly.
- Independently recomputing the nine primary-signal seed-0 ROC-AUCs from saved
  event scores agrees within 2.3e-16 (floating-point rounding). Rejection at
  signal efficiency 0.3 agrees exactly for all nine models.
- Both quantum models and the parameter-matched classical AE have 60 trainable
  parameters. Historical 48-parameter descriptions do not describe this run.

## Main observations

Values below are mean ± sample standard deviation across 15 training seeds on
one fixed data partition, not uncertainty over independent datasets.

| Model | Primary ROC-AUC | Primary rejection at 0.3 signal efficiency | Three-prong ROC-AUC |
|---|---:|---:|---:|
| RY quantum AE | 0.5790 ± 0.0618 | 4.98 ± 1.18 | 0.7130 ± 0.0310 |
| Parameter-matched classical AE | 0.6658 ± 0.0258 | 9.03 ± 2.07 | 0.5673 ± 0.0219 |

The classical model has better mean performance on the primary signal; the RY
quantum model has better mean AUC on the alternate signal. These observations
do not establish general quantum advantage or a fully optimised comparison.

## Release caveats and unfinished checks

1. Every matched-classical run reaches 600 epochs. Its best checkpoint is at
   epoch 599 or 600. Validation loss falls by approximately 40.7–48.2% over the
   last 100 epochs. This is substantive evidence that its reconstruction
   objective has not plateaued, not merely a cap flag. Better reconstruction
   does not necessarily imply better anomaly AUC.
2. Selected depth is the upper grid boundary (9). The selected matched-AE
   learning rate is the lower boundary (0.001), and ZZ's is the upper boundary
   (0.2). Selection is budget- and grid-limited; do not claim optimality.
3. These data are simulated LHCO benchmark events, not measured collisions.
4. Event-score checks above cover the saved primary seed only. They do not
   independently reconstruct every seed or the alternate-signal evaluation.
5. Source/environment provenance, all figures, report inference for
   deterministic fixed-split controls, and release-document consistency still
   need completion before promotion. Published historical results are unchanged.

## Recommended next experiment

Preserve this run as the fixed-budget result. Add a separately labelled
training-budget sensitivity check for the matched classical AE, keeping its
architecture, selected learning rate, partition and preprocessing fixed. Choose
the extended budget and stopping rule before running it, using background
validation rather than test AUC to assess convergence. Do not silently replace
the original comparison or select a preferred checkpoint using signal metrics.
This diagnostic need not repeat quantum training and does not establish that
hyperparameter selection at the longer budget is optimal.
