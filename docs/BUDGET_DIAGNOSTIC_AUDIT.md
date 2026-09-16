# Matched classical AE: extended-budget audit

Audited 2026-09-16 from `matched-ae-budget-3000.zip`, extracted separately into
`runs/received-budget-3000`. No published study artifacts were replaced.

## Verified

- Fifteen seed records, matching 60-parameter architecture (latent dimension 6).
- Both signal summaries recompute exactly from their per-seed metrics.
- All recorded source-file hashes match the current local source files.
- For every seed, the first 600 validation losses match the earlier final-v2
  study exactly.
- All runs stop before the 3000-epoch cap: 1029–1398 epochs, best checkpoints
  1009–1378. Each stops 20 epochs after its best checkpoint, consistent with
  the specified patience. This is stopping-rule evidence, not proof of global
  convergence or optimal hyperparameters.
- Saved seed-0 scores independently reproduce both signal AUCs to numerical
  precision and all six background-rejection records exactly.
- Recorded summed training time is approximately 17.0 minutes; this excludes
  setup and evaluation.

## Results

Means ± sample SD over the same 15 training seeds on a fixed partition:

| Matched AE metric | Original 600-epoch budget | Extended 3000-epoch budget |
|---|---:|---:|
| Primary ROC-AUC | 0.6658 ± 0.0258 | 0.6554 ± 0.0167 |
| Primary rejection at signal efficiency 0.3 | 9.03 ± 2.07 | 8.54 ± 1.31 |
| Three-prong ROC-AUC | 0.5673 ± 0.0219 | 0.5618 ± 0.0175 |

The mean primary AUC remains above the RY QAE's 0.5790, and the mean
three-prong AUC remains below its 0.7130. Longer training does not reverse
these observed rankings. Retain this as a separate budget-sensitivity result,
not a silent substitution into the equal-epoch-ceiling study.

## Resolved provenance check

The diagnostic's recorded original-metrics SHA-256 is
`4ef09fa8542955e84e14062c8ff5df941ae24adf16fcfce4a251e0eceef1265d`.
The earlier received final-v2 metrics file has SHA-256
`b8ef70d89f132ce2d4378f476313e9895ec8419bc5f66603378ea09ebb70a3b4`.
The subsequently supplied Drive file matches the diagnostic's checksum.
Parsed comparison shows differences only in per-model `seconds`,
`wall_seconds` and `wall_seconds_summed`. All configurations, loss histories
and detection metrics match exactly. Both source files are retained under
`results/final-v2/`; `tools/audit_release.py` verifies timing-only equivalence.
