# Final-v2 artifacts

`metrics.json`, `selection.json`, `run_manifest.json` and `figures/` come from
the received full-study archive. They are preserved independently of the
historical study in `../legacy-v1/`. `report.md` is regenerated
with current reporting code; corrections to statistical wording do not change
the archived measurements.

The generalisation figure was regenerated from the same metrics to correct
its labels and show mean ± sample SD across training seeds for all nine models.
Both signals are evaluation-only; neither is a trained-on signal.

`budget-3000/` contains the separate matched-classical-AE diagnostic: manifest,
15 per-seed records, summary and the exact source study file named by its
manifest. That source file differs from the primary archived metrics **only
in recorded durations** (per-model seconds and worker wall times). Configs,
loss histories and all measured detection results are identical. The audit
script checks this equivalence and the original-file SHA-256.

`source_snapshot/` preserves the original source modules and runner from before
release-reporting changes. It is evidence of the run implementation, not an
alternative current entry point. The budget manifest records hashes of every
source module; the original full-run manifest records the combined module and
runner hash. Current source is under the repository's top-level `src/`.

The user-supplied ZIPs contain primary seed-0 event scores, diagnostic seed-0
scores for both signals, and diagnostic weights. These large/binary artifacts
remain outside Git. The committed data support per-seed summary and history
audits but not independent recalculation of every event-level metric. Model
weights are not loaded by the audit.

Regenerate reports with `python tools/build_release_report.py`; verify them
without writing with `python tools/build_release_report.py --check`.
Run `python tools/audit_release.py` for provenance and numerical consistency.
