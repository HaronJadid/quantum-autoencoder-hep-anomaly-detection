# Local release check — 2026-09-16

Completed locally:

- Final-v2 study and extended classical diagnostic packaged separately.
- Historical legacy-v1 metrics/reference artifacts retained.
- Source snapshot and exact-byte evidence preservation rules added.
- Original-file checksum discrepancy resolved as timing-only differences.
- README headline table generated from saved JSON; detailed report generated
  separately, with fixed-split constant-difference tests suppressed.
- Notebook inspect/reproduce modes updated to final-v2; inspect execution tested.
- Reusable classical-only budget diagnostic added; tiny fixture training,
  output generation and completed-seed resume tested. Full scientific runs
  are the user-supplied Colab runs, not a new local full retraining.
- All seven study figures visually inspected; misleading trained-on-signal
  labels corrected in the alternate-signal comparison figure.
- Dependency prose no longer claims that matching metrics establishes a unique
  environment or guarantees cross-platform reproduction.
- CV preparation is kept outside the Git checkout, in the workspace's private folder.
- Headline table includes background-mass correlation and top-1% median-mass
  shift, with across-seed SD and a warning against causal/sculpting claims.
- All-model two-topology figure now displays sample-SD error bars on shared axes.
- Selection budgets are quantified as maximum event exposures, not compute.
- Process notes are under `docs/`; old results are under `results/legacy-v1/`.

Before announcing the GitHub release:

- Review and commit the complete intended diff, including the earlier uncommitted
  protocol work. No commit or push was performed as part of this local pass.
- Publish the reviewed changes to the repository. The Colab clone fallback
  cannot fetch these local-only changes until then.
- Optional stronger archival release: publish original score/weight bundles
  under a stable release or DOI. They currently remain local, not in Git.

Scope of verification: recorded artifacts, numerical summaries, provenance,
local notebook inspect mode and regression tests. No fresh dependency install,
Docker build, quantum retraining or new clean-Colab end-to-end run was performed
during this release-packaging pass. Do not describe those checks as completed.
