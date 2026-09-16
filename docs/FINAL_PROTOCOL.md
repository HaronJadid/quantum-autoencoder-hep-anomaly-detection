# Final-v2 execution plan

The current `results/legacy-v1/metrics.json` remains the historical legacy-v1 study.
Final-v2 has now completed and is packaged separately in `results/final-v2/`;
the historical root metrics file is intentionally retained, not replaced.
The new run is an exploratory reanalysis of a benchmark already examined during
development, not a newly blinded discovery test.

## Frozen choices for this run

- Data: the same checksum-verified LHCO v2 and three-prong feature files.
- Inputs and quantum/classical architectures: unchanged.
- One partition (split seed 0): 100,000 background training events, 20,000
  validation events, remaining 880,000 background test events; both signals
  are used for evaluation only. All 15 training seeds share this partition.
- Scaling is fitted only on training background, with fixed preprocessing seed 0.
- Validation is the event-weighted mean, including the short final batch.
- Adam, batch size 4096, maximum 600 training epochs, patience 20.
- RY depth candidates: 1, 3, 5, 7, 9; 100 selection epochs on 25,000 training
  events, using background validation loss. ZZ depth is pinned to the chosen
  RY depth. Its own depth scan is diagnostic, not the trained depth selection.
- Learning-rate grid remains 0.2, 0.1, 0.05, 0.01, 0.001. Quantum selection
  uses the 100-epoch/25k budget. Classical candidates use 100k training events,
  up to 600 epochs, patience 20, and initialisation seeds 10000–10002. Choose
  the rate with smallest mean best validation loss across those initialisations.
- Ties within 1e-9 use the smaller rate/shallower depth. This is a numerical
  tolerance, not a claim about statistical indistinguishability.
- Train seeds 0–14; save seed-0 event scores; evaluate both signals. No changes
  selected by comparing test AUCs. Keep boundary/cap warnings visible.
- Do not rerun the historical activation/epoch attribution diagnostic.

Quantum tuning has a smaller budget than classical tuning for runtime reasons;
this is not equal-compute tuning or a claim of fully optimised models.
The exact-parameter classical model may lack a bottleneck, so the untied and
dense controls remain essential. Training seeds measure conditional optimisation
variation, not independent datasets, detector systematics or hardware robustness.

## Colab

Use the updated notebook, `MODE = 'study'`, `RUN_NAME = 'qae-final-v2'`.
Keep the files in `/content/repo/data` or restore them there after a runtime reset.
The output is a separate Drive run directory, not the historical results folder.

Equivalent command:

```bash
python -u tools/colab_run.py --device cuda --seeds 15 --chunk 5 \
  --save-scores --no-ablation --protocol final-v2 --split-seed 0 \
  --selection-seed 0 --select-epochs 100 \
  --out /content/drive/MyDrive/QAE-runs/qae-final-v2/study
```

After completion, run Results, Figures and Export. Return the archive for audit:
protocol guards, 15 complete seeds, cap flags, selected budgets, both summaries,
score/figure consistency, and environment provenance. If a selected candidate or
training run hits its cap, assess background validation histories before claiming
convergence; a cap warning alone is not proof of undertraining.

Final handoff still requires interpretation from the new metrics, documentation
consistency, a clean reproducible entry point, and reviewed commits. Do not
change CV headline numbers to the new run before that audit.
