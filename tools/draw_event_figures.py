"""Draw the two per-event figures from a saved score file.

The score-distribution and sculpting figures need one anomaly score per test
event per model -- around a million numbers each -- which is far too much to
carry in metrics.json. A run that trains its seeds in separate processes
therefore saves them to an .npz (`--save-first`) and this draws from that,
so those two figures still come from the real scores of the real run rather
than being approximated from anything summarised.

    python tools/draw_event_figures.py results/parallel/first_seed_scores.npz
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.figures import score_figure, sculpting_figure  # noqa: E402


def main(npz="results/parallel/first_seed_scores.npz",
         outdir="results/figures") -> int:
    d = np.load(npz)
    scores = {k[len("score__"):]: d[k] for k in d.files if k.startswith("score__")}
    if not scores:
        raise SystemExit(f"{npz} holds no per-model scores")
    label, mjj = d["label"], d["mjj"]
    os.makedirs(outdir, exist_ok=True)
    score_figure(scores, label, os.path.join(outdir, "scores.png"))
    sculpting_figure(scores, label, mjj, os.path.join(outdir, "sculpting.png"))
    print(f"wrote scores.png and sculpting.png to {outdir} "
          f"({len(scores)} models, {len(label):,} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
