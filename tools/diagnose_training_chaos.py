"""Show that the classical autoencoders' training is chaotic, and the QAE's is not.

Why this exists: the five classical autoencoders in this study do not give the
same per-seed AUC on two different machines, while the quantum models, PCA and
the m_J1 control give identical numbers to every decimal place reported. That
asymmetry needs a cause, and "it's a GPU rounding thing" is not one -- the
classical models run on the CPU in both cases.

The cause is that their optimisation trajectories are chaotic at the selected
learning rates. This perturbs the training data by one part in 1e15 -- far
below any physical meaning, roughly the size of a rounding difference between
two machines -- and retrains from the same seed. A stable model ignores it. A
chaotic one lands in a different basin and scores differently.

Measured on the laptop this study was developed on (6 threads, torch 2.14):

    ae_matched  seed 0   0.4521 -> 0.6210   (+0.169)
    ae_matched  seed 3   0.6374 -> 0.5663   (-0.071)
    ae_dense    seed 0   0.5749 -> 0.5588   (-0.016)
    ae_dense    seed 3   0.5489 -> 0.5666   (+0.018)
    ae_untied58 seed 0   0.5632 -> 0.5626   (-0.001)
    ae_untied58 seed 3   0.5725 -> 0.5762   (+0.004)

For contrast, the quantum models are not perturbed here but are checked
against a stronger test that has already been run: the same seeds trained on
two different machines, two Python versions, two torch versions and two
devices gave AUCs identical to four decimal places on all 13 comparable
seeds. The mechanism is visible in the training logs -- ae_matched at lr=0.1
shows validation-loss excursions to 2.7 from a baseline near 0.03, while the
QAE settles onto a flat plateau where dozens of epochs are within 1% of the
best.

Note this is NOT an early-stopping tie: ae_matched's best epoch wins by
1.3e-3 with a single epoch within 1% of it, while qae_ry -- the reproducible
one -- has 37. The trajectory diverges, not the choice of epoch along it.

    python tools/diagnose_training_chaos.py                # seeds 0 and 3
    python tools/diagnose_training_chaos.py --seeds 1 2 5

Takes a few minutes: only the classical models are trained, and they train in
seconds each.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import DenseAE, UntiedAE, ae_scores, matched_ae  # noqa: E402
from src.data import FEATURES, load_rnd, make_splits  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.run_study import LATENT, scale  # noqa: E402
from src.train import ae_loss, train_model  # noqa: E402

# The learning rates the study selected, so the diagnostic describes the models
# as actually trained rather than some other configuration of them.
MODELS = {
    "ae_matched": (lambda s: matched_ae(len(FEATURES), LATENT, 48, seed=s)[0], 0.1),
    "ae_untied32": (lambda s: UntiedAE(len(FEATURES), 2, seed=s), 0.05),
    "ae_untied45": (lambda s: UntiedAE(len(FEATURES), 3, seed=s), 0.05),
    "ae_untied58": (lambda s: UntiedAE(len(FEATURES), 4, seed=s), 0.05),
    "ae_dense": (lambda s: DenseAE(len(FEATURES), 16, LATENT, seed=s), 0.05),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--relative-perturbation", type=float, default=1e-15,
                    help="relative size of the perturbation (default 1e-15)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    df = load_rnd(args.data_dir)
    eps = args.relative_perturbation
    print(f"\nPerturbing the training data by one part in {1/eps:.0e} and "
          f"retraining from the same seed.")
    print("A rounding difference between two machines is this size or larger.\n")
    print(f"{'model':14s} {'seed':>4s} {'unperturbed':>12s} {'perturbed':>10s} "
          f"{'dAUC':>9s}")

    worst = {}
    for seed in args.seeds:
        sp = make_splits(df, 100_000, 20_000, seed=seed)
        xtr, xva, xte = scale(sp.train, sp.val, sp.test, seed=seed)
        rng = np.random.default_rng(0)
        xtr_p = xtr * (1.0 + eps * rng.standard_normal(xtr.shape))
        for name, (build, lr) in MODELS.items():
            aucs = []
            for data in (xtr, xtr_p):
                m = build(seed)
                train_model(m, ae_loss, data, xva, epochs=600, batch_size=4096,
                            lr=lr, seed=seed, patience=20, verbose=False)
                aucs.append(evaluate(ae_scores(m, xte), sp.test_label)["auc"])
            d = aucs[1] - aucs[0]
            worst[name] = max(worst.get(name, 0.0), abs(d))
            print(f"{name:14s} {seed:4d} {aucs[0]:12.4f} {aucs[1]:10.4f} "
                  f"{d:+9.4f}")

    print(f"\n{'model':14s} {'largest |dAUC| from a 1e-15 perturbation':>42s}")
    for name, d in sorted(worst.items(), key=lambda kv: -kv[1]):
        print(f"{name:14s} {d:42.4f}")
    print("\nAn |dAUC| here that is comparable to, or larger than, the model's "
          "across-seed spread means its per-seed number is not reproducible "
          "at the precision this study reports it, on any other machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
