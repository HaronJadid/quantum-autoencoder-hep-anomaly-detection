"""Run the full comparison: QAE vs classical baselines on LHCO R&D.

    python -m src.run_study                 # full study, 5 seeds
    python -m src.run_study --quick         # small smoke test
    python -m src.run_study --seeds 0 1 2   # pick seeds

Writes results/metrics.json and results/figures/*.png.

Each seed re-draws the train/val/test split AND re-initialises every model, so
the quoted spread covers both sources of variation rather than initialisation
alone. Every model in a seed sees exactly the same events.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from .baselines import (DenseAE, ae_scores, count_params, matched_ae, pca_scores,
                        single_feature_scores)
from .data import FEATURES, load_rnd, make_splits
from .encoding_analysis import compressibility, report
from .evaluate import aggregate, evaluate, roc_points
from .qae import FEATURE_MAPS, QuantumAutoencoder, n_params, verify_against_qiskit
from .train import ae_loss, qae_loss, train_model

# Ansatz depth is not fixed here: it is selected on background validation loss
# in main() from --reps-candidates, and recorded in the results.
N_QUBITS, N_TRASH = 6, 2
LATENT = N_QUBITS - N_TRASH


def scale(train, *others, seed=0):
    """Per-feature quantile transform to [0, pi], fit on background training data.

    Quantile (not min-max) because mj1 has a long tail: min-max would compress
    almost all events into a narrow angular range and waste the encoding.
    Fitting on the training background only avoids leaking test information.
    The transform is monotone per feature, so it cannot change the AUC of any
    single-feature baseline.
    """
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform",
                             subsample=200_000, random_state=seed).fit(train)
    out = [np.pi * qt.transform(train)]
    out += [np.pi * qt.transform(o) for o in others]
    return out


def select_reps(xtr, xva, feature_map, candidates, epochs, batch_size, lr,
                seed=0, verbose=True):
    """Choose ansatz depth by validation loss on BACKGROUND ONLY.

    This is legitimate unsupervised model selection: it uses no labels, no
    signal events and no test data -- only the background compression the model
    is actually trained to achieve. Selecting depth by test AUC instead would
    be tuning on the evaluation set, which would invalidate the comparison.
    """
    scores = {}
    for reps in candidates:
        m = QuantumAutoencoder(N_QUBITS, N_TRASH, reps, seed=seed,
                               feature_map=feature_map)
        h = train_model(m, qae_loss, xtr, xva, epochs=epochs,
                        batch_size=batch_size, lr=lr, seed=seed, verbose=False)
        scores[reps] = min(h.val_loss)
        if verbose:
            print(f"    reps={reps:2d} ({n_params(N_QUBITS, reps):3d} par.): "
                  f"best val loss {scores[reps]:.5f}")
    best = min(scores, key=scores.get)
    if verbose:
        print(f"    -> selected reps={best} for the {feature_map} encoding")
    return best, scores


def run_seed(splits, seed, epochs, batch_size, lr, reps_by_map, verbose=True):
    """Train and score every model on one split. Returns {name: scores}."""
    xtr, xva, xte = scale(splits.train, splits.val, splits.test, seed=seed)
    scores, histories, params = {}, {}, {}

    # Two quantum autoencoders, identical except for the encoding.
    # qae_ry  : the working model.
    # qae_zz  : the textbook ZZFeatureMap choice, included because it is the
    #           obvious thing to try and because showing that it provably
    #           cannot compress (see encoding_analysis) is a result in itself.
    for tag, fmap in (("qae_ry", "ry"), ("qae_zz", "zz")):
        reps = reps_by_map[fmap]
        if verbose:
            print(f"  [{tag}] {n_params(N_QUBITS, reps)} params, {N_QUBITS} qubits, "
                  f"{LATENT} latent + {N_TRASH} trash, {fmap} encoding, reps={reps}")
        qae = QuantumAutoencoder(N_QUBITS, N_TRASH, reps, seed=seed,
                                 feature_map=fmap)
        histories[tag] = train_model(qae, qae_loss, xtr, xva, epochs=epochs,
                                     batch_size=batch_size, lr=lr, seed=seed,
                                     verbose=verbose)
        scores[tag] = qae.score(xte)
        params[tag] = count_params(qae)

    # Matched to the QAE's ACTUAL parameter count, which depends on the depth
    # selected above rather than being fixed in advance.
    target = n_params(N_QUBITS, reps_by_map["ry"])
    tied, spec = matched_ae(len(FEATURES), LATENT, target, seed=seed)
    if verbose:
        print(f"  [ae_matched] tied-weight AE, {spec['params']} par. "
              f"(exact match to the QAE), latent {spec['latent']}, "
              f"output_affine={spec['output_affine']}, "
              f"latent_bias={spec['latent_bias']}")
    histories["ae_matched"] = train_model(tied, ae_loss, xtr, xva, epochs=epochs,
                                          batch_size=batch_size, lr=lr, seed=seed,
                                          verbose=verbose)
    scores["ae_matched"] = ae_scores(tied, xte)
    params["ae_matched"] = count_params(tied)

    if verbose:
        print("  [ae_dense] 6-16-4-16-6 dense AE")
    dense = DenseAE(len(FEATURES), 16, LATENT, seed=seed)
    histories["ae_dense"] = train_model(dense, ae_loss, xtr, xva, epochs=epochs,
                                        batch_size=batch_size, lr=lr, seed=seed,
                                        verbose=verbose)
    scores["ae_dense"] = ae_scores(dense, xte)
    params["ae_dense"] = count_params(dense)

    scores["pca"] = pca_scores(xtr, xte, n_components=LATENT)
    params["pca"] = LATENT * len(FEATURES)

    # The mj1 control uses RAW mj1, not the transformed value: the quantile
    # transform saturates at the training-set maximum, which would tie together
    # the most extreme signal events and understate this baseline.
    scores["mj1_only"] = single_feature_scores(splits.test, FEATURES.index("mj1"))
    params["mj1_only"] = 0

    return scores, histories, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=60)
    # 4096 measured ~2.3x faster per epoch than 1024 on CPU: the statevector
    # itself is tiny (64 amplitudes), so per-batch Python overhead dominates.
    # Larger still (16k+) gets slower again.
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reps-candidates", type=int, nargs="+", default=[1, 3, 5],
                    help="ansatz depths to choose between, on validation loss")
    ap.add_argument("--select-epochs", type=int, default=25,
                    help="epoch budget for depth selection (ordering only)")
    ap.add_argument("--select-n", type=int, default=25_000,
                    help="training events used for depth selection")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true",
                    help="tiny run to check the pipeline end to end")
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.n_train, args.n_val, args.epochs = [0], 4000, 1000, 5

    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    t_start = time.time()

    print("=" * 70)
    print("Verifying the PennyLane circuits against the Qiskit specification")
    # Every (encoding, depth) combination that could be selected below, so the
    # check always covers the circuit actually trained.
    verify = {}
    for fmap in FEATURE_MAPS:
        for reps in args.reps_candidates:
            d = verify_against_qiskit(N_QUBITS, reps, n_trials=8, feature_map=fmap)
            verify[f"{fmap}_reps{reps}"] = d
            print(f"  {fmap:3s} encoding, reps={reps}: "
                  f"max |amplitude difference| = {d:.3e}  (tol 1e-10)  PASS")

    print("=" * 70)
    df = load_rnd(args.data_dir)

    # Ansatz-independent ceiling on what each encoding can compress. Computed
    # once, on the first seed's background training set.
    print("=" * 70)
    print("Encoding compressibility (upper bound over ALL ansaetze)")
    probe = make_splits(df, args.n_train, args.n_val, seed=args.seeds[0])
    xprobe, = scale(probe.train, seed=args.seeds[0])
    encoding = {}
    for fmap in FEATURE_MAPS:
        encoding[fmap] = compressibility(FEATURE_MAPS[fmap], xprobe,
                                         N_QUBITS, N_TRASH)
        print(report(encoding[fmap], f"  {fmap} encoding"))
        print()

    # Ansatz depth, chosen on background validation loss only (no labels, no
    # signal, no test data). Done once, on the first seed, then held fixed.
    print("=" * 70)
    print(f"Selecting ansatz depth on background validation loss "
          f"({args.select_n:,} events, {args.select_epochs} epochs)")
    xtr_p, xva_p = scale(probe.train, probe.val, seed=args.seeds[0])
    xtr_p = xtr_p[:args.select_n]
    reps_by_map, reps_scan = {}, {}
    for fmap in FEATURE_MAPS:
        print(f"  {fmap} encoding:")
        best, scan = select_reps(xtr_p, xva_p, fmap, args.reps_candidates,
                                 args.select_epochs, args.batch_size, args.lr,
                                 seed=args.seeds[0])
        reps_by_map[fmap] = best
        reps_scan[fmap] = {str(k): v for k, v in scan.items()}

    per_model_runs, all_hist, all_params, roc_cache = {}, {}, {}, {}
    for seed in args.seeds:
        print("=" * 70)
        print(f"SEED {seed}")
        splits = make_splits(df, args.n_train, args.n_val, seed=seed)
        scores, hist, params = run_seed(splits, seed, args.epochs,
                                        args.batch_size, args.lr, reps_by_map)
        all_params = params
        for name, sc in scores.items():
            res = evaluate(sc, splits.test_label, splits.test_mjj)
            per_model_runs.setdefault(name, []).append(res)
            r30 = res["rejection"]["eps_s=0.3"]
            print(f"    {name:12s} AUC {res['auc']:.4f}   "
                  f"1/eps_B@0.3 = {r30['rejection']:8.1f}   "
                  f"rho(score,mjj) = {res['sculpting']['spearman_score_vs_mjj']:+.3f}")
            if seed == args.seeds[0]:
                tpr, fpr = roc_points(sc, splits.test_label)
                roc_cache[name] = (tpr.tolist(), fpr.tolist())
        if seed == args.seeds[0]:
            # Keep the first seed's raw scores in memory: the score-distribution
            # and sculpting figures need per-event values, which are far too
            # large to round-trip through metrics.json.
            first = (scores, splits.test_label, splits.test_mjj)
        all_hist[seed] = {k: {"train_loss": v.train_loss, "val_loss": v.val_loss,
                              "best_epoch": v.best_epoch, "seconds": v.seconds}
                          for k, v in hist.items()}

    summary = {name: aggregate(runs) for name, runs in per_model_runs.items()}

    out = {
        "config": {
            "n_qubits": N_QUBITS, "n_latent": LATENT, "n_trash": N_TRASH,
            "ansatz_reps": reps_by_map, "reps_selection": reps_scan,
            "reps_candidates": args.reps_candidates,
            "reps_selection_budget": {"n_events": args.select_n,
                                      "epochs": args.select_epochs,
                                      "criterion": "background validation loss"},
            "features": FEATURES,
            "n_train": args.n_train, "n_val": args.n_val,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "seeds": args.seeds, "quick": args.quick,
            "simulation": "noiseless statevector, infinite shots",
            "qiskit_pennylane_max_amplitude_diff": verify,
        },
        "encoding_compressibility": encoding,
        "parameter_counts": all_params,
        "summary": summary,
        "per_seed": per_model_runs,
        "histories": all_hist,
        "roc_first_seed": roc_cache,
        "wall_seconds": time.time() - t_start,
    }
    path = os.path.join(args.out, "metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("=" * 70)
    print(f"wrote {path}  ({time.time()-t_start:.0f}s total)")

    figdir = os.path.join(args.out, "figures")
    try:
        from .figures import make_all, score_figure, sculpting_figure
        make_all(path, figdir)
        sc0, lab0, mjj0 = first
        score_figure(sc0, lab0, os.path.join(figdir, "scores.png"))
        sculpting_figure(sc0, lab0, mjj0, os.path.join(figdir, "sculpting.png"))
        print(f"wrote score and sculpting figures to {figdir}")
    except Exception as exc:                          # noqa: BLE001
        # Never lose a completed study to a plotting failure; metrics.json is
        # already on disk and `python -m src.figures` can regenerate later.
        print(f"WARNING: figure generation failed ({type(exc).__name__}: {exc}); "
              f"metrics.json is intact")

    print()
    print(f"{'model':14s} {'params':>7s} {'AUC':>16s} {'1/eps_B @ eps_S=0.3':>22s}")
    for name, agg in summary.items():
        a, r = agg["auc"], agg["rejection[eps_s=0.3]"]
        print(f"{name:14s} {all_params[name]:7d} "
              f"{a['mean']:.4f} +/- {a['std']:.4f}   "
              f"{r['mean']:10.1f} +/- {r['std']:.1f}")


if __name__ == "__main__":
    main()
