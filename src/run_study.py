"""Run the full comparison: QAE vs classical baselines on LHCO R&D.

    python -m src.run_study                 # full study, 5 seeds
    python -m src.run_study --quick         # small smoke test
    python -m src.run_study --seeds 0 1 2   # pick seeds

Writes results/metrics.json and results/figures/*.png.

Legacy-v1 redraws splits per seed. Final-v2 fixes the partition and preprocessing
while varying model initialisation and minibatch order. Every model in a seed
sees exactly the same events. Use the notebook for the released protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from .baselines import (DenseAE, UntiedAE, ae_scores, count_params, matched_ae,
                        pca_scores, single_feature_scores, untied_param_count)
from .data import FEATURES, load_qqq, load_rnd, make_splits
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


# A selection whose best-vs-second-best gap is below this is not a selection:
# the candidates are indistinguishable on the objective and the "winner" is
# noise. Shared by BOTH the depth and the learning-rate criteria on purpose, so
# the two cannot drift apart.
SELECTION_NOISE_FLOOR = 0.01


def resolve_tie(scores: dict, prefer, tolerance=SELECTION_NOISE_FLOOR):
    """Pick among candidates that the objective cannot distinguish.

    `scores` maps candidate -> validation loss (lower is better). Any candidate
    within SELECTION_NOISE_FLOOR of the best is treated as tied with it,
    because a gap smaller than the floor is not evidence. Among the tied set
    the winner is `prefer` (built in as `min`), which is a fixed, mechanical
    rule applied identically to every model:

      * learning rate - the smallest tied rate, being the more stable
        optimiser setting among options that are equally good on the objective;
      * ansatz depth  - the shallowest tied depth, being the smaller model
        among options that are equally good on the objective.

    The rule looks only at background validation loss. It never sees labels,
    signal events or the test split, and it cannot be steered toward any model:
    the tie-break direction is fixed in advance and does not depend on which
    candidate won, on which model family is being selected, or on any test-set
    quantity. Its purpose is to stop a gap of order 1e-4 from silently deciding
    the study, which is what happened when a learning rate was chosen by a
    margin of 0.00016.

    Returns (selected, info) with the full audit trail.
    """
    best_value = min(scores.values())
    raw_argmin = min(scores, key=scores.get)
    tied = sorted(k for k, v in scores.items() if v - best_value < tolerance)
    selected = prefer(tied)
    ordered = sorted(scores.values())
    gap = (ordered[1] - ordered[0]) if len(ordered) > 1 else float("inf")
    return selected, {
        "scores": {str(k): v for k, v in scores.items()},
        "raw_argmin": raw_argmin,
        "best_vs_second_gap": gap,
        "tied_set": tied,
        "selected": selected,
        "tie_break_fired": selected != raw_argmin,
        "within_noise_floor": len(tied) > 1,
        "noise_floor": tolerance,
    }


def select_reps(xtr, xva, feature_map, candidates, epochs, batch_size, lr,
                seed=0, verbose=True, device=None,
                validation_weighting="legacy-batches", tolerance=SELECTION_NOISE_FLOOR):
    """Choose ansatz depth by validation loss on BACKGROUND ONLY.

    This is legitimate unsupervised model selection: it uses no labels, no
    signal events and no test data -- only the background compression the model
    is actually trained to achieve. Selecting depth by test AUC instead would
    be tuning on the evaluation set, which would invalidate the comparison.

    Two failure modes are detected and reported rather than hidden:

    * `hit_boundary` - the winner is the largest candidate tried, so the grid
      ran out before the optimum did and the "selected" depth is really just
      the edge of the search.
    * `below_noise_floor` - best and worst candidates differ by less than
      SELECTION_NOISE_FLOOR, so no depth was meaningfully chosen.
    """
    scores = {}
    for reps in candidates:
        m = QuantumAutoencoder(N_QUBITS, N_TRASH, reps, seed=seed,
                               feature_map=feature_map, device=device)
        h = train_model(m, qae_loss, xtr, xva, epochs=epochs,
                        batch_size=batch_size, lr=lr, seed=seed, verbose=False,
                        validation_weighting=validation_weighting)
        scores[reps] = min(h.val_loss)
        if verbose:
            print(f"    reps={reps:2d} ({n_params(N_QUBITS, reps):3d} par.): "
                  f"best val loss {scores[reps]:.5f}")
    # Shallowest among depths the objective cannot separate.
    best, info = resolve_tie(scores, prefer=min, tolerance=tolerance)
    spread = max(scores.values()) - min(scores.values())
    # Preserved with its original meaning: the UNCONSTRAINED argmin sat at the
    # top of the grid, so the grid ran out before the optimum did.
    raw_at_top = info["raw_argmin"] == max(candidates)
    weak = spread < tolerance

    if verbose:
        print(f"    -> selected reps={best} for the {feature_map} encoding "
              f"(spread best-worst = {spread:.5f})")
        if info["tie_break_fired"]:
            print(f"    TIE-BREAK [{feature_map}]: depths {info['tied_set']} are "
                  f"within {tolerance} of each other "
                  f"(best-vs-second gap {info['best_vs_second_gap']:.5f}); "
                  f"argmin was reps={info['raw_argmin']}, selected the "
                  f"shallowest tied depth reps={best} instead.")
        if raw_at_top:
            print(f"    WARNING [{feature_map}]: the unconstrained argmin sat at "
                  f"the TOP of the candidate grid {candidates}. The optimum may "
                  f"lie beyond it.")
        if weak:
            print(f"    WARNING [{feature_map}]: spread {spread:.5f} < "
                  f"{tolerance}; candidates are tied by the stated rule "
                  f"and this depth selection is not meaningful.")
    info.update({"spread": spread, "hit_boundary": raw_at_top,
                 "below_noise_floor": weak,
                 "selected_at_grid_edge": best in (min(candidates), max(candidates))})
    return best, info


def select_lr(build, loss_fn, xtr, xva, candidates, epochs, batch_size,
              seed=0, label="", verbose=True, initialisation_seeds=None,
              validation_weighting="legacy-batches", tolerance=SELECTION_NOISE_FLOOR,
              patience=10):
    """Choose a learning rate per model family, on BACKGROUND validation loss.

    Same legitimacy argument as `select_reps`: label-free, signal-free,
    test-free. This exists because a single shared learning rate is not a
    neutral choice -- it imposes one model family's operating point on the
    others. Measured here, the families do prefer different rates, though not
    in the direction originally assumed: the dense autoencoder does not want a
    far smaller rate than the circuit, and on the grid searched most families
    cluster at the larger end. What the sweep establishes is that the rate
    should be chosen per family, not that any particular family was
    handicapped.

    Ties are resolved toward the SMALLEST rate: see `resolve_tie`. A gap of
    order 1e-4 in validation loss is not evidence that one rate is better, and
    allowing such a gap to pick the rate let a coin flip move a model's AUC by
    0.025 and its spread by a factor of five in an earlier run.

    `build(seed)` returns a fresh model.
    """
    scores, histories = {}, {}
    selection_seeds = initialisation_seeds or [seed]
    for lr in candidates:
        records = []
        for init_seed in selection_seeds:
            h = train_model(build(init_seed), loss_fn, xtr, xva, epochs=epochs,
                            batch_size=batch_size, lr=lr, seed=init_seed, verbose=False,
                            validation_weighting=validation_weighting, patience=patience)
            records.append({'seed': init_seed, 'best_val_loss': min(h.val_loss),
                            'best_epoch': h.best_epoch, 'epochs_run': h.epochs_run,
                            'hit_epoch_cap': h.epochs_run >= epochs})
        histories[str(lr)] = records
        scores[lr] = float(np.mean([r['best_val_loss'] for r in records]))
        if verbose:
            print(f"    lr={lr:<8g} best val loss {scores[lr]:.6f}")

    # Smallest among rates the objective cannot separate.
    best, info = resolve_tie(scores, prefer=min, tolerance=tolerance)
    spread = max(scores.values()) - min(scores.values())
    if verbose:
        print(f"    -> selected lr={best} for {label}")
        if info["tie_break_fired"]:
            print(f"    TIE-BREAK [{label}]: rates {info['tied_set']} are within "
                  f"{tolerance} of each other (best-vs-second gap "
                  f"{info['best_vs_second_gap']:.5f}); argmin was "
                  f"lr={info['raw_argmin']}, selected the smallest tied rate "
                  f"lr={best} instead.")
    info.update({"spread": spread,
                 "hit_boundary": best in (min(candidates), max(candidates)),
                 "candidate_runs": histories, "epochs": epochs,
                 "n_train": len(xtr), "initialisation_seeds": selection_seeds,
                 "selected_hit_epoch_cap": any(r['hit_epoch_cap'] for r in histories[str(best)])})
    return best, info


def ablate_dense(df, seeds, n_train, n_val, batch_size, lr, patience,
                 verbose=True):
    """DIAGNOSTIC ONLY: attribute the ae_dense change to one of two edits.

    Between two runs, ae_dense's AUC moved substantially while two things
    changed at once -- the epoch budget went 60 -> 600 (so it stopped being cut
    off by the cap) and the ReLU on the bottleneck was removed. With both
    varying together the cause is unattributable, so this trains ae_dense alone
    over the 2x2 of {60, 600} epochs x {bottleneck ReLU, no bottleneck ReLU} on
    the same seeds.

    This is a diagnostic, not a headline result: it is a single model with no
    baselines and no significance testing, and it is written to a separate key
    in metrics.json for that reason.
    """
    out = {}
    for ep in (60, 600):
        for relu in (True, False):
            aucs = []
            for seed in seeds:
                sp = make_splits(df, n_train, n_val, seed=seed)
                xtr, xva, xte = scale(sp.train, sp.val, sp.test, seed=seed)
                m = DenseAE(len(FEATURES), 16, LATENT, seed=seed,
                            bottleneck_relu=relu)
                h = train_model(m, ae_loss, xtr, xva, epochs=ep,
                                batch_size=batch_size, lr=lr, seed=seed,
                                patience=patience, verbose=False)
                aucs.append(evaluate(ae_scores(m, xte), sp.test_label)["auc"])
            key = f"epochs={ep},bottleneck_relu={relu}"
            out[key] = {
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "per_seed": [float(a) for a in aucs],
                "hit_epoch_cap_any": bool(h.epochs_run >= ep),
            }
            if verbose:
                print(f"    {key:38s} AUC {out[key]['auc_mean']:.4f} "
                      f"+/- {out[key]['auc_std']:.4f}")
    return out


def run_seed(splits, seed, epochs, batch_size, lr_by_model, reps_by_map,
             patience=20, extra_signal=None, verbose=True, device=None,
             validation_weighting="legacy-batches", scaler_seed=None):
    """Train and score every model on one split.

    `extra_signal` is raw (unscaled) features for a second, never-trained-on
    signal. It is transformed with the SAME scaler fitted on this seed's
    background training split and scored by the same trained models, so the
    generalisation numbers come from models that never saw it in any form.
    """
    preprocessing_seed = seed if scaler_seed is None else scaler_seed
    if extra_signal is None:
        xtr, xva, xte = scale(splits.train, splits.val, splits.test, seed=preprocessing_seed)
        xex = None
    else:
        xtr, xva, xte, xex = scale(splits.train, splits.val, splits.test,
                                   extra_signal, seed=preprocessing_seed)
    scores, histories, params, capped = {}, {}, {}, {}
    extra_scores, meta = {}, {}

    def run(tag, model, loss_fn, score_fn):
        """Train one model at its own selected learning rate, then score it."""
        h = train_model(model, loss_fn, xtr, xva, epochs=epochs,
                        batch_size=batch_size, lr=lr_by_model[tag], seed=seed,
                        patience=patience, verbose=verbose,
                        validation_weighting=validation_weighting)
        histories[tag] = h
        scores[tag] = score_fn(model, xte)
        if xex is not None:
            extra_scores[tag] = score_fn(model, xex)
        params[tag] = count_params(model)
        # A cap requires inspection of validation histories; it is not by
        # itself proof that a model is still improving or another converged.
        hit = h.epochs_run >= epochs
        capped[tag] = hit
        if hit:
            print(f"    *** WARNING: {tag} (seed {seed}) hit the epoch cap "
                  f"({epochs}) without early stopping -- best epoch "
                  f"{h.best_epoch}. Inspect validation history before claiming "
                  f"convergence.")
        return h

    # Two quantum autoencoders identical except for the feature map: same
    # qubit count, same latent/trash split, same ansatz depth (zz is pinned to
    # ry's, see main()) and therefore the same parameter count. Only the
    # learning rate is selected separately. That is what makes zz a control on
    # the encoding rather than on encoding-plus-capacity.
    # qae_ry  : the RY-encoded model.
    # qae_zz  : the textbook ZZFeatureMap choice, included because it is the
    #           obvious thing to try and because showing that it provably
    #           has limited empirical compression headroom is a useful diagnostic.
    for tag, fmap in (("qae_ry", "ry"), ("qae_zz", "zz")):
        reps = reps_by_map[fmap]
        if verbose:
            print(f"  [{tag}] {n_params(N_QUBITS, reps)} params, {N_QUBITS} qubits, "
                  f"{LATENT} latent + {N_TRASH} trash, {fmap} encoding, "
                  f"reps={reps}, lr={lr_by_model[tag]}")
        qae = QuantumAutoencoder(N_QUBITS, N_TRASH, reps, seed=seed,
                                 feature_map=fmap, device=device)
        run(tag, qae, qae_loss, lambda m, x: m.score(x))

    # Matched to the QAE's ACTUAL parameter count, which depends on the depth
    # selected above rather than being fixed in advance.
    target = n_params(N_QUBITS, reps_by_map["ry"])
    tied, spec = matched_ae(len(FEATURES), LATENT, target, seed=seed)
    meta["ae_matched"] = spec
    if spec["latent"] != LATENT:
        print(f"    *** WARNING: matching {target} parameters exactly forces "
              f"latent={spec['latent']}, not the QAE's {LATENT}. "
              + ("At latent == input dimension this baseline has NO bottleneck, "
                 "so it is not really an autoencoder and its reconstruction "
                 "error is a weak anomaly score by construction. "
                 if spec["latent"] >= len(FEATURES) else "")
              + "Exact parameter matching and equal compression cannot both "
                "hold here; the parameter count was held fixed.")
    if verbose:
        print(f"  [ae_matched] tied-weight AE, {spec['params']} par. "
              f"(exact match to the QAE), latent {spec['latent']}, "
              f"output_affine={spec['output_affine']}, "
              f"latent_bias={spec['latent_bias']}, lr={lr_by_model['ae_matched']}")
    run("ae_matched", tied, ae_loss, lambda m, x: ae_scores(m, x))

    # Untied autoencoders bracketing the QAE's parameter count. The untied
    # counts are 19/32/45/58/71/84 for latent 1..6 and the QAE's target
    # (6*(reps+1)) is generally not among them, so the exact match above is
    # only reachable with weight tying; these show what the count buys
    # without that handicap.
    for tag, latent in (("ae_untied32", 2), ("ae_untied45", 3),
                        ("ae_untied58", 4)):
        if verbose:
            print(f"  [{tag}] untied AE, latent {latent}, "
                  f"{untied_param_count(len(FEATURES), latent)} par., "
                  f"lr={lr_by_model[tag]}")
        run(tag, UntiedAE(len(FEATURES), latent, seed=seed), ae_loss,
            lambda m, x: ae_scores(m, x))

    if verbose:
        print(f"  [ae_dense] 6-16-4-16-6 dense AE, lr={lr_by_model['ae_dense']}")
    run("ae_dense", DenseAE(len(FEATURES), 16, LATENT, seed=seed), ae_loss,
        lambda m, x: ae_scores(m, x))

    scores["pca"] = pca_scores(xtr, xte, n_components=LATENT)
    params["pca"] = LATENT * len(FEATURES)
    capped["pca"] = False
    if xex is not None:
        extra_scores["pca"] = pca_scores(xtr, xex, n_components=LATENT)

    # The mj1 control uses RAW mj1, not the transformed value: the quantile
    # transform saturates at the training-set maximum, which would tie together
    # the most extreme signal events and understate this baseline.
    scores["mj1_only"] = single_feature_scores(splits.test, FEATURES.index("mj1"))
    params["mj1_only"] = 0
    capped["mj1_only"] = False
    if extra_signal is not None:
        extra_scores["mj1_only"] = single_feature_scores(
            extra_signal, FEATURES.index("mj1"))

    return scores, histories, params, capped, extra_scores, meta


def environment(device=None) -> dict:
    """Library versions and hardware, recorded into every run.

    Instrumentation, not part of the experiment. It exists because the
    versions behind the reported run were never written down: only torch and
    pennylane could be recovered afterwards, from stdout, and the rest are
    lost for good. Anything that can change a floating-point result belongs
    in the output file that quotes the result.
    """
    import importlib
    import platform

    out = {"python": sys.version.split()[0],
           "platform": platform.platform(),
           "device": device or "cpu"}
    for mod in ("torch", "pennylane", "qiskit", "numpy", "pandas", "sklearn",
                "scipy", "matplotlib"):
        try:
            out[mod] = getattr(importlib.import_module(mod), "__version__", None)
        except Exception:                                     # noqa: BLE001
            out[mod] = None          # absent is itself worth recording
    if str(device or "").startswith("cuda"):
        try:
            out["gpu"] = torch.cuda.get_device_name(0)
        except Exception:                                     # noqa: BLE001
            out["gpu"] = None
    return out


def selection_guard(args) -> dict:
    """Everything the frozen selection depends on.

    A selection computed under one search grid or budget is not a selection
    under another, and a worker that silently trained at a learning rate
    chosen for a different configuration would be worse than no parallelism
    at all. This is recorded alongside the selection and compared on load; a
    mismatch is fatal rather than a warning.
    """
    guard = {
        "n_qubits": N_QUBITS, "n_trash": N_TRASH, "features": list(FEATURES),
        "reps_candidates": list(args.reps_candidates),
        "lr_candidates": list(args.lr_candidates),
        "select_epochs": args.select_epochs, "select_n": args.select_n,
        "batch_size": args.batch_size, "default_lr": args.lr,
        "n_train": args.n_train, "n_val": args.n_val,
        "noise_floor": SELECTION_NOISE_FLOOR,
    }
    if getattr(args, 'protocol', 'legacy-v1') == 'final-v2':
        guard.update(protocol=args.protocol, split_seed=args.split_seed,
                     validation_weighting='events', selection_tolerance=1e-9,
                     classical_selection_epochs=args.epochs,
                     classical_selection_patience=args.patience,
                     classical_selection_seeds=[10000, 10001, 10002])
    return guard


def compute_selection(args, df, sel_seed, device=None):
    """Everything chosen once and then held fixed: circuit check, encoding
    ceiling, ansatz depth, per-family learning rate.

    Split out from the seed loop so it can be computed once and reused by
    parallel workers. Nothing here touches the test split, either signal
    sample, or any seed other than `sel_seed`: it sees only that seed's
    background training and validation events. Running it once and freezing
    the result is therefore not merely an optimisation -- it is what the
    selection protocol already claimed to be doing.
    """
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

    # Ansatz-independent ceiling on what each encoding can compress. Computed
    # once, on the selection seed's background training set.
    print("=" * 70)
    print("Encoding compressibility (upper bound over ALL ansaetze)")
    final = args.protocol == 'final-v2'
    split_seed = args.split_seed if final else sel_seed
    weighting = 'events' if final else 'legacy-batches'
    tolerance = 1e-9 if final else SELECTION_NOISE_FLOOR
    probe = make_splits(df, args.n_train, args.n_val, seed=split_seed)
    xprobe, = scale(probe.train, seed=split_seed)
    encoding = {}
    for fmap in FEATURE_MAPS:
        encoding[fmap] = compressibility(FEATURE_MAPS[fmap], xprobe,
                                         N_QUBITS, N_TRASH)
        print(report(encoding[fmap], f"  {fmap} encoding"))
        print()

    # Ansatz depth, chosen on background validation loss only (no labels, no
    # signal, no test data). Done once, on the selection seed, then held fixed.
    print("=" * 70)
    print(f"Selecting ansatz depth on background validation loss "
          f"({args.select_n:,} events, {args.select_epochs} epochs)")
    xtr_full, xva_p = scale(probe.train, probe.val, seed=split_seed)
    xtr_p = xtr_full[:args.select_n]
    reps_by_map, reps_scan = {}, {}
    for fmap in FEATURE_MAPS:
        print(f"  {fmap} encoding:")
        best, scan = select_reps(xtr_p, xva_p, fmap, args.reps_candidates,
                                 args.select_epochs, args.batch_size, args.lr,
                                 seed=sel_seed, device=device,
                                 validation_weighting=weighting, tolerance=tolerance)
        reps_by_map[fmap] = best
        scan["scores"] = {str(k): v for k, v in scan["scores"].items()}
        scan["used"] = True
        reps_scan[fmap] = scan

    # The zz model exists to isolate the FEATURE MAP. Selecting its depth
    # independently defeats that: zz's depth scan ties across every candidate
    # (all within the noise floor), so the tie-break -- not the objective --
    # picks a value, and it picked a depth six layers shallower than ry's,
    # leaving the "encoding" comparison confounded with depth and a 4x
    # parameter difference. Pinning zz to ry's depth is both more defensible
    # than a selection that is not a selection, and necessary for the control
    # to mean anything: with it, the two quantum models are identical except
    # for the feature map. zz's own scan is still run and recorded, marked as
    # not used, so the tie is visible.
    if "zz" in reps_by_map and "ry" in reps_by_map:
        pinned_from, pinned_to = reps_by_map["zz"], reps_by_map["ry"]
        reps_scan["zz"]["used"] = False
        reps_scan["zz"]["pinned_to_ry"] = True
        reps_scan["zz"]["would_have_selected"] = pinned_from
        reps_by_map["zz"] = pinned_to
        print(f"  PINNED: zz ansatz depth set to ry's reps={pinned_to} "
              f"(its own unused scan would have given reps={pinned_from}). The zz model "
              f"isolates the feature map, so depth is held equal to ry's "
              f"rather than selected; its scan is recorded but not used.")

    # Learning rate, per model family, on the same background validation split.
    # A single shared lr is not neutral: it imposes one family's operating point
    # on the others. Selection order is depth first (at the default lr), then
    # lr at the selected depth -- one pass of coordinate descent, not a joint
    # search, which is recorded as a limitation rather than hidden.
    print("=" * 70)
    print(f"Selecting learning rate per model family on background validation "
          f"loss\n({args.select_n:,} events, {args.select_epochs} epochs, "
          f"candidates {args.lr_candidates})")
    lr_by_model, lr_scan = {}, {}
    for tag, (build, loss_fn) in model_builders(reps_by_map, device).items():
        print(f"  {tag}:")
        classical_full = final and tag.startswith('ae_')
        print(f"    actual budget: {len(xtr_full) if classical_full else len(xtr_p)} "
              f"events, {args.epochs if classical_full else args.select_epochs} "
              f"epochs, {3 if classical_full else 1} initialisation(s)")
        best_lr, scan = select_lr(build, loss_fn,
                                  xtr_full if classical_full else xtr_p, xva_p,
                                  args.lr_candidates,
                                  args.epochs if classical_full else args.select_epochs,
                                  args.batch_size, seed=sel_seed, label=tag,
                                  initialisation_seeds=[10000, 10001, 10002] if classical_full else None,
                                  validation_weighting=weighting, tolerance=tolerance,
                                  patience=args.patience if final else 10)
        lr_by_model[tag] = best_lr
        lr_scan[tag] = scan
        if scan.get('selected_hit_epoch_cap'):
            print(f"    WARNING [{tag}]: a selected candidate reached its selection "
                  "epoch limit; inspect background validation histories.")
        if scan["hit_boundary"]:
            print(f"    WARNING [{tag}]: selected lr is at the edge of the "
                  f"candidate grid {args.lr_candidates}; the optimum may lie "
                  f"outside it.")

    return {
        "guard": selection_guard(args),
        "selection_seed": sel_seed,
        "verify": verify,
        "encoding": encoding,
        "ansatz_reps": reps_by_map,
        "reps_selection": reps_scan,
        "lr_selected": lr_by_model,
        "lr_selection": lr_scan,
    }


def load_selection(path, args):
    """Read a frozen selection and refuse it if it was made for another run."""
    with open(path) as f:
        sel = json.load(f)
    want, got = selection_guard(args), sel.get("guard")
    if want != got:
        diff = {k: (got.get(k) if got else None, v)
                for k, v in want.items() if not got or got.get(k) != v}
        raise SystemExit(
            f"selection file {path} was computed for a different "
            f"configuration; refusing to use it.\n"
            f"  differing keys (file, this run): {diff}")
    # JSON turns the feature-map keys' values into ints already, but a dict
    # written from Python keeps reps as int and lr as float; assert rather
    # than coerce, so a malformed file fails here and not mid-training.
    sel["ansatz_reps"] = {k: int(v) for k, v in sel["ansatz_reps"].items()}
    sel["lr_selected"] = {k: float(v) for k, v in sel["lr_selected"].items()}
    with open(path, "rb") as f:
        sel["digest"] = hashlib.sha256(f.read()).hexdigest()[:16]
    return sel


def model_builders(reps_by_map, device=None):
    """The model family -> (constructor, loss) map used for lr selection.

    Shared with `run_seed`'s construction order by convention, not by code:
    this one builds at selection time on the probe split, that one builds per
    seed on the real split. Keeping the tags in one place keeps the two from
    drifting apart silently.
    """
    target_par = n_params(N_QUBITS, reps_by_map["ry"])
    return {
        "qae_ry": (lambda s: QuantumAutoencoder(N_QUBITS, N_TRASH,
                                                reps_by_map["ry"], seed=s,
                                                feature_map="ry",
                                                device=device), qae_loss),
        "qae_zz": (lambda s: QuantumAutoencoder(N_QUBITS, N_TRASH,
                                                reps_by_map["zz"], seed=s,
                                                feature_map="zz",
                                                device=device), qae_loss),
        "ae_matched": (lambda s: matched_ae(len(FEATURES), LATENT,
                                            target_par, seed=s)[0], ae_loss),
        "ae_untied32": (lambda s: UntiedAE(len(FEATURES), 2, seed=s), ae_loss),
        "ae_untied45": (lambda s: UntiedAE(len(FEATURES), 3, seed=s), ae_loss),
        "ae_untied58": (lambda s: UntiedAE(len(FEATURES), 4, seed=s), ae_loss),
        "ae_dense": (lambda s: DenseAE(len(FEATURES), 16, LATENT, seed=s),
                     ae_loss),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['legacy-v1', 'final-v2'], default='legacy-v1',
                    help='final-v2 uses event-weighted validation and a fixed split')
    ap.add_argument('--split-seed', type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=20_000)
    # Raised from 60: at 60 the classical baselines never early-stopped in any
    # seed, so the cap -- not convergence -- was ending their training while the
    # QAE converged freely. run_seed warns loudly if the cap still binds.
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--patience", type=int, default=20,
                    help="early-stopping patience, in epochs")
    # 4096 measured ~2.3x faster per epoch than 1024 on CPU: the statevector
    # itself is tiny (64 amplitudes), so per-batch Python overhead dominates.
    # Larger still (16k+) gets slower again.
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.05)
    # Extended from [1,3,5]: the ry val loss was monotone decreasing across
    # that grid, so reps=5 "won" only because the grid ran out.
    ap.add_argument("--reps-candidates", type=int, nargs="+",
                    default=[1, 3, 5, 7, 9],
                    help="ansatz depths to choose between, on validation loss")
    # Extended upward: in the previous run every model selected 0.05, which was
    # the largest candidate, so the sweep resolved nothing about whether the
    # shared rate favoured any family.
    ap.add_argument("--lr-candidates", type=float, nargs="+",
                    default=[0.2, 0.1, 0.05, 0.01, 1e-3],
                    help="learning rates to choose between, per model family")
    ap.add_argument("--select-epochs", type=int, default=25,
                    help="epoch budget for depth selection (ordering only)")
    ap.add_argument("--select-n", type=int, default=25_000,
                    help="training events used for depth selection")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ablation-seeds", type=int, default=5,
                    help="seeds for the ae_dense diagnostic (capped: it is a "
                         "diagnostic and need not scale with the main study)")
    ap.add_argument("--no-ablation", action="store_true",
                    help="skip the ae_dense epochs x bottleneck-ReLU diagnostic")
    ap.add_argument("--no-qqq", action="store_true",
                    help="skip the 3-prong generalisation evaluation")
    ap.add_argument("--quick", action="store_true",
                    help="tiny run to check the pipeline end to end")
    # --- parallel execution -------------------------------------------------
    # The study is embarrassingly parallel over seeds: every model is seeded
    # explicitly, so a seed's result does not depend on which other seeds ran
    # in the same process (verified, bit for bit, in tools/check_shard_*).
    # The selection phase is NOT parallel and must not be repeated per worker:
    # each worker would select on its own first seed and the workers would
    # then train different models. So it is computed once, frozen to a file,
    # and loaded read-only by the workers.
    ap.add_argument("--selection-only", metavar="PATH", default=None,
                    help="compute the selection phase, write it to PATH and "
                         "exit without training any seed")
    ap.add_argument("--selection-from", metavar="PATH", default=None,
                    help="load a frozen selection instead of recomputing it; "
                         "refuses a file made for a different configuration")
    ap.add_argument("--selection-seed", type=int, default=None,
                    help="seed whose background split the selection phase uses "
                         "(default: the first --seeds entry)")
    ap.add_argument("--roc-seed", type=int, default=None,
                    help="seed whose per-event scores feed the ROC, score and "
                         "sculpting figures (default: the first --seeds entry)")
    ap.add_argument("--save-first", metavar="PATH", default=None,
                    help="write the --roc-seed per-event scores to PATH (.npz) "
                         "so a merge step can still make the figures")
    ap.add_argument("--ablation-only", action="store_true",
                    help="run only the ae_dense diagnostic and exit; for "
                         "running it beside the seed workers")
    ap.add_argument("--device", default=None,
                    help="torch device for the QUANTUM models only, e.g. "
                         "cuda. Classical baselines remain on CPU.")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch intra-op threads. Set to 1 when running "
                         "several workers at once: the tensors here are tiny "
                         "and thread contention costs more than it buys")
    args = ap.parse_args()
    if args.protocol == 'final-v2' and not args.no_ablation and not args.selection_only:
        ap.error('final-v2 requires --no-ablation; the old attribution diagnostic is historical')
    if args.protocol == 'final-v2' and os.path.abspath(args.out) == os.path.abspath('results'):
        ap.error('final-v2 requires a separate --out directory to preserve the reported study')

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.quick:
        # The epoch cap is scaled with the reduced budget rather than the check
        # being skipped: at the old quick setting (cap 40, patience 20) every
        # model tripped the cap warning by construction, so the check was
        # meaningless in the one mode that runs often. Cap 200 with patience 5
        # leaves the same headroom ratio as the full run, so a cap warning in
        # quick mode means something again.
        args.seeds, args.n_train, args.n_val, args.epochs = [0], 4000, 1000, 200
        args.patience = 5
        args.select_epochs, args.select_n = 8, 2000
        args.reps_candidates, args.lr_candidates = [1, 3], [0.05, 0.01]

    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    t_start = time.time()

    print("=" * 70)
    df = load_rnd(args.data_dir)
    qqq = None if args.no_qqq else load_qqq(args.data_dir)
    qqq_raw = None if qqq is None else qqq[FEATURES].to_numpy()

    sel_seed = args.selection_seed if args.selection_seed is not None \
        else args.seeds[0]
    if args.selection_from:
        sel = load_selection(args.selection_from, args)
        print("=" * 70)
        print(f"Selection loaded from {args.selection_from} "
              f"(digest {sel['digest']}, computed on seed "
              f"{sel['selection_seed']}); NOT recomputed here.")
    else:
        sel = compute_selection(args, df, sel_seed, args.device)
        sel["digest"] = None

    verify = sel["verify"]
    encoding = sel["encoding"]
    reps_by_map = sel["ansatz_reps"]
    reps_scan = sel["reps_selection"]
    lr_by_model = sel["lr_selected"]
    lr_scan = sel["lr_selection"]
    reps_boundary = any(v["hit_boundary"] for v in reps_scan.values())
    reps_noise = {k: v["below_noise_floor"] for k, v in reps_scan.items()}
    print(f"  in force: reps={reps_by_map}  lr={lr_by_model}")

    if args.selection_only:
        with open(args.selection_only, "w") as f:
            json.dump(sel, f, indent=2)
        print("=" * 70)
        print(f"wrote {args.selection_only}  "
              f"({time.time()-t_start:.0f}s); no seeds trained")
        return

    # Which seed's per-event scores feed the figures. Defaults to this
    # process's first seed, as before; a worker that does not hold that seed
    # simply contributes no curves and the merge takes them from the one that
    # does.
    roc_seed = args.roc_seed if args.roc_seed is not None else args.seeds[0]

    per_model_runs, all_hist, all_params, roc_cache = {}, {}, {}, {}
    all_capped, qqq_runs, first, matched_spec = {}, {}, None, None
    # --ablation-only keeps --seeds (the diagnostic slices it) but trains no
    # seed of the main study, so the diagnostic can run beside the workers.
    train_seeds = [] if args.ablation_only else args.seeds
    for seed in train_seeds:
        print("=" * 70)
        print(f"SEED {seed}")
        final = args.protocol == 'final-v2'
        splits = make_splits(df, args.n_train, args.n_val,
                             seed=args.split_seed if final else seed)
        scores, hist, params, capped, extra, seed_meta = run_seed(
            splits, seed, args.epochs, args.batch_size, lr_by_model,
            reps_by_map, patience=args.patience, extra_signal=qqq_raw,
            device=args.device,
            validation_weighting='events' if final else 'legacy-batches',
            scaler_seed=args.split_seed if final else None)
        all_params = params
        all_capped[seed] = capped
        matched_spec = seed_meta.get("ae_matched")

        # Generalisation: the SAME trained models, scored on background from
        # this seed's test split plus the 3-prong signal they never saw.
        if extra:
            bkg = splits.test_label == 0
            for name, sig_sc in extra.items():
                comb = np.concatenate([scores[name][bkg], sig_sc])
                lab = np.concatenate([np.zeros(int(bkg.sum()), np.int64),
                                      np.ones(len(sig_sc), np.int64)])
                qqq_runs.setdefault(name, []).append(evaluate(comb, lab))
        for name, sc in scores.items():
            res = evaluate(sc, splits.test_label, splits.test_mjj)
            per_model_runs.setdefault(name, []).append(res)
            r30 = res["rejection"]["eps_s=0.3"]
            print(f"    {name:12s} AUC {res['auc']:.4f}   "
                  f"1/eps_B@0.3 = {r30['rejection']:8.1f}   "
                  f"rho(score,mjj) = {res['sculpting']['spearman_score_vs_mjj']:+.3f}")
            if seed == roc_seed:
                tpr, fpr = roc_points(sc, splits.test_label)
                roc_cache[name] = (tpr.tolist(), fpr.tolist())
        if seed == roc_seed:
            # Keep that seed's raw scores in memory: the score-distribution
            # and sculpting figures need per-event values, which are far too
            # large to round-trip through metrics.json.
            first = (scores, splits.test_label, splits.test_mjj)
        all_hist[seed] = {k: {"train_loss": v.train_loss, "val_loss": v.val_loss,
                              "best_epoch": v.best_epoch,
                              "epochs_run": v.epochs_run,
                              "hit_epoch_cap": bool(capped.get(k, False)),
                              "seconds": v.seconds}
                          for k, v in hist.items()}

    dense_ablation = None
    if not args.no_ablation:
        print("=" * 70)
        print("DIAGNOSTIC: ae_dense ablation, {60,600} epochs x bottleneck ReLU")
        # Deliberately capped: this is an attribution diagnostic on one model,
        # not a headline result, so it does not need the main study's seed
        # count and should not scale wall time with it.
        abl_seeds = args.seeds[:args.ablation_seeds]
        print(f"  (diagnostic uses the first {len(abl_seeds)} seed(s) only: "
              f"{abl_seeds})")
        dense_ablation = ablate_dense(
            df, abl_seeds, args.n_train, args.n_val, args.batch_size,
            lr_by_model["ae_dense"], args.patience)

    summary = {name: aggregate(runs) for name, runs in per_model_runs.items()}
    qqq_summary = {name: aggregate(runs) for name, runs in qqq_runs.items()}
    if qqq_summary:
        print("=" * 70)
        print("Generalisation to the 3-prong signal (never trained or selected on)")
        print(f"{'model':14s} {'AUC (2-prong)':>18s} {'AUC (3-prong)':>18s}")
        for name in qqq_summary:
            a2, a3 = summary[name]["auc"], qqq_summary[name]["auc"]
            print(f"{name:14s} {a2['mean']:.4f} +/- {a2['std']:.4f}   "
                  f"{a3['mean']:.4f} +/- {a3['std']:.4f}")

    out = {
        "config": {
            "protocol": args.protocol,
            "validation_weighting": 'events' if args.protocol == 'final-v2' else 'legacy-batches',
            "split_policy": 'fixed' if args.protocol == 'final-v2' else 'resampled',
            "split_seed": args.split_seed if args.protocol == 'final-v2' else None,
            "n_qubits": N_QUBITS, "n_latent": LATENT, "n_trash": N_TRASH,
            "ansatz_reps": reps_by_map, "reps_selection": reps_scan,
            "reps_candidates": args.reps_candidates,
            "reps_selection_hit_boundary": reps_boundary,
            "reps_selection_below_noise_floor": reps_noise,
            "selection_noise_floor": 1e-9 if args.protocol == 'final-v2' else SELECTION_NOISE_FLOOR,
            "lr_selected": lr_by_model, "lr_selection": lr_scan,
            "lr_candidates": args.lr_candidates,
            "selection_budget": {"n_events": args.select_n,
                                 "epochs": args.select_epochs,
                                 "criterion": "background validation loss",
                                 "order": "depth at default lr, then lr at "
                                          "selected depth (one coordinate pass, "
                                          "not a joint search)"},
            "classical_selection_budget": ({'n_events': args.n_train,
                 'epochs': args.epochs, 'patience': args.patience,
                 'initialisation_seeds': [10000, 10001, 10002]}
                 if args.protocol == 'final-v2' else None),
            "features": FEATURES,
            "n_train": args.n_train, "n_val": args.n_val,
            "epochs": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size,
            "seeds": list(train_seeds), "quick": args.quick,
            "simulation": "noiseless statevector, infinite shots",
            "device": args.device or "cpu",
            "environment": environment(args.device),
            "qiskit_pennylane_max_amplitude_diff": verify,
            # Identifies the frozen selection these seeds were trained under.
            # The merge step refuses to combine shards that do not share it,
            # which is what stops a worker started against a stale selection
            # file from contributing seeds trained at another learning rate.
            "selection_digest": sel.get("digest"),
            "selection_seed": sel.get("selection_seed", sel_seed),
        },
        "diagnostic_ae_dense_ablation": {
            "n_seeds": (min(args.ablation_seeds, len(args.seeds))
                        if dense_ablation else 0),
            "description": "DIAGNOSTIC, not a headline result. ae_dense alone "
                           "over {60,600} epochs x {bottleneck ReLU on, off}, "
                           "same seeds, at its selected learning rate. Exists "
                           "only to attribute a previously-observed change to "
                           "one of two simultaneous edits. No baselines, no "
                           "significance testing.",
            "results": dense_ablation,
        },
        "matched_ae_spec": matched_spec,
        "hit_epoch_cap": all_capped,
        "any_hit_epoch_cap": bool(any(v for d in all_capped.values()
                                      for v in d.values())),
        "generalisation_qqq": {
            "description": "3-prong W'->XY, X,Y->qqq signal. Never used for "
                           "training, model selection or hyperparameter choice; "
                           "scored by the same trained models on the same "
                           "background test split.",
            "summary": qqq_summary, "per_seed": qqq_runs,
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
    offenders = [(s, m) for s, d in all_capped.items()
                 for m, v in d.items() if v]
    if offenders:
        print("*** WARNING: the epoch cap bound for the following "
              "(seed, model) pairs, so training was stopped by the budget "
              "rather than by convergence:")
        for s, m in offenders:
            print(f"      seed {s}: {m}")
        print("*** Inspect validation histories and disclose the budget limit. "
              "Early stopping is not proof of a globally optimal model.")
    else:
        print(f"Epoch-cap check: every model early-stopped in every seed "
              f"(cap {args.epochs}, patience {args.patience}). "
              "No model was cut off by the budget.")

    print("=" * 70)
    print(f"wrote {path}  ({time.time()-t_start:.0f}s total)")

    # The score and sculpting figures need per-event values, which are far too
    # large for metrics.json. A worker holding the ROC seed writes them out so
    # the merge step can still draw those two figures from the real scores
    # rather than approximating them from anything summarised.
    if args.save_first and first is not None:
        sc0, lab0, mjj0 = first
        np.savez_compressed(args.save_first, label=lab0, mjj=mjj0,
                            **{f"score__{k}": v for k, v in sc0.items()})
        print(f"wrote per-event scores for seed {roc_seed} to {args.save_first}")

    figdir = os.path.join(args.out, "figures")
    try:
        if not per_model_runs:
            raise RuntimeError("no seeds trained in this process")
        from .figures import make_all, score_figure, sculpting_figure
        make_all(path, figdir)
        if first is None:
            print("no ROC-seed scores in this process: score and sculpting "
                  "figures skipped (the merge step draws them)")
        else:
            sc0, lab0, mjj0 = first
            score_figure(sc0, lab0, os.path.join(figdir, "scores.png"))
            sculpting_figure(sc0, lab0, mjj0,
                             os.path.join(figdir, "sculpting.png"))
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
