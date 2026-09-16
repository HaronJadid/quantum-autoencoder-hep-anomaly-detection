"""Combine per-seed shards of a parallel run into one results/metrics.json.

The study is embarrassingly parallel over seeds -- every model is seeded
explicitly, so a seed's numbers do not depend on which other seeds shared its
process -- but the SELECTION phase is not, and must not be repeated per
worker. The run is therefore:

    1. one process computes the selection and freezes it to selection.json;
    2. N workers each train a disjoint subset of seeds, loading that file
       read-only (--selection-from) and writing their own metrics.json;
    3. this script stitches the shards back into the single file the report
       and figures consume.

What it refuses to do, loudly, rather than papering over:

  * combine shards carrying different `selection_digest` values -- that would
    mean seeds trained under different learning rates, silently averaged;
  * combine shards that disagree on the model set, the parameter counts or
    the matched-AE spec;
  * count a seed twice, or emit seeds in an order that differs between
    models. The paired tests in src/report.py pair seed i of model A with
    seed i of model B, so a per-model ordering mismatch would silently
    scramble every paired comparison. Seeds are therefore re-emitted in
    ascending order for every model, from an explicit seed -> result map.

    python tools/merge_shards.py --shards results/shard_*/metrics.json \\
        --selection results/selection.json --ablation results/abl/metrics.json \\
        --out results/metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import aggregate  # noqa: E402


def die(msg: str) -> None:
    raise SystemExit(f"merge_shards: {msg}")


def load(paths):
    shards = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            shards.append((p, json.load(f)))
    if not shards:
        die("no shards given")
    return shards


def check_consistent(shards):
    """Everything that must be identical across shards, checked before merging."""
    ref_path, ref = shards[0]
    digest = ref["config"].get("selection_digest")
    if digest is None:
        die(f"{ref_path} has no selection_digest: it was not run with "
            f"--selection-from, so it may have selected its own "
            f"hyperparameters. Refusing to merge.")
    fields = {
        "protocol": lambda d: d['config'].get('protocol', 'legacy-v1'),
        "validation_weighting": lambda d: d['config'].get('validation_weighting', 'legacy-batches'),
        "split_policy": lambda d: d['config'].get('split_policy', 'resampled'),
        "split_seed": lambda d: d['config'].get('split_seed'),
        "selection_digest": lambda d: d["config"]["selection_digest"],
        "ansatz_reps": lambda d: d["config"]["ansatz_reps"],
        "lr_selected": lambda d: d["config"]["lr_selected"],
        "epochs": lambda d: d["config"]["epochs"],
        "patience": lambda d: d["config"]["patience"],
        "batch_size": lambda d: d["config"]["batch_size"],
        "n_train": lambda d: d["config"]["n_train"],
        "n_val": lambda d: d["config"]["n_val"],
        "features": lambda d: d["config"]["features"],
        "parameter_counts": lambda d: d["parameter_counts"],
        "matched_ae_spec": lambda d: d["matched_ae_spec"],
        "model_set": lambda d: sorted(d["per_seed"]),
    }
    for name, get in fields.items():
        want = get(ref)
        for path, d in shards[1:]:
            got = get(d)
            if got != want:
                die(f"shards disagree on {name}:\n"
                    f"  {ref_path}: {want}\n  {path}: {got}")
    return digest


def seed_order(shards):
    """Global ascending seed order, refusing duplicates."""
    seen = {}
    for path, d in shards:
        for s in d["config"]["seeds"]:
            if s in seen:
                die(f"seed {s} appears in both {seen[s]} and {path}")
            seen[s] = path
    if not seen:
        die("the shards contain no trained seeds")
    return sorted(seen), seen


def gather(shards, block, seeds):
    """seed -> result per model, re-emitted in ascending seed order.

    `block` is "per_seed" (in-distribution) or the 3-prong equivalent. A shard
    lists its results in the order it trained them, which is the order of its
    own config["seeds"], so that is the key used to attach a seed to a result.
    """
    by_model = {}
    for path, d in shards:
        src = d[block] if block == "per_seed" else d["generalisation_qqq"]["per_seed"]
        for model, runs in src.items():
            order = d["config"]["seeds"]
            if len(runs) != len(order):
                die(f"{path}: {block}/{model} has {len(runs)} results for "
                    f"{len(order)} seeds")
            for s, r in zip(order, runs):
                by_model.setdefault(model, {})[s] = r
    out = {}
    for model, per in by_model.items():
        missing = [s for s in seeds if s not in per]
        if missing:
            die(f"{block}/{model} is missing seeds {missing}")
        out[model] = [per[s] for s in seeds]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True,
                    help="shard metrics.json paths (globs allowed)")
    ap.add_argument("--selection", required=True,
                    help="the frozen selection.json the shards were run under")
    ap.add_argument("--ablation", default=None,
                    help="metrics.json from the --ablation-only process")
    ap.add_argument("--out", default="results/metrics.json")
    args = ap.parse_args()

    paths = sorted({p for pat in args.shards for p in glob.glob(pat)} or
                   set(args.shards))
    shards = load(paths)
    digest = check_consistent(shards)

    with open(args.selection, "rb") as f:
        import hashlib
        sel_digest = hashlib.sha256(f.read()).hexdigest()[:16]
    if sel_digest != digest:
        die(f"--selection {args.selection} has digest {sel_digest} but the "
            f"shards were trained under {digest}")

    seeds, owner = seed_order(shards)
    print(f"merging {len(shards)} shard(s), {len(seeds)} seed(s): {seeds}")
    for p, d in shards:
        print(f"  {os.path.relpath(p):45s} seeds {d['config']['seeds']}")

    per_seed = gather(shards, "per_seed", seeds)
    qqq_per_seed = gather(shards, "qqq", seeds) if any(
        d["generalisation_qqq"]["per_seed"] for _, d in shards) else {}

    # The ROC curves and the per-event figures come from one seed. Take them
    # from whichever shard holds it; require exactly one so the figure cannot
    # silently come from a different seed than the caption claims.
    with_roc = [(p, d) for p, d in shards if d["roc_first_seed"]]
    if len(with_roc) != 1:
        die(f"expected exactly one shard with ROC curves, found "
            f"{len(with_roc)}: {[p for p, _ in with_roc]}. Run the workers "
            f"with a single --roc-seed held by one of them.")
    roc_path, roc_shard = with_roc[0]
    roc_seed = roc_shard["config"]["seeds"][0]

    out = dict(shards[0][1])
    out["config"] = dict(shards[0][1]["config"])
    out["config"]["seeds"] = seeds
    out["config"]["parallel"] = {
        "shards": {os.path.relpath(p): d["config"]["seeds"] for p, d in shards},
        "selection_digest": digest,
        "note": "seeds were trained in separate worker processes under one frozen "
                "selection; each seed's result is independent of which "
                "process trained it, every model being explicitly seeded.",
    }
    out["per_seed"] = per_seed
    out["summary"] = {k: aggregate(v) for k, v in per_seed.items()}
    out["generalisation_qqq"] = dict(shards[0][1]["generalisation_qqq"])
    out["generalisation_qqq"]["per_seed"] = qqq_per_seed
    out["generalisation_qqq"]["summary"] = {k: aggregate(v)
                                            for k, v in qqq_per_seed.items()}
    out["histories"] = {k: v for _, d in shards
                        for k, v in d["histories"].items()}
    out["hit_epoch_cap"] = {k: v for _, d in shards
                            for k, v in d["hit_epoch_cap"].items()}
    out["any_hit_epoch_cap"] = bool(any(v for d in out["hit_epoch_cap"].values()
                                        for v in d.values()))
    out["roc_first_seed"] = roc_shard["roc_first_seed"]
    out["wall_seconds"] = max(d["wall_seconds"] for _, d in shards)
    out["wall_seconds_summed"] = sum(d["wall_seconds"] for _, d in shards)

    if args.ablation:
        with open(args.ablation, encoding="utf-8") as f:
            abl = json.load(f)
        if abl["config"].get("selection_digest") != digest:
            die(f"--ablation was run under selection digest "
                f"{abl['config'].get('selection_digest')}, not {digest}")
        out["diagnostic_ae_dense_ablation"] = abl["diagnostic_ae_dense_ablation"]
        out["wall_seconds"] = max(out["wall_seconds"], abl["wall_seconds"])
        out["wall_seconds_summed"] += abl["wall_seconds"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    n = len(seeds)
    print(f"\nwrote {args.out}: {n} seeds, {len(per_seed)} models, "
          f"ROC curves from seed {roc_seed} ({os.path.relpath(roc_path)})")
    abl = (out.get("diagnostic_ae_dense_ablation") or {}).get("results")
    print(f"  ablation: {'present' if abl else 'ABSENT'}")
    print(f"  3-prong : {len(qqq_per_seed)} models")
    print(f"  worker wall time: {out['wall_seconds']/60:.0f} min longest, "
          f"{out['wall_seconds_summed']/60:.0f} min summed; "
          "neither includes the complete selection/orchestration time")
    for name in sorted(out["summary"]):
        a = out["summary"][name]["auc"]
        print(f"    {name:14s} AUC {a['mean']:.4f} +/- {a['std']:.4f} "
              f"(n={a['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
