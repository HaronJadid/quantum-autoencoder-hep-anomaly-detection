"""Re-run a few seeds and diff them against the reported run, per model.

The reproducibility claim in this repository is not uniform, and a single
pass/fail would misrepresent it. Two quantum models, PCA and the m_J1 control
reproduce across machines to every decimal place reported; the five classical
autoencoders do not, because their optimisation is chaotic at the selected
learning rates (tools/diagnose_training_chaos.py demonstrates this directly).
This script makes both halves of that statement executable: it trains the
requested seeds, compares each model against the frozen snapshot in
results/reference/reported_run.json, and prints which matched and by how much
the rest drifted.

    python tools/verify_reproduction.py                    # seeds 0 1 2
    python tools/verify_reproduction.py --seeds 2 5        # pick seeds
    python tools/verify_reproduction.py --device cuda      # on a GPU
    python tools/verify_reproduction.py --skip-run --metrics results/metrics.json

Exit code is 0 if every model that is expected to reproduce did so, and 1 if
one of them drifted -- drift in the classical autoencoders is expected and
does NOT fail the check, because it is the documented behaviour.

Hyperparameters come from results/reference/selection.json, the selection the
reported numbers were trained under, so this verifies the seed loop rather
than re-deriving the selection (which takes ~20 minutes and is separately
known to be platform-independent: the CPU and GPU runs selected identically).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join("results", "reference", "reported_run.json")
SELECTION = os.path.join("results", "reference", "selection.json")

# Agreement at the precision the study reports (4 decimal places on AUC).
# Tighter than this is reported as bit-identical; looser is drift.
TOL = 5e-5


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="seeds to re-run (default: 0 1 2)")
    ap.add_argument("--device", default=None,
                    help="passed to src.run_study for the quantum models")
    ap.add_argument("--out", default=os.path.join("results", "verify"))
    ap.add_argument("--skip-run", action="store_true",
                    help="do not train; diff an existing metrics.json")
    ap.add_argument("--metrics", default=None,
                    help="metrics.json to diff (default: <out>/metrics.json)")
    args = ap.parse_args()

    ref_path = os.path.join(HERE, REFERENCE)
    if not os.path.exists(ref_path):
        print(f"missing {REFERENCE}; nothing to verify against")
        return 2
    ref = load(ref_path)
    metrics = args.metrics or os.path.join(args.out, "metrics.json")

    if not args.skip_run:
        sel = os.path.join(HERE, SELECTION)
        if not os.path.exists(sel):
            print(f"missing {SELECTION}; cannot pin the reported selection")
            return 2
        cmd = [sys.executable, "-u", "-m", "src.run_study",
               "--seeds", *map(str, args.seeds),
               "--selection-from", SELECTION,
               "--no-ablation", "--out", args.out]
        if args.device:
            cmd += ["--device", args.device]
        print("$ " + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=HERE)
        if rc:
            print(f"\nrun failed (exit {rc}); nothing to compare")
            return rc

    got = load(os.path.join(HERE, metrics))
    order = got["config"]["seeds"]
    common = [s for s in order if any(e["seed"] == s
                                      for e in next(iter(ref["per_seed"].values())))]
    if not common:
        print("no seeds in common with the reference snapshot")
        return 2

    expected = set(ref["reproduces_exactly"])
    print()
    print("=" * 78)
    print(f"Reproduction check against {REFERENCE}")
    # Describe the CANDIDATE FILE, not the interpreter running this script.
    # With --skip-run the file may come from another machine entirely, and
    # pairing its device with the local torch version invents an environment
    # that never existed. Runs record config.environment; fall back to the
    # local interpreter only when diffing a file that predates that.
    env = (got["config"].get("environment") or {})
    cand_torch = env.get("torch") or f"{__import__('torch').__version__} (this interpreter; not recorded in the file)"
    print(f"  reference run : {ref['produced_by']['device']}, "
          f"torch {ref['produced_by']['torch']}")
    print(f"  candidate     : {env.get('device') or got['config'].get('device', 'cpu')}, "
          f"torch {cand_torch}")
    print(f"  candidate file: {metrics}")
    print(f"  seeds compared: {common}")
    print("=" * 78)
    # Scientific notation, not fixed decimals: a difference of 1e-16 is the
    # interesting answer here, and %f would print it as 0.000000 next to a
    # verdict that says it is not bit-identical, which reads as a bug.
    print(f"{'model':14s} {'expected':>10s} {'max |dAUC|':>11s} "
          f"{'max |d 1/eB|':>13s}  verdict")

    failures, drifted = [], []
    for name in sorted(got["per_seed"]):
        if name not in ref["per_seed"]:
            continue
        by_seed = {e["seed"]: e for e in ref["per_seed"][name]}
        da = dr = 0.0
        for s in common:
            g = got["per_seed"][name][order.index(s)]
            r = by_seed[s]
            da = max(da, abs(g["auc"] - r["auc"]))
            dr = max(dr, abs(g["rejection"]["eps_s=0.3"]["rejection"]
                             - r["rejection_eps_s_0.3"]))
        must = name in expected
        if da == 0.0:
            verdict = "EXACT (bit for bit)"
        elif da < 1e-12:
            # A few ulp of float64: the two machines did the same arithmetic
            # in a different order. Nothing reported here resolves this far.
            verdict = "identical to floating-point noise"
        elif da < TOL:
            verdict = "matches at reported precision"
        else:
            verdict = "DRIFTED"
            (failures if must else drifted).append((name, da))
        print(f"{name:14s} {'exact' if must else 'drift ok':>10s} "
              f"{da:11.2e} {dr:13.2f}  {verdict}")

    print()
    if drifted:
        print("Drift in the classical autoencoders is expected and documented: "
              "their training is chaotic, so a rounding difference between "
              "machines changes the trajectory. See "
              "tools/diagnose_training_chaos.py and the Limitations section.")
    if failures:
        print("FAIL: model(s) that should reproduce exactly did not:")
        for n, d in failures:
            print(f"  {n}: max |dAUC| = {d:.6f} (tolerance {TOL})")
        print("That is a real discrepancy -- investigate before quoting these "
              "numbers. Check that the selection in use matches the reference "
              "and that no experimental code has changed.")
        return 1
    print(f"PASS: every model expected to reproduce did, to within {TOL} AUC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
