"""Run the full study on a Colab GPU, in resumable chunks.

Measured on a Tesla T4: 1.69 s/epoch against 12.2 s/epoch on the laptop this
study was developed on, a 7.2x speed-up that turns a 6.5 h run into roughly
one hour. The GPU is used for the two quantum models only; the classical
baselines train in 0.03 s each and stay on the CPU, where moving them would
cost more in transfers than it saves.

Chunking is for RESUMABILITY, not speed. A free-tier Colab session can be
reclaimed at any point, and a single process that writes its results only at
the end loses everything when that happens. Each chunk writes its own
metrics.json, so a session that dies at seed 9 costs one chunk, not the run;
re-running this script skips chunks that already finished and merges what
exists. On one GPU the chunks are deliberately run one after another --
parallelism across processes was measured to be a net LOSS even on CPU.

Usage in a Colab cell, after uploading and unzipping the repo:

    !pip -q install pennylane qiskit
    %cd /content/repo
    !python tools/colab_run.py --seeds 15

Then download results/metrics.json and results/figures/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=HERE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15,
                    help="number of seeds, 0..N-1")
    ap.add_argument("--chunk", type=int, default=5,
                    help="seeds per resumable chunk")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true",
                    help="redo chunks that already have a metrics.json")
    ap.add_argument("--save-scores", action="store_true",
                    help="also dump the ROC seed's raw per-event scores "
                         "(~80 MB) so the per-event figures can be redrawn "
                         "off this machine")
    # Anything else is handed straight to src.run_study, so this wrapper
    # never has to grow a mirror of its option list (and cannot silently
    # disagree with it).
    args, passthrough = ap.parse_known_args()
    if passthrough:
        print(f"passing through to src.run_study: {' '.join(passthrough)}")

    py = sys.executable
    seeds = list(range(args.seeds))
    work = os.path.join(args.out, "parallel")
    os.makedirs(work, exist_ok=True)
    sel = os.path.join(work, "selection.json")
    base = ([py, "-u", "-m", "src.run_study", "--device", args.device]
            + passthrough)

    try:
        import torch
        print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
              f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
        if args.device == "cuda" and not torch.cuda.is_available():
            print("\nERROR: --device cuda but no GPU is visible. "
                  "Runtime > Change runtime type > T4 GPU, then re-run.")
            return 2
    except ImportError:
        pass

    t0 = time.time()

    # 1. Selection, once. Everything downstream loads it read-only, so every
    #    chunk trains the same models at the same learning rates. Nothing here
    #    touches the test split or either signal sample.
    if args.force or not os.path.exists(sel):
        if sh(base + ["--seeds", "0", "--selection-only", sel,
                      "--out", os.path.join(work, "sel")]):
            return 1
    else:
        print(f"\nselection already present at {sel}; not recomputed")

    # 2. Seed chunks, each resumable.
    chunks, shard_paths = [], []
    for i in range(0, len(seeds), args.chunk):
        part = seeds[i:i + args.chunk]
        d = os.path.join(work, f"shard{part[0]:02d}")
        chunks.append((part, d))
        shard_paths.append(os.path.join(d, "metrics.json"))

    for part, d in chunks:
        done = os.path.join(d, "metrics.json")
        if os.path.exists(done) and not args.force:
            print(f"\nchunk {part}: already done ({done}); skipping")
            continue
        cmd = base + ["--seeds", *map(str, part), "--selection-from", sel,
                      "--roc-seed", "0", "--no-ablation", "--out", d]
        # The chunk holding the ROC seed has that seed's per-event scores in
        # memory and draws the two figures that need them itself; they are
        # copied out after the merge. --save-scores additionally dumps the
        # raw scores (about 80 MB) for redrawing them elsewhere.
        if seeds[0] in part and args.save_scores:
            cmd += ["--save-first", os.path.join(work, "first_seed_scores.npz")]
        if sh(cmd):
            print(f"\nchunk {part} FAILED. Re-run this script to retry it; "
                  f"finished chunks are kept.")
            return 1

    # 3. The ae_dense diagnostic, separately: it is capped at 5 seeds and does
    #    not scale with the study, so it is not worth a chunk of its own size.
    abl = os.path.join(work, "ablation")
    if args.force or not os.path.exists(os.path.join(abl, "metrics.json")):
        if sh(base + ["--seeds", *map(str, seeds), "--selection-from", sel,
                      "--ablation-only", "--out", abl]):
            return 1
    else:
        print("\nablation already done; skipping")

    # 4. Stitch, then draw.
    if sh([py, "tools/merge_shards.py", "--shards", *shard_paths,
           "--selection", sel,
           "--ablation", os.path.join(abl, "metrics.json"),
           "--out", os.path.join(args.out, "metrics.json")]):
        return 1
    figdir = os.path.join(args.out, "figures")
    if sh([py, "-m", "src.figures",
           "--results", os.path.join(args.out, "metrics.json"),
           "--outdir", figdir]):
        print("WARNING: figures failed; metrics.json is intact")

    # The score and sculpting figures are drawn by the chunk that holds the
    # ROC seed, from per-event values that never reach metrics.json. Copy
    # them beside the rest rather than redrawing from anything summarised.
    import shutil
    roc_chunk = chunks[0][1]
    for name in ("scores.png", "sculpting.png"):
        src_png = os.path.join(roc_chunk, "figures", name)
        if os.path.exists(src_png):
            shutil.copy2(src_png, os.path.join(figdir, name))
            print(f"copied {name} from the ROC-seed chunk")
        else:
            print(f"WARNING: {src_png} missing; {name} not updated")

    print(f"\n{'='*70}\ndone in {(time.time()-t0)/60:.0f} min. "
          f"Download {args.out}/metrics.json and {args.out}/figures/.")
    print("The score and sculpting figures need the per-event scores in "
          f"{work}/first_seed_scores.npz; draw them locally with "
          "tools/draw_event_figures.py if you want them from this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
