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
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    ap.add_argument("--selection-from", default=None,
                    help="copy and use a frozen selection rather than selecting again")
    ap.add_argument("--no-ablation", action="store_true",
                    help="skip the separate classical diagnostic")
    # Anything else is handed straight to src.run_study, so this wrapper
    # never has to grow a mirror of its option list (and cannot silently
    # disagree with it).
    args, passthrough = ap.parse_known_args()
    if args.seeds < 1 or args.chunk < 1:
        ap.error("--seeds and --chunk must be positive")
    if any(arg.split('=')[0] in {'--selection-only', '--ablation-only',
                                '--roc-seed', '--save-first', '--quick'}
           for arg in passthrough):
        ap.error("use wrapper options for selection, scores and seed counts; "
                 "--quick is not compatible with resumable full-study chunks")
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
        print("ERROR: torch is missing; install dependencies first")
        return 2

    # Resuming is valid only for the same code, settings and environment.
    # Old folders without a manifest must not be silently adopted.
    import importlib.metadata as md
    source_hash = hashlib.sha256()
    for path in sorted(Path(HERE, 'src').glob('*.py')):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    source_hash.update(Path(__file__).read_bytes())
    frozen = Path(args.selection_from).resolve() if args.selection_from else None
    signature = {
        'seeds': args.seeds, 'chunk': args.chunk, 'device': args.device,
        'save_scores': args.save_scores, 'no_ablation': args.no_ablation,
        'passthrough': passthrough, 'source_sha256': source_hash.hexdigest(),
        'selection_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest()
                            if frozen else None,
        'python': sys.version,
        'packages': {name: md.version(name) for name in
                     ('torch', 'numpy', 'pennylane', 'qiskit', 'scikit-learn',
                      'scipy', 'pandas', 'tables', 'matplotlib', 'pylatexenc')},
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'threads': torch.get_num_threads(),
    }
    manifest = Path(work, 'run_manifest.json')
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature and not args.force:
            print('ERROR: resume settings/code/environment differ. Use a new '
                  '--out directory, or --force to recompute this run.')
            return 2
    elif any(Path(work).iterdir()) and not args.force:
        print('ERROR: existing run has no resume manifest. Use a new --out directory.')
        return 2
    manifest.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    if frozen and (args.force or not Path(sel).exists()):
        if frozen != Path(sel).resolve():
            shutil.copy2(frozen, sel)
    if frozen and Path(sel).read_bytes() != frozen.read_bytes():
        raise ValueError('cached selection differs from the requested frozen selection')

    t0 = time.time()

    # 1. Selection, once. Everything downstream loads it read-only, so every
    #    chunk trains the same models at the same learning rates. Nothing here
    #    touches the test split or either signal sample.
    if not frozen and (args.force or not os.path.exists(sel)):
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
            cached = json.loads(Path(done).read_text())
            digest = hashlib.sha256(Path(sel).read_bytes()).hexdigest()[:16]
            if (cached['config']['seeds'] != part or
                    cached['config'].get('selection_digest') != digest):
                raise ValueError(f'cached chunk {done} has different seeds or selection')
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
    if args.no_ablation:
        print('Separate ablation skipped by request')
    elif args.force or not os.path.exists(os.path.join(abl, "metrics.json")):
        if sh(base + ["--seeds", *map(str, seeds), "--selection-from", sel,
                      "--ablation-only", "--out", abl]):
            return 1
    else:
        print("\nablation already done; skipping")

    # 4. Stitch, then draw.
    ablation_args = ([] if args.no_ablation else
                     ['--ablation', os.path.join(abl, 'metrics.json')])
    if sh([py, "tools/merge_shards.py", "--shards", *shard_paths,
           "--selection", sel, *ablation_args,
           "--out", os.path.join(args.out, "metrics.json")]):
        return 1
    figdir = os.path.join(args.out, "figures")
    if sh([py, "-m", "src.figures",
           "--results", os.path.join(args.out, "metrics.json"),
           "--outdir", figdir]):
        print("WARNING: figures failed; metrics.json is intact")
        return 1

    # The score and sculpting figures are drawn by the chunk that holds the
    # ROC seed, from per-event values that never reach metrics.json. Copy
    # them beside the rest rather than redrawing from anything summarised.
    roc_chunk = chunks[0][1]
    missing = []
    for name in ("scores.png", "sculpting.png"):
        src_png = os.path.join(roc_chunk, "figures", name)
        if os.path.exists(src_png):
            shutil.copy2(src_png, os.path.join(figdir, name))
            print(f"copied {name} from the ROC-seed chunk")
        else:
            print(f"WARNING: {src_png} missing; {name} not updated")
            missing.append(name)
    if missing:
        print('ERROR: metrics are intact, but per-event figures are incomplete. '
              'Use saved scores to redraw them, or rerun with --force.')
        return 1

    print(f"\n{'='*70}\ndone in {(time.time()-t0)/60:.0f} min. "
          f"Download {args.out}/metrics.json and {args.out}/figures/.")
    print("The score and sculpting figures need the per-event scores in "
          f"{work}/first_seed_scores.npz; draw them locally with "
          "tools/draw_event_figures.py if you want them from this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
