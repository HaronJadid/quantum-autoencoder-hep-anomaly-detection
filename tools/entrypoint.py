"""Container entrypoint: pin the thread count, then run what was asked.

Thread count is not cosmetic here. torch's reduction order depends on how many
intra-op threads it uses, so the same code on the same machine gives different
last-bit results at 1 thread and at 6, and the classical autoencoders in this
study amplify a last-bit difference into an AUC difference of order 0.1 (see
tools/diagnose_training_chaos.py). Fixing the count to 1 inside the container
means two runs of the same image on machines with different core counts at
least start from the same arithmetic, instead of silently diverging because
one host had more cores than the other.

Three settings, with different reach, measured in the container rather than
assumed:

  OMP_NUM_THREADS=1 (set in the image)   reaches every process, including
                                         subprocesses; pins the intra-op pool
  torch.set_num_threads(1)               this process
  torch.set_num_interop_threads(1)       this process, and only if called
                                         before set_num_threads

So a command run IN-PROCESS (`smoke`, `verify`, anything of the form
`python -m mod` or `python script.py`) gets intra-op 1 and inter-op 1. A
command handed to a subprocess (`python -c ...`, a shell, a non-python
program) gets intra-op 1 from the environment variable but keeps the host's
inter-op count, because torch has no environment variable for that pool.
Inter-op parallelism schedules independent ops and does not affect the
reduction order inside one, so it is not the setting that moves the last
bits -- but the difference is real and is stated rather than papered over.

    docker run --rm qae-lhco                          # this help
    docker run --rm qae-lhco smoke                    # ~2 min pipeline check
    docker run --rm qae-lhco verify                   # reproduce 3 seeds
    docker run --rm qae-lhco python -m src.run_study --seeds 0 1 2
"""

from __future__ import annotations

import os
import runpy
import sys

# Running this file directly makes sys.path[0] the tools/ directory, so
# `import src...` fails even though the working directory is the repo root.
# Put the repo root first, and make it the working directory, so every command
# below behaves the same whether it is run in-process or as a subprocess.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

USAGE = """
LHCO quantum-autoencoder study -- container image
=================================================

This image does NOT run the full study by default. The full study is hours of
compute and writes over results/, so it has to be asked for explicitly.

  smoke          Tiny end-to-end check of the pipeline (~2 minutes, downloads
                 the 74 MB dataset on first run). Trains one seed at reduced
                 size and prints a results table. Proves the image works; the
                 numbers are not meaningful. Writes to results/smoke/, never
                 over results/metrics.json.

  verify         Re-run seeds 0-2 at the reported settings and diff them
                 against results/legacy-v1/reference/reported_run.json, per model.
                 Prints which models reproduced exactly and how far the others
                 drifted. Takes roughly an hour on CPU.

  <command...>   Anything else is executed as given, e.g.
                   python -m src.run_study --seeds 0 1 2
                   python -m src.report results/final-v2/metrics.json
                   python tools/diagnose_training_chaos.py

Mount both volumes so the dataset is cached and the results survive the
container:

  docker run --rm -v "$(pwd)/results:/app/results" \\
                  -v "$(pwd)/data:/app/data" qae-lhco smoke

What reproduces, and what does not: see the header of the dockerfile and the
Limitations section of README.md. Briefly -- qae_ry, qae_zz, pca and mj1_only
reproduce across machines to every decimal place reported; the five classical
autoencoders do not, because their training is chaotic.
"""

ALIASES = {
    # --out is NOT left at its default here. results/ is normally a mounted
    # host directory, and src.run_study's default output path is results/
    # itself, so a smoke test run with the default would overwrite the
    # committed metrics.json with throwaway numbers from a 4000-event run.
    "smoke": ["python", "-m", "src.run_study", "--quick",
              "--out", "results/smoke"],
    "verify": ["python", "tools/verify_reproduction.py",
               "--seeds", "0", "1", "2", "--out", "results/verify"],
}


def pin_threads() -> None:
    import torch

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # Inter-op FIRST. set_num_threads() initialises the thread pools, after
    # which set_num_interop_threads() raises "cannot set number of interop
    # threads after parallel work has started" -- which an except block then
    # swallows, leaving the inter-op pool at the host's core count while
    # everything claims it is pinned. Verified in the container: called in
    # this order both report 1; called the other way round inter-op stays 6.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:                               # noqa: BLE001
        print(f"WARNING: inter-op thread pool not pinned ({exc}); "
              f"results may depend on the host's core count", file=sys.stderr)
    torch.set_num_threads(1)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    pin_threads()
    argv = ALIASES.get(argv[0], argv)

    # "python -m mod args" and "python script.py args" are run in-process so
    # the thread pinning above applies to them directly. Anything else (an
    # interactive shell, python -c, a non-python command) is handed to a
    # subprocess, which inherits OMP_NUM_THREADS -- torch reads that for its
    # intra-op default, so the pinning still holds there.
    if argv[0] in ("python", "python3", sys.executable):
        rest = argv[1:]
        if not rest:
            print(USAGE)
            return 0
        if rest[0] == "-m" and len(rest) > 1:
            sys.argv = [rest[1]] + rest[2:]
            runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
            return 0
        if not rest[0].startswith("-") and os.path.isfile(rest[0]):
            sys.argv = list(rest)
            runpy.run_path(rest[0], run_name="__main__")
            return 0

    # Resolve a bare "python" to the interpreter actually running this, so the
    # container never falls through to some other interpreter on PATH.
    if argv[0] in ("python", "python3"):
        argv = [sys.executable] + argv[1:]
    import subprocess
    return subprocess.call(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
