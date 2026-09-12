"""Compare two metrics.json files, ignoring only timing and bookkeeping.

Used to prove that running seeds in parallel shards reproduces the
single-process run exactly, rather than merely closely: every number a result
depends on must be bit-identical, so the comparison is on the JSON values
themselves, with no tolerance.
"""
from __future__ import annotations
import json, sys

# Wall-clock and the parallel bookkeeping are expected to differ; nothing a
# reported number depends on is in here.
IGNORE_LEAF = {"seconds", "wall_seconds", "wall_seconds_summed",
               "selection_digest", "selection_seed", "parallel"}


def walk(a, b, path=""):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE_LEAF:
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: present in only one file")
            else:
                diffs += walk(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += walk(x, y, f"{path}[{i}]")
    else:
        if a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def main(p1, p2):
    a = json.load(open(p1, encoding="utf-8"))
    b = json.load(open(p2, encoding="utf-8"))
    d = walk(a, b)
    if not d:
        print(f"IDENTICAL (bit for bit, ignoring {sorted(IGNORE_LEAF)}):"
              f"\n  {p1}\n  {p2}")
        return 0
    print(f"{len(d)} difference(s) between {p1} and {p2}:")
    for line in d[:60]:
        print("  " + line)
    if len(d) > 60:
        print(f"  ... and {len(d)-60} more")
    return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
