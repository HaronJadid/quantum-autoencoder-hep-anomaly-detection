"""Flag result-shaped numbers in hand-written README prose that no run produced.

The recurring failure in this project has been prose quoting a number from an
earlier run after the numbers changed. The generated block cannot drift -- it
is regenerated -- but hand-written prose around it can, and has, repeatedly.

Scope is deliberately narrow to keep the signal-to-noise usable:

  * only text OUTSIDE the generated block is checked;
  * only decimals with >= 3 decimal places are considered, because that is what
    a metric looks like. Dataset facts (3.5 TeV, 1.2 TeV, 74 MB), architecture
    facts (6 qubits, 32/45/58 parameters) and citations (PRD 105, GTM 169) are
    integers or low-precision and are skipped rather than reported as noise;
  * a literal passes if it appears anywhere in metrics.json at any rounding
    from 3 to 6 decimal places, or is a derived quantity the checker knows how
    to reconstruct (differences and ranges of stored values).

A hit is not proof of staleness -- it means "this number is not obviously
traceable to the current results, go and look". Exit code 1 on any hit so it
can be wired into CI.

    python tools/check_stale_numbers.py [README.md] [results/final-v2/metrics.json]
"""

from __future__ import annotations

import itertools
import json
import re
import sys

BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"
MIN_DP = 3


def iter_numbers(obj):
    """Every float/int anywhere in the metrics tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_numbers(k)
            yield from iter_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_numbers(v)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        try:
            yield float(obj)
        except ValueError:
            return


# Keys holding densely-sampled curves. Including them makes the known-value
# set cover essentially every 3-dp decimal in [0, 1], so every check passes and
# the tool silently stops working. Excluded deliberately.
DENSE_KEYS = {"roc_first_seed", "histories"}


def known_values(metrics: dict) -> set:
    """Every stored number, plus pairwise differences of same-metric values.

    Prose often quotes a gap or a range rather than a stored value, so a plain
    membership test would flag legitimate derived numbers. Differences between
    AUC means and between per-seed AUCs cover the cases that actually occur
    (spreads, ranges, paired differences) without exploding combinatorially.
    """
    sparse = {k: v for k, v in metrics.items() if k not in DENSE_KEYS}
    vals = {round(v, dp) for v in iter_numbers(sparse)
            for dp in range(MIN_DP, 7)}

    aucs = []
    for block in ("summary",):
        for m in (metrics.get(block) or {}).values():
            if isinstance(m, dict) and "auc" in m:
                aucs.append(m["auc"]["mean"])
    for runs in (metrics.get("per_seed") or {}).values():
        aucs += [r["auc"] for r in runs if isinstance(r, dict) and "auc" in r]
    for a, b in itertools.combinations(aucs, 2):
        for dp in range(MIN_DP, 7):
            vals.add(round(abs(a - b), dp))
    # alpha thresholds: 0.05 / n for any plausible test count
    for n in range(1, 200):
        for dp in range(MIN_DP, 7):
            vals.add(round(0.05 / n, dp))
    return vals


def main(readme="README.md", metrics_path="results/final-v2/metrics.json") -> int:
    text = open(readme, encoding="utf-8").read()
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        hand = head + tail
    else:
        hand = text
    metrics = json.load(open(metrics_path, encoding="utf-8"))
    known = known_values(metrics)

    # Identifiers are decimal-shaped but are not results. Dropping whole lines
    # would hide real numbers that sit beside a citation, so strip only the
    # identifier spans and keep the rest of the line.
    IDENT = re.compile(
        r"https?://\S+"                 # any URL
        r"|arxiv:\s*\d+\.\d+"           # arXiv:1612.02806
        r"|10\.\d{4,}/\S*"              # DOIs
        r"|\bv?\d+\.\d+\.\d+\b",        # version strings, e.g. Delphes 3.4.1
        re.I)

    hits = []
    for lineno, line in enumerate(hand.split("\n"), 1):
        if line.lstrip().startswith(">"):        # withdrawal notices
            continue
        scrubbed = IDENT.sub(" ", line)
        for tok in re.findall(r"\d+\.\d{%d,}" % MIN_DP, scrubbed):
            v = float(tok)
            if not any(round(v, dp) in known for dp in range(MIN_DP, 7)):
                hits.append((lineno, tok, line.strip()[:100]))

    if not hits:
        print(f"OK: no untraceable result-shaped numbers in {readme} prose "
              f"(checked decimals with >= {MIN_DP} dp outside the generated block)")
        return 0
    print(f"{len(hits)} result-shaped number(s) in {readme} prose could not be "
          f"traced to {metrics_path}:")
    for lineno, tok, ctx in hits:
        print(f"  line {lineno}: {tok}   in: {ctx}")
    print("\nEach is either stale, or a derived quantity this checker cannot "
          "reconstruct. Verify by hand; prefer deleting the prose copy and "
          "pointing at the generated table.")
    return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
