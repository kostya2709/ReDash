#!/usr/bin/env python3
"""Summarise ladder sweeps into the per-step view of the paper's ladder table.

Usage:  python3 scripts/summarise.py data/canonical/ladder_sweep1 data/canonical/ladder_sweep2 \\
            data/canonical/ladder_sweep3
        python3 scripts/summarise.py results/sweep1 results/sweep2 results/sweep3

Each step is compared with the step before it, per metric and table size.
  * Claim rule: the step has an effect at a size only where the two pooled ranges do not
    overlap. A pooled range is [min over sweeps of the per-sweep minimum, max over sweeps of
    the per-sweep maximum].
  * Size of the effect, as in the paper's ladder table: the median across sweeps of the
    within-sweep medians, step against predecessor.
  * --minima prints the effect from the medians across sweeps of the within-sweep minima
    instead, which is the estimator the rest of the paper uses.
Reads both the published CSVs (column `milestone`, names m0..m9) and fresh ones written by
the bench binary (column `variant`, names step0..step9). Standard library only.
"""
import csv, os, statistics as st, sys

STEPS = [f"step{i}" for i in range(10)]
CHANGE = {"step1": "SIMD fingerprint compare", "step2": "raw-pointer directory",
          "step3": "pre-faulted pool allocator", "step4": "FoldHash (vs. SipHash)",
          "step5": "routing bits in value", "step6": "rehash-free placement",
          "step7": "separated metadata", "step8": "positional split",
          "step9": "insert-path software prefetch"}
METRICS = ("insert", "find_hit", "find_miss")

def size(n):
    """100000 -> 1e5, 300000 -> 3e5."""
    e = len(str(n)) - 1
    m = n // 10 ** e
    return f"{m}e{e}" if m != 1 else f"1e{e}"

def load(d):
    """One sweep directory -> {(step, n): row}. A fresh sweep (column `variant`) must carry
    the COMPLETE marker the benchmark writes last; the published sweeps (column `milestone`)
    predate the marker and are accepted as archived."""
    out, fresh = {}, False
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"):
            continue
        for r in csv.DictReader(open(os.path.join(d, f))):
            fresh |= "variant" in r
            name = r.get("variant") or r.get("milestone")
            if name and name[0] == "m" and name[1:].isdigit():
                name = "step" + name[1:]          # published files use m0..m9
            key = (name, int(r["n"]))
            if key in out:
                sys.exit(f"{d}: duplicate row for {key}")
            out[key] = r
    if fresh and not os.path.exists(os.path.join(d, "COMPLETE")):
        sys.exit(f"{d}: no COMPLETE marker; the sweep was interrupted and must not be used")
    return out

def main(argv):
    use_min = "--minima" in argv
    dirs = [a for a in argv if not a.startswith("--")]
    if len(dirs) < 2:
        sys.exit(__doc__)
    sweeps = [load(d) for d in dirs]
    sizes = sorted({n for s in sweeps for (_, n) in s})
    present = [st_ for st_ in STEPS if any(k[0] == st_ for s in sweeps for k in s)]
    for d, s in zip(dirs, sweeps):
        missing = [(v, n) for v in present for n in sizes if (v, n) not in s]
        if missing:
            sys.exit(f"{d}: missing cells {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    col = "_ns" if use_min else "_med"

    def rung(a, b, m, n):
        A = [s[(a, n)] for s in sweeps]; B = [s[(b, n)] for s in sweeps]
        alo = min(float(r[m + "_ns"]) for r in A); ahi = max(float(r[m + "_max"]) for r in A)
        blo = min(float(r[m + "_ns"]) for r in B); bhi = max(float(r[m + "_max"]) for r in B)
        ma = st.median(float(r[m + col]) for r in A); mb = st.median(float(r[m + col]) for r in B)
        return (bhi < alo or blo > ahi), 100 * (mb - ma) / ma

    print(f"{len(dirs)} sweeps; effect = median across sweeps of within-sweep "
          f"{'minima' if use_min else 'medians'}; listed only where pooled ranges do not overlap\n")
    for a, b in zip(present, present[1:]):
        if int(b[4:]) != int(a[4:]) + 1:
            continue                              # a subset run: compare adjacent steps only
        parts = []
        for m in METRICS:
            hits = [(n, d) for n in sizes for sep, d in [rung(a, b, m, n)] if sep]
            if not hits:
                parts.append(f"{m}: none")
                continue
            ds = [d for _, d in hits]
            where = f"all {len(sizes)}" if len(hits) == len(sizes) else ", ".join(size(n) for n, _ in hits)
            rng = f"{min(ds):+.0f} to {max(ds):+.0f}%" if len(ds) > 1 else f"{ds[0]:+.0f}%"
            parts.append(f"{m}: {where} ({rng})")
        print(f"{b:6} {CHANGE[b]:32} " + "; ".join(parts))

if __name__ == "__main__":
    main(sys.argv[1:])
