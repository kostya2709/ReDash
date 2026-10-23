# ReDash

Code and data accompanying the paper *ReDASH: A Cache-Conscious Extendible Hash Table for
DRAM* (submitted to Труды ИСП РАН / Proceedings of ISP RAS).

[DASH](https://github.com/baotonglu/dash) (Lu et al., VLDB 2020) is an extendible hash table
designed for persistent memory. ReDASH is a DRAM redesign of it whose main change is
storing bucket metadata separately from the key/value records. This repository is being
filled in stages. **At present it contains the implementation ladder only**: ten Rust
implementations of DASH, each derived from the previous one, almost always by a single change, plus the
benchmark that measures them and the measurements the paper reports.

> **ReDASH itself is not in this repository yet.** The paper's endpoint (16 slots per
> bucket, a one-byte metadata word per slot, 256-byte bucket stride, padded segment stride)
> and the 14/16-slot separated variants of the matched-capacity control are separate code
> bases and will be added later. Step 7 below separates the metadata at the original
> 14 slots; it is a ladder step, not ReDASH.

## The ladder

| Step | Crate | Change from the previous step | Paper |
|---|---|---|---|
| 0 | `ladder/step0-naive-port` | naive but correct Rust port: `Rc<RefCell>` directory, SipHash, scalar fingerprint compare, rehash on split | |
| 1 | `ladder/step1-simd-fingerprint` | SSE2 fingerprint compare (as the C++ original's `SSE_CMP8`) | |
| 2 | `ladder/step2-raw-pointer-directory` | raw-pointer directory instead of `Rc<RefCell>`, as in the C++ original | |
| 3 | `ladder/step3-pool-allocator` | segments come from a pre-faulted pool | |
| 4 | `ladder/step4-foldhash` | FoldHash instead of SipHash | **the baseline** (§3.4) |
| 5 | `ladder/step5-routing-bits` | hash bits stored in the value's top 16 bits, so a split decides stay/migrate without rehashing | |
| 6 | `ladder/step6-rehash-free-placement` | a migrating entry's bucket comes from its position, not a rehash | interleaved side of §4.1 |
| 7 | `ladder/step7-separated-metadata` | metadata in its own array, separate from the records (still 14 slots) | separated side of §4.1 |
| 8 | `ladder/step8-positional-split` | split copies the segment byte for byte and drops entries in place; also recompacts the stash | |
| 9 | `ladder/step9-insert-prefetch` | write-intent prefetch on the insert path | |

Each step is self-contained (`src/lib.rs` describes its change in detail). Every step but
one is a single change; step 8 also recompacts stash entries, so it does not isolate
positional copying alone. A step's effect is conditional on the steps below it.

All steps share one correctness suite (`testsuite/`), run against each step by its
`tests/correctness.rs`.

**Values must fit in 48 bits** from step 5 on: the top 16 bits of a stored value carry
routing bits.

## Build and test

Requires a stable Rust toolchain (the published data used rustc 1.95.0) on x86-64; steps 1+
use SSE2 intrinsics and the benchmark reads the TSC.

```sh
cargo test --release --locked     # all ten steps + the shared suite
```

The release profile (fat LTO, one codegen unit) is set in the workspace `Cargo.toml`;
keep it, since crate boundaries alone moved one step's `find_hit` delta by 20–30 points.

## Reproducing the paper's ladder table from the published data

`data/canonical/` holds the three ladder sweeps the paper reports, unmodified (see
[Data](#data)). To regenerate the insert and lookup columns of the ladder table (not its split-cost
figures) without rerunning anything:

```sh
python3 scripts/summarise.py data/canonical/ladder_sweep1 data/canonical/ladder_sweep2 \
    data/canonical/ladder_sweep3
```

It compares each step with the previous one. An effect is listed at a size only where the
two pooled ranges (min over sweeps of the per-sweep minimum, max over sweeps of the
per-sweep maximum) do not overlap; its size is the median across sweeps of the within-sweep
medians, as in the paper's ladder table. `--minima` uses the within-sweep minima instead,
the estimator the rest of the paper uses.

## Running fresh measurements

```sh
cargo build --release --locked -p redash_bench
for s in 1 2 3; do
  OUT=results/sweep$s RUNS=7 REPS=3 MEAS=20000 \
  SIZES=100000,300000,1000000,3000000,10000000,30000000 ./target/release/redash_bench
done
python3 scripts/summarise.py results/sweep1 results/sweep2 results/sweep3
```

The benchmark measures every variant in one process, interleaved within each run in a
freshly shuffled order, pinned to core 0, timed with `rdtscp`; a trial that sees a core
migration is discarded. Metrics are insert, positive lookup (`find_hit`) and negative
lookup (`find_miss`) in ns per operation, plus split latency. Besides the ten steps it runs
`step4_siphash`, step 4 built with SipHash, which separates the hasher from the rest of
step 4. Before timing, each variant is checked once per size: every key must return exactly
its value and no absent key may be found. Each sweep needs a new, empty `OUT` directory, which gets a
`COMPLETE` file only after every size of every variant is written; `summarise.py` refuses a
fresh sweep without one. The run aborts rather than write numbers if a pool runs out
(cold heap allocations would contaminate the timing) or a variant has no usable trial.
`VARIANTS="step6 step7"` restricts a run to a subset, comparable only within itself.

Use an idle machine; the full grid takes hours. For a quick look, use smaller `SIZES` and
`RUNS=2 REPS=1`.

**Fresh runs are not the published runs.** This benchmark binary is a cleaned-up version
of the one that produced `data/canonical/`: it measures the same ladder code with the same
protocol, but it bounds values to 48 bits, adds the untimed correctness check, writes one
directory per sweep, and drops comparisons that are not part of the ladder. Expect the
same direction and similar size of effects, not identical numbers.

## Data

`data/canonical/ladder_sweep{1,2,3}/` are the three ladder sweeps behind the paper's ladder
table and its negative-lookup figure, copied unmodified (column `milestone`, names `m0`–`m9`
= steps 0–9). `harness.log` in each is the run's console output. `manifest.txt` records the
run:

- machine: KVM guest on AMD EPYC (Genoa), 8 vCPUs (the benchmark pins itself to one core), 15 GiB RAM,
  Ubuntu 24.04.4, Linux 6.8.0-110;
- toolchain: rustc 1.95.0, release profile as in this repository; the build recipe
  specifies no extra `RUSTFLAGS` (no `target-cpu=native`, unlike the paper's layout
  experiments). The manifest does not record the inherited environment;
- source: research-repository commit `00d382f1`, run 2026-08-21 (UTC); the ladder code in
  `ladder/` and `testsuite/` is identical to that commit apart from crate names, comments
  and the removal of unused imports;
- invocation: `SIZES=100000,300000,1000000,3000000,10000000,30000000 RUNS=7 REPS=3
  VARIANTS="m0 m1 m2 m3 m4 m5 m6 m7 m8 m9"` (MEAS left at its default, 20000), three
  sequential sweeps;
- run with `DASH_STRICT_POOL=1` (abort on pool exhaustion), and every row has
  `pool_exhausted = 0`; values in that run were not bounded to 48 bits, which
  does not affect the timed presence checks but is why fresh runs bound them.

## Repository layout

```
ladder/       the ten step crates
testsuite/    shared correctness suite
bench/        interleaved benchmark (x86-64 Linux)
scripts/      summarise.py: sweeps -> per-step table
data/         published measurements and their provenance
```

## Attribution

The ladder is a Rust reimplementation of the DASH design and follows the structure of the
original C++ implementation (MIT License, © 2019 Baotong Lu, Xiangpeng Hao, Tianzheng
Wang); see [NOTICE](NOTICE).
