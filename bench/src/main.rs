// Ladder benchmark: measures every step of the ladder in one process.
//
// CPU frequency and thermal state drift between processes, so a metric measured in one
// process is not comparable with the same metric measured in another. All variants are
// therefore measured in ONE process, interleaved within each run, in a freshly shuffled
// order every run (so no variant is systematically first or warm). A trial that saw a core
// migration (rdtscp aux changed) is discarded. Each metric is reported as min / median /
// max over runs.
//
// insert, find_hit and find_miss are the interleaved headline metrics. Split cost is
// measured over REPS repetitions (median + spread). The fill histogram is structural.
// `step4_siphash` is step 4 built with SipHash: it isolates the step 3 -> step 4 hasher
// effect from the rest of that step.
//
// Before timing, every variant is checked once per size (untimed): each inserted key must
// return exactly its value, and no absent key may be found.
//
// Run:  OUT=results/sweep1 SIZES=100000,300000,1000000,3000000,10000000,30000000 \
//       RUNS=7 MEAS=20000 REPS=3 cargo run --release -p redash_bench
// Writes OUT/<variant>.csv, then OUT/COMPLETE once every size of every variant is written.

#[cfg(target_os = "linux")]
fn pin_to_core() {
    let rc = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(0, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set)
    };
    if rc != 0 { eprintln!("[harness] WARNING: sched_setaffinity failed (rc={})", rc); }
    let start = std::time::Instant::now();
    let mut x = 0u64;
    while start.elapsed().as_millis() < 200 { x = x.wrapping_add(1); }
    std::hint::black_box(x);
}
#[cfg(not(target_os = "linux"))]
fn pin_to_core() {}

#[inline(always)]
fn rdtscp() -> (u64, u32) {
    unsafe {
        let mut aux = 0u32;
        core::arch::x86_64::_mm_lfence();
        let t = core::arch::x86_64::__rdtscp(&mut aux);
        core::arch::x86_64::_mm_lfence();
        (t, aux)
    }
}
fn time_ns() -> u128 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts); }
    (ts.tv_sec as u128) * 1_000_000_000 + ts.tv_nsec as u128
}
fn ns_per_cycle() -> f64 {
    let t0 = time_ns(); let (c0, _) = rdtscp();
    let start = std::time::Instant::now();
    let mut x = 0u64;
    while start.elapsed().as_millis() < 80 { x = x.wrapping_add(1); }
    std::hint::black_box(x);
    let (c1, _) = rdtscp(); let t1 = time_ns();
    (t1 - t0) as f64 / (c1 - c0) as f64
}
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
fn stats(v: &[f64]) -> (f64, f64, f64) { // (min, median, max)
    if v.is_empty() { return (f64::NAN, f64::NAN, f64::NAN); }
    let mut s = v.to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[0], s[s.len() / 2], s[s.len() - 1])
}
fn minf(v: &[f64]) -> f64 { v.iter().cloned().fold(f64::INFINITY, f64::min) }

// Deterministic per-run shuffle of variant indices (Fisher-Yates + xorshift seeded by
// run). TRUE shuffle, not cyclic rotation: rotation preserves relative order, so a pool
// variant that always follows other pool variants reuses their warm/freed arena pages
// (we saw m4_siphash systematically beat the identical m3 by up to 42% at 10M). Shuffling
// breaks that, and min-over-runs then captures each variant's best (warm) draw fairly.
fn shuffled_order(len: usize, run: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ (run as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    for i in (1..len).rev() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let j = (s % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}


// Values must fit in 48 bits: from step 5 on, the top 16 bits of a stored value carry
// routing bits. The multiplier spreads values; the mask keeps them in range.
const VAL_MUL: u64 = 2654435761;
const VAL_MASK: u64 = (1 << 48) - 1;
#[inline(always)]
fn val(i: u64) -> u64 { i.wrapping_mul(VAL_MUL) & VAL_MASK }

// One interleaved trial -> (insert_ns, find_hit_ns, find_miss_ns); None if a core
// migration happened during any timed region (rdtscp aux changed).
type TrialFn = Box<dyn Fn(u64) -> Option<(f64, f64, f64)>>;
struct Aux {
    nonsplit_min: f64, nonsplit_med: f64,
    split_p50: f64, split_p50_spread: f64, split_p90: f64, split_p99: f64, split_count: u64,
    segments: u64, bucket_count: u64, fill_hist: Vec<u64>, pool_exhausted: usize,
}
type AuxFn = Box<dyn Fn(u64, u64, usize, usize) -> Aux>; // (n, meas, runs, reps)

fn make_trial<M: 'static>(
    npc: f64,
    mk: impl Fn(u64) -> M + 'static,
    ins: impl Fn(&mut M, u64) + 'static,
    get: impl Fn(&M, u64) -> bool + 'static,
) -> TrialFn {
    Box::new(move |n| {
        let mut m = mk(n);
        let (c0, a0) = rdtscp();
        for i in 0..n { ins(&mut m, i); }
        let (c1, a1) = rdtscp();
        if a0 != a1 { return None; } // migrated
        let insert_ns = (c1 - c0) as f64 * npc / n as f64;
        let (c0, a0) = rdtscp();
        let mut s = 0u64; for i in 0..n { if get(&m, i) { s += 1; } }
        let (c1, a1) = rdtscp(); std::hint::black_box(s);
        if a0 != a1 { return None; }
        let fh = (c1 - c0) as f64 * npc / n as f64;
        let (c0, a0) = rdtscp();
        let mut s2 = 0u64; for i in n..2 * n { if get(&m, i) { s2 += 1; } }
        let (c1, a1) = rdtscp(); std::hint::black_box(s2);
        if a0 != a1 { return None; }
        let fm = (c1 - c0) as f64 * npc / n as f64;
        Some((insert_ns, fh, fm))
    })
}


#[allow(clippy::too_many_arguments)]
fn make_aux<M: 'static>(
    npc: f64,
    mk: impl Fn(u64) -> M + 'static,
    ins: impl Fn(&mut M, u64) + 'static,
    bc: impl Fn(&M) -> u64 + 'static,
    sc: impl Fn(&M) -> u64 + 'static,
    fh: impl Fn(&M) -> Vec<u64> + 'static,
    exhausted: impl Fn(&M) -> usize + 'static, // 0 for non-pool
) -> AuxFn {
    Box::new(move |n, meas, runs, reps| {
        // clean non-split insert (best-effort; NaN if every window hit a split)
        let mut ns_clean = Vec::new();
        let mut pool_exhausted = 0usize;
        for _ in 0..runs {
            let mut m = mk(n + meas);
            for i in 0..n { ins(&mut m, i); }
            let before = bc(&m);
            let (c0, a0) = rdtscp();
            for i in n..n + meas { ins(&mut m, i); }
            let (c1, a1) = rdtscp();
            if a0 == a1 && bc(&m) == before {
                ns_clean.push((c1 - c0) as f64 * npc / meas as f64);
            }
            pool_exhausted = pool_exhausted.max(exhausted(&m));
        }
        // split-cost: REPS per-op passes; exact split detection via O(1) segment_count(),
        // poll excluded from deltas (t0/t1/t2). seg delta tracked (one insert can split>1x
        // across retries). Report median + spread (max-min) of per-rep p50.
        let mut p50s = Vec::new(); let mut p90s = Vec::new(); let mut p99s = Vec::new();
        let mut split_count = 0u64;
        let mut last_hist = vec![]; let mut last_bc = 0u64;
        for _ in 0..reps {
            let mut split_t = Vec::new();
            let mut m = mk(n);
            let mut prev_seg = sc(&m);
            let (mut t_prev, _) = rdtscp();
            for i in 0..n {
                ins(&mut m, i);
                let (t1, _) = rdtscp();
                let delta = (t1 - t_prev) as f64 * npc;
                let seg = sc(&m);
                if seg > prev_seg { split_t.push(delta); prev_seg = seg; }
                let (t2, _) = rdtscp();
                t_prev = t2;
            }
            split_t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            p50s.push(percentile(&split_t, 50.0));
            p90s.push(percentile(&split_t, 90.0));
            p99s.push(percentile(&split_t, 99.0));
            split_count = (sc(&m) as u64).saturating_sub(2); // segments - initial(2)
            last_hist = fh(&m); last_bc = bc(&m);
            pool_exhausted = pool_exhausted.max(exhausted(&m));
        }
        let (p50_min, p50_med, p50_max) = stats(&p50s);
        Aux {
            nonsplit_min: if ns_clean.is_empty() { f64::NAN } else { minf(&ns_clean) },
            nonsplit_med: { let (_, m, _) = stats(&ns_clean); m },
            split_p50: p50_med, split_p50_spread: p50_max - p50_min,
            split_p90: { let (_, m, _) = stats(&p90s); m },
            split_p99: { let (_, m, _) = stats(&p99s); m },
            split_count,
            segments: last_bc / 66, bucket_count: last_bc, fill_hist: last_hist, pool_exhausted,
        }
    })
}


// Untimed correctness check: build a table of n keys, then require get(i) == Some(val(i))
// for every inserted key and get(i) == None for n absent keys.
type VerifyFn = Box<dyn Fn(u64) -> Result<(), String>>;
fn make_verify<M: 'static>(
    mk: impl Fn(u64) -> M + 'static,
    ins: impl Fn(&mut M, u64) + 'static,
    get: impl Fn(&M, u64) -> Option<u64> + 'static,
) -> VerifyFn {
    Box::new(move |n| {
        let mut m = mk(n);
        for i in 0..n { ins(&mut m, i); }
        for i in 0..n {
            if get(&m, i) != Some(val(i)) { return Err(format!("key {i}: wrong or missing value")); }
        }
        for i in n..2 * n {
            if get(&m, i).is_some() { return Err(format!("absent key {i} was found")); }
        }
        Ok(())
    })
}

struct Variant { name: &'static str, trial: TrialFn, aux: AuxFn, verify: VerifyFn }

// Pool capacity in segments for a table of h keys, with headroom so no run exhausts it.
fn pool_cap(h: u64) -> usize { (h as usize / 450).max(64) + 2048 }

fn registry(npc: f64) -> Vec<Variant> {
    let mut v = Vec::new();
    // Steps 0-2 have no pool allocator and use their default constructor.
    macro_rules! plain {
        ($name:literal, $krate:ident) => {{
            use $krate::DashExtendibleHashMap as Map;
            type M = Map<u64, u64>;
            v.push(Variant {
                name: $name,
                trial: make_trial::<M>(npc, |_| Map::new(),
                    |m, i| { m.insert(i, val(i)); }, |m, i| m.get(&i).is_some()),
                aux: make_aux::<M>(npc, |_| Map::new(), |m, i| { m.insert(i, val(i)); },
                    |m| m.bucket_count() as u64, |m| m.segment_count() as u64,
                    |m| m.dash_map_fill_histogram(), |_| 0),
                verify: make_verify::<M>(|_| Map::new(), |m, i| { m.insert(i, val(i)); },
                    |m, i| m.get(&i)),
            });
        }};
    }
    // Steps 3-9 draw segments from a pre-faulted pool sized to the run.
    macro_rules! pooled {
        ($name:literal, $krate:ident, $hasher:ident) => {
            pooled!($name, $krate, $hasher, |m: &M| m.pool_exhausted_count())
        };
        // pool_exhausted_count() exists only for a step's default hasher; a control built
        // with another hasher reports 0 (its pool is sized with the same headroom).
        ($name:literal, $krate:ident, $hasher:ident, $exhausted:expr) => {{
            use $krate::{DashExtendibleHashMap as Map, PoolAllocator, $hasher};
            type M = Map<u64, u64, PoolAllocator<u64, u64>, $hasher>;
            let mk = |h: u64| M::with_allocator_and_hasher(PoolAllocator::new(pool_cap(h)), $hasher::default());
            v.push(Variant {
                name: $name,
                trial: make_trial::<M>(npc, mk,
                    |m, i| { m.insert(i, val(i)); }, |m, i| m.get(&i).is_some()),
                aux: make_aux::<M>(npc, mk, |m, i| { m.insert(i, val(i)); },
                    |m| m.bucket_count() as u64, |m| m.segment_count() as u64,
                    |m| m.dash_map_fill_histogram(), $exhausted),
                verify: make_verify::<M>(mk, |m, i| { m.insert(i, val(i)); }, |m, i| m.get(&i)),
            });
        }};
    }
    plain!("step0", redash_step0);
    plain!("step1", redash_step1);
    plain!("step2", redash_step2);
    pooled!("step3", redash_step3, SipHashBuilder);
    pooled!("step4", redash_step4, DefaultHashBuilder);
    pooled!("step4_siphash", redash_step4, SipHashBuilder, |_: &M| 0);
    pooled!("step5", redash_step5, DefaultHashBuilder);
    pooled!("step6", redash_step6, DefaultHashBuilder);
    pooled!("step7", redash_step7, DefaultHashBuilder);
    pooled!("step8", redash_step8, DefaultHashBuilder);
    pooled!("step9", redash_step9, DefaultHashBuilder);
    v
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() {
    use std::io::Write;
    pin_to_core();
    let npc = ns_per_cycle();
    let runs: usize = env_or("RUNS", 7);
    let reps: usize = env_or("REPS", 3);
    let meas: u64 = env_or("MEAS", 20_000);
    let sizes: Vec<u64> = std::env::var("SIZES")
        .unwrap_or_else(|_| "100000,300000,1000000,3000000,10000000,30000000".into())
        .split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let out = std::env::var("OUT").unwrap_or_else(|_| "results/sweep".into());
    let mut reg = registry(npc);
    // VARIANTS="step6 step7" restricts the run to those variants, still interleaved among
    // themselves. A subset is only comparable within itself, never with another run.
    if let Ok(sel) = std::env::var("VARIANTS") {
        let want: Vec<&str> = sel.split_whitespace().collect();
        if !want.is_empty() { reg.retain(|v| want.contains(&v.name)); }
    }
    let names: Vec<&str> = reg.iter().map(|v| v.name).collect();
    if runs == 0 || reps == 0 || meas == 0 || sizes.is_empty() || names.is_empty() {
        panic!("nothing to measure: need RUNS, REPS, MEAS >= 1, at least one size and one variant");
    }
    // A pool that runs out falls back to the heap, and those cold allocations contaminate the
    // timing. The pool-backed steps abort instead of falling back when this is set.
    std::env::set_var("DASH_STRICT_POOL", "1");
    eprintln!("[bench] npc={npc:.4} runs={runs} reps={reps} meas={meas} sizes={sizes:?} variants={names:?} out={out}");

    // Each sweep writes into its own new directory, so files from an interrupted or earlier
    // run can never be mixed into it. A directory without COMPLETE must not be used.
    if let Ok(mut d) = std::fs::read_dir(&out) {
        if d.next().is_some() { panic!("{out} is not empty; choose a new OUT for every sweep"); }
    }
    std::fs::create_dir_all(&out).unwrap();
    let hdr = "variant,n,insert_ns,insert_med,insert_max,find_hit_ns,find_hit_med,find_hit_max,find_miss_ns,find_miss_med,find_miss_max,clean_trials,nonsplit_min_ns,nonsplit_med_ns,split_p50_ns,split_p50_spread_ns,split_p90_ns,split_p99_ns,split_count,pool_exhausted,segments,bucket_count,fill_histogram\n";
    let mut files: std::collections::HashMap<&str, std::fs::File> = std::collections::HashMap::new();
    for &nm in &names {
        let mut f = std::fs::File::create(format!("{out}/{nm}.csv")).unwrap();
        f.write_all(hdr.as_bytes()).unwrap();
        files.insert(nm, f);
    }

    for &n in &sizes {
        for v in &reg {
            if let Err(e) = (v.verify)(n) { panic!("{} n={n}: correctness check failed: {e}", v.name); }
        }
        // Headline metrics: every run measures every variant once, in a shuffled order.
        let mut acc: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, usize)> =
            (0..reg.len()).map(|_| (vec![], vec![], vec![], 0)).collect();
        for run in 0..runs {
            for &idx in &shuffled_order(reg.len(), run) {
                match (reg[idx].trial)(n) {
                    Some((ins, fh, fm)) => { let e = &mut acc[idx]; e.0.push(ins); e.1.push(fh); e.2.push(fm); }
                    None => acc[idx].3 += 1, // core migration: trial discarded
                }
            }
        }
        for (idx, v) in reg.iter().enumerate() {
            let a = (v.aux)(n, meas, runs, reps);
            let (iv, hv, mv, disc) = &acc[idx];
            let (i_min, i_med, i_max) = stats(iv);
            let (h_min, h_med, h_max) = stats(hv);
            let (m_min, m_med, m_max) = stats(mv);
            if a.pool_exhausted > 0 {
                panic!("{} n={n}: pool exhausted {} times; cold allocations contaminate the timing", v.name, a.pool_exhausted);
            }
            if iv.is_empty() || ![i_min, h_min, m_min].iter().all(|x| x.is_finite()) {
                panic!("{} n={n}: no usable trial ({disc} discarded after core migration)", v.name);
            }
            let hist = a.fill_hist.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
            let line = format!("{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1},{},{},{},{},\"{}\"\n",
                v.name, n, i_min, i_med, i_max, h_min, h_med, h_max, m_min, m_med, m_max,
                iv.len(), a.nonsplit_min, a.nonsplit_med, a.split_p50, a.split_p50_spread,
                a.split_p90, a.split_p99, a.split_count, a.pool_exhausted, a.segments, a.bucket_count, hist);
            let f = files.get_mut(v.name).unwrap();
            f.write_all(line.as_bytes()).unwrap();
            f.flush().unwrap();
            eprintln!("  {:<14} n={n:<9} insert(min/med)={i_min:.0}/{i_med:.0} find_hit={h_min:.0}/{h_med:.0} find_miss={m_min:.0}/{m_med:.0} split_p50={:.0}±{:.0} discarded={disc}",
                v.name, a.split_p50, a.split_p50_spread);
        }
    }
    for (_, f) in files.drain() { f.sync_all().unwrap(); }
    std::fs::write(format!("{out}/COMPLETE"), format!("sizes={sizes:?} runs={runs} reps={reps} meas={meas}\n")).unwrap();
    eprintln!("[bench] done -> {out}/<variant>.csv, marked COMPLETE");
}
