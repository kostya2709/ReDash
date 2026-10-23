//! Shared correctness suite for the ReDASH step ladder.
//!
//! Every step crate (steps 0–9) has identical *external* semantics — only the
//! internals change — so the correctness tests are written ONCE here and reused via
//! a macro. A step's `tests/correctness.rs` does:
//!
//! ```ignore
//! use redash_step0::DashExtendibleHashMap;
//! fn mk(cap: usize) -> DashExtendibleHashMap<u64, u64> {
//!     let _ = cap;                       // pool-backed steps size to `cap`
//!     DashExtendibleHashMap::new()
//! }
//! redash_testsuite::dash_correctness_suite!(mk);
//! ```
//!
//! The map `M` returned by `mk(cap)` must expose the stable public API:
//! `insert(K,V)->bool` (false if key already present, DASH does NOT overwrite),
//! `get(&K)->Option<V>`, `remove(&K)->bool`, `len()->usize`, `is_empty()->bool`,
//! `bucket_count()->usize`, `dash_map_fill_histogram()->Vec<u64>`,
//! `global_depth()->usize`, and `debug_validate()->Result<(),String>` (structural
//! invariants: per-bucket count==popcount(alloc), member&!alloc==0, count<=14, and
//! per-segment count==sum bucket counts, sum==len()).
//!
//! Keys/values are `u64`. `cap` is an upper bound on live entries (for pool sizing).

/// Small, fast, deterministic PRNG (SplitMix64) so every step sees the SAME stream.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Value tag for a key — distinctive so any cross-slot corruption is caught.
///
/// Masked to the low 48 bits: from step 5 on, the routing optimization stores 16 hash routing
/// bits in the value's UPPER 16 bits, so only ≤48-bit values round-trip (matching the
/// benchmark, which uses bounded integer / pointer values). 48 bits of entropy is still
/// ample to catch any cross-slot corruption. Steps 0–4 (no routing) are unaffected — they
/// round-trip these values exactly as before.
pub const fn val_of(k: u64) -> u64 {
    (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_A5A5_A5A5_A5A5) & 0x0000_FFFF_FFFF_FFFF
}

#[macro_export]
macro_rules! dash_correctness_suite {
    ($mk:path) => {
        use $crate::{splitmix64, val_of};
        use std::collections::HashMap;

        // 1. Empty map: every lookup misses, len 0.
        #[test]
        fn empty_map_misses() {
            let m = $mk(64);
            assert_eq!(m.len(), 0);
            assert!(m.is_empty());
            for k in [0u64, 1, 7, 99, u64::MAX] {
                assert_eq!(m.get(&k), None, "empty map must miss {k}");
            }
        }

        // 2. Single element round-trip + duplicate rejection (no overwrite).
        #[test]
        fn single_and_duplicate() {
            let mut m = $mk(64);
            assert!(m.insert(42, 100));
            assert_eq!(m.get(&42), Some(100));
            assert_eq!(m.len(), 1);
            assert!(!m.insert(42, 999), "duplicate insert must be rejected");
            assert_eq!(m.get(&42), Some(100), "value must NOT be overwritten");
            assert_eq!(m.len(), 1, "rejected duplicate must not grow len");
        }

        // 3. Large sequential insert -> readback all (forces many splits + doublings).
        #[test]
        fn sequential_readback() {
            let n: u64 = 500_000;
            let mut m = $mk(n as usize);
            for i in 0..n { assert!(m.insert(i, val_of(i)), "fresh insert {i}"); }
            assert_eq!(m.len(), n as usize);
            for i in 0..n { assert_eq!(m.get(&i), Some(val_of(i)), "readback {i}"); }
            for i in n..n + 50_000 { assert_eq!(m.get(&i), None, "absent {i}"); }
        }

        // 4. Random keys (different bucket/fingerprint distribution than sequential).
        #[test]
        fn random_keys_readback() {
            let n = 300_000usize;
            let mut m = $mk(n);
            let mut s = 0xDEAD_BEEF_u64;
            let mut keys = Vec::with_capacity(n);
            let mut model: HashMap<u64, u64> = HashMap::new();
            while model.len() < n {
                let k = splitmix64(&mut s);
                if model.insert(k, val_of(k)).is_none() {
                    assert!(m.insert(k, val_of(k)), "fresh random insert {k}");
                    keys.push(k);
                }
            }
            assert_eq!(m.len(), n);
            for &k in &keys { assert_eq!(m.get(&k), Some(val_of(k)), "random readback {k}"); }
        }

        // 5. Dense structured keys (low-byte-patterned). NOTE: this does NOT guarantee
        //    fingerprint collisions — meta_hash is the low byte of the *hashed* key, not
        //    of the key — but the structured distribution still stresses bucket fill,
        //    displacement and stash differently from sequential/random. (A true
        //    same-fingerprint test requires an identity hasher; see path_coverage.)
        #[test]
        fn dense_structured_keys() {
            let mut m = $mk(100_000);
            let mut inserted = Vec::new();
            let fp = 0x5Au64;
            for base in 0..50_000u64 {
                let k = (base << 8) | fp;
                assert!(m.insert(k, val_of(k)), "insert {k}");
                inserted.push(k);
            }
            for &k in &inserted { assert_eq!(m.get(&k), Some(val_of(k)), "readback {k}"); }
            assert_eq!(m.len(), inserted.len());
            m.debug_validate().expect("structural invariants after dense inserts");
        }

        // 6. Remove everything -> empty; removing absent keys returns false.
        #[test]
        fn remove_all_and_absent() {
            let n = 200_000u64;
            let mut m = $mk(n as usize);
            for i in 0..n { m.insert(i, val_of(i)); }
            assert!(!m.remove(&(n + 1)), "removing absent must be false");
            for i in 0..n { assert!(m.remove(&i), "remove present {i}"); }
            assert_eq!(m.len(), 0, "len 0 after removing all");
            for i in 0..n { assert_eq!(m.get(&i), None, "gone after remove {i}"); }
            assert!(!m.remove(&0), "double remove must be false");
        }

        // 7. Remove half, verify partition, reinsert with NEW values.
        #[test]
        fn remove_half_reinsert() {
            let n = 200_000u64;
            let mut m = $mk(n as usize);
            for i in 0..n { m.insert(i, val_of(i)); }
            for i in (0..n).step_by(2) { assert!(m.remove(&i)); }      // remove evens
            assert_eq!(m.len(), (n / 2) as usize);
            for i in 0..n {
                if i % 2 == 0 { assert_eq!(m.get(&i), None, "even {i} removed"); }
                else { assert_eq!(m.get(&i), Some(val_of(i)), "odd {i} kept"); }
            }
            for i in (0..n).step_by(2) {                               // reinsert evens, new val
                assert!(m.insert(i, val_of(i) ^ 0xFFFF), "reinsert {i}");
            }
            assert_eq!(m.len(), n as usize);
            for i in 0..n {
                let want = if i % 2 == 0 { val_of(i) ^ 0xFFFF } else { val_of(i) };
                assert_eq!(m.get(&i), Some(want), "post-reinsert {i}");
            }
        }

        // 8. Histogram invariants: bucket-count (every bucket binned) AND the stronger
        //    WEIGHTED sum (total occupied slots == len()), plus full structural validate.
        #[test]
        fn fill_histogram_invariants() {
            let mut m = $mk(1_000_000);
            for n in [10_000u64, 100_000, 1_000_000] {
                for i in 0..n { let _ = m.insert(i, val_of(i)); }
                let hist = m.dash_map_fill_histogram();
                let hsum: u64 = hist.iter().sum();
                assert_eq!(hsum as usize, m.bucket_count(),
                    "hist sum {hsum} must equal bucket_count {} at n={n}", m.bucket_count());
                let weighted: u64 = hist.iter().enumerate().map(|(i, &c)| i as u64 * c).sum();
                assert_eq!(weighted as usize, m.len(),
                    "weighted hist {weighted} (occupied slots) must equal len {} at n={n}", m.len());
                m.debug_validate().expect("structural invariants in histogram test");
            }
        }

        // 9. MODEL-BASED ORACLE: random insert/remove/get stream cross-checked against
        //    std::HashMap. The strongest test — catches any divergence in semantics.
        #[test]
        fn oracle_random_churn() {
            let mut m = $mk(200_000);
            let mut model: HashMap<u64, u64> = HashMap::new();
            let mut s = 0x0123_4567_89AB_CDEF_u64;
            let key_space = 80_000u64;             // dense enough to force collisions/churn
            for step in 0..1_500_000u64 {
                let r = splitmix64(&mut s);
                let k = r % key_space;
                match (r >> 20) % 10 {
                    0..=5 => {                      // 60% insert
                        let v = val_of(k) ^ (step & 0xFF);
                        let got = m.insert(k, v);
                        let exp = !model.contains_key(&k);
                        assert_eq!(got, exp, "insert({k}) return at step {step}");
                        model.entry(k).or_insert(v);   // DASH does NOT overwrite
                        // immediate check: a rejected duplicate must not have overwritten.
                        assert_eq!(m.get(&k), model.get(&k).copied(),
                            "post-insert value for {k} at step {step}");
                    }
                    6..=7 => {                      // 20% remove
                        let got = m.remove(&k);
                        let exp = model.remove(&k).is_some();
                        assert_eq!(got, exp, "remove({k}) return at step {step}");
                    }
                    _ => {                          // 20% get
                        assert_eq!(m.get(&k), model.get(&k).copied(),
                            "get({k}) mismatch at step {step}");
                    }
                }
                if step % 200_000 == 0 {
                    assert_eq!(m.len(), model.len(), "len divergence at step {step}");
                    m.debug_validate().expect("structural invariants during churn");
                }
            }
            // Final FULL-KEYSPACE reconciliation: scan every key, not just live model keys,
            // so a "stale extra key + missing real key" pair (len balanced) cannot hide.
            assert_eq!(m.len(), model.len(), "final len");
            for k in 0..key_space {
                assert_eq!(m.get(&k), model.get(&k).copied(), "final full-keyspace key {k}");
            }
            m.debug_validate().expect("final structural invariants");
        }

        // 11. HIGH-LOAD churn + path coverage: grow large (forces directory doublings and
        //     many splits), churn, and prove displacement/stash actually fired (buckets
        //     reach high occupancy) — the paths behind the bug class we fixed.
        #[test]
        fn high_load_and_path_coverage() {
            let n = 800_000u64;
            let mut m = $mk(n as usize);
            for i in 0..n { assert!(m.insert(i, val_of(i))); }
            // splits + doublings must have happened
            assert!(m.global_depth() > 1, "directory must have doubled");
            assert!(m.bucket_count() > 66 * 4, "many splits expected, got {}", m.bucket_count());
            // displacement/stash imply buckets reach near-full occupancy
            let hist = m.dash_map_fill_histogram();
            let high_occupancy: u64 = hist.iter().skip(12).sum(); // buckets with >=12 of 14
            assert!(high_occupancy > 0, "expected near-full buckets (displacement/stash path)");
            m.debug_validate().expect("structural invariants at high load");
            // churn: remove a third, reinsert disjoint keys, verify the partition
            for i in (0..n).step_by(3) { assert!(m.remove(&i)); }
            for i in n..n + n / 3 { assert!(m.insert(i, val_of(i))); }
            m.debug_validate().expect("structural invariants after churn");
            for i in 0..n {
                let want = if i % 3 == 0 { None } else { Some(val_of(i)) };
                assert_eq!(m.get(&i), want, "post-churn key {i}");
            }
            for i in n..n + n / 3 { assert_eq!(m.get(&i), Some(val_of(i)), "new key {i}"); }
        }

        // 10b. segment_count() is monotonic non-decreasing and segment_count()*66 ==
        //      bucket_count() throughout a growing build (catches a wrong/decremented
        //      cached counter — the class of the original unique_tables doubling bug).
        #[test]
        fn segment_count_monotonic_and_consistent() {
            let mut m = $mk(300_000);
            let mut last = 0usize;
            for i in 0..300_000u64 {
                m.insert(i, val_of(i));
                if i % 4096 == 0 {
                    let sc = m.segment_count();
                    assert!(sc >= last, "segment_count must be monotonic: {sc} < {last}");
                    assert_eq!(sc * 66, m.bucket_count(), "segment_count*66 == bucket_count");
                    last = sc;
                }
            }
        }

        // 10. Value integrity under churn near bucket capacity (corruption detector).
        #[test]
        fn value_integrity_dense() {
            // Pack many keys into a narrow space so buckets fill, displace, stash, split.
            let mut m = $mk(64_000);
            for i in 0..64_000u64 { assert!(m.insert(i, val_of(i))); }
            // overwrite attempts must all be rejected, values intact
            for i in (0..64_000u64).step_by(7) { assert!(!m.insert(i, 0xDEAD)); }
            for i in 0..64_000u64 { assert_eq!(m.get(&i), Some(val_of(i)), "integrity {i}"); }
        }
    };
}
