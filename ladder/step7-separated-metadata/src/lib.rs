// Step 7 — SEPARATED METADATA, the novel contribution. Single change vs
// step 6: a segment previously stored an array-of-structs (`[Bucket; 66]`, each bucket 32 B of
// metadata interleaved with 224 B of KV pairs). It now stores two PARALLEL arrays — a dense
// `metas: [BucketMeta; 66]` and `pairs: [[Pair;14]; 66]`. Fingerprint scans (get/remove) walk
// only the compact metas array (all 66 in ~2 KiB) and touch a bucket's pairs solely for
// fingerprint candidates, so a negative lookup never pulls the ~14.8 KiB pairs region into
// cache. Every bitmap / fingerprint / displacement / split rule is byte-for-byte identical to
// step 6 (routing bits, no-rehash split, SIMD, pool, FoldHash all unchanged) — only the physical
// storage layout moved from interleaved to separated.
//
// The u64 routing impl (from step 5) steals the top 16 bits of the value, so the workload uses
// values that fit in 48 bits — the shared test suite masks `val_of` to 48 bits for this.
// Unused helpers are kept so each step's source matches the code that was measured.
#![allow(dead_code)]

mod allocator;
mod bucket;
mod directory;
mod table;
mod hash;

pub use hash::{DashExtendibleHashMap, DefaultHashBuilder, SipHashBuilder};
pub use allocator::{DefaultAllocator, PoolAllocator, TableAllocator};

/// Number of hash routing bits stored in the upper bits of the value.
/// 16 bits = 64 - 48 (pointer address space); covers splits from `initial_depth`
/// up to `initial_depth + 16` without rehashing. Deeper splits fall back to rehash.
const ROUTING_BITS: usize = 16;
const VALUE_MASK_U64: u64 = (1u64 << (64 - ROUTING_BITS)) - 1; // 0x0000_FFFF_FFFF_FFFF

/// Trait for values that can store hash routing bits in their upper bits.
/// Enables the split optimization: at insert time, store routing bits from the hash;
/// during split, read them to decide stay/migrate without rehashing.
///
/// The default implementation is a no-op (no bits stored, always falls back to rehash);
/// `u64` implements real routing.
pub trait RoutingValue: Copy + Default {
    /// Pack routing bits from `key_hash` into the value's upper bits. Called at insert time.
    fn pack_routing(self, _key_hash: u64, _initial_depth: usize) -> Self { self }
    /// Return the original value with routing bits cleared. Called at get time.
    fn unpack_routing(self) -> Self { self }
    /// Whether this entry migrates when a segment at `old_depth` splits to `old_depth+1`.
    /// `Some(true)`=migrate, `Some(false)`=stay, `None`=depth exceeds stored bits → rehash.
    fn should_migrate(&self, _old_depth: usize, _initial_depth: usize) -> Option<bool> { None }
}

#[cfg(test)]
mod routing_tests {
    use super::*;

    /// The whole optimization hinges on ONE invariant: the routing-bit migrate decision
    /// must equal what a rehash would decide, `(key_hash >> (64 - (d+1))) == ((pat<<1)|1)`,
    /// for every covered depth `d`. If it ever disagrees, entries land in the wrong segment.
    #[test]
    fn should_migrate_matches_rehash_decision() {
        let initial_depth = 1usize;
        // A spread of hashes (splitmix-ish) exercised across all covered depths.
        let mut h = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..5000 {
            h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            h ^= h >> 31;
            let packed = 0u64.pack_routing(h, initial_depth);
            for d in initial_depth..(initial_depth + ROUTING_BITS) {
                let new_depth = d + 1;
                // The entry's pattern at depth d is the top d bits of h; migrate iff the
                // next bit down makes it the odd child ((pat<<1)|1).
                let pat = h >> (64 - d);
                let new_pattern = (pat << 1) | 1;
                let rehash = (h >> (64 - new_depth)) == new_pattern;
                assert_eq!(
                    packed.should_migrate(d, initial_depth), Some(rehash),
                    "hash={h:#018x} depth={d}"
                );
            }
            // Beyond the stored bits → fall back to rehash (None).
            assert_eq!(packed.should_migrate(initial_depth + ROUTING_BITS, initial_depth), None);
        }
    }

    #[test]
    fn pack_unpack_preserves_low_48_bits() {
        let h = 0xDEAD_BEEF_CAFE_1234u64;
        for &v in &[0u64, 1, 0x0000_FFFF_FFFF_FFFF, 0x0000_1234_5678_9ABC] {
            assert_eq!(v.pack_routing(h, 1).unpack_routing(), v & VALUE_MASK_U64);
        }
    }
}

impl RoutingValue for u64 {
    #[inline(always)]
    fn pack_routing(self, key_hash: u64, initial_depth: usize) -> Self {
        let clean = self & VALUE_MASK_U64;
        let shift = 64 - initial_depth - ROUTING_BITS;
        let bits = ((key_hash >> shift) & ((1u64 << ROUTING_BITS) - 1)) << (64 - ROUTING_BITS);
        clean | bits
    }

    #[inline(always)]
    fn unpack_routing(self) -> Self {
        self & VALUE_MASK_U64
    }

    #[inline(always)]
    fn should_migrate(&self, old_depth: usize, initial_depth: usize) -> Option<bool> {
        if old_depth < initial_depth {
            return None;
        }
        let bit_idx = old_depth - initial_depth;
        if bit_idx >= ROUTING_BITS {
            return None;
        }
        // routing bit `bit_idx` lives at value bit (63 - bit_idx); a set bit == migrate,
        // matching the rehash test `(key_hash >> (64 - new_depth)) == new_pattern`.
        Some((*self >> (63 - bit_idx)) & 1 == 1)
    }
}
