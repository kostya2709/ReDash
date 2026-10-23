// Table/Segment implementation for Dash extendible hashing.
//
// Step 7 — SEPARATED METADATA. Single change vs step 6: the segment stored an array of
// `Bucket` structs, each interleaving 32 bytes of metadata with 224 bytes of KV pairs
// (array-of-structs). It now stores TWO parallel arrays — a dense `metas: [BucketMeta; 66]`
// and `pairs: [[Pair;14]; 66]`. Fingerprint scans (get/remove) walk only the compact metas
// array and touch a bucket's pairs solely for fingerprint candidates. Every bitmap /
// fingerprint / displacement / split rule is byte-for-byte identical to step 6; only the physical
// storage layout moved from interleaved to separated.

use crate::bucket::{BucketMeta, Pair, K_NUM_PAIR_PER_BUCKET};

const K_NUM_BUCKET: usize = 64;
const STASH_BUCKET: usize = 2;
const TOTAL_BUCKETS: usize = K_NUM_BUCKET + STASH_BUCKET;
const BUCKET_MASK: usize = K_NUM_BUCKET - 1;
const STASH_MASK: usize = STASH_BUCKET - 1;
const K_FINGER_BITS: usize = 8;

pub struct Table<K, V> {
    // Separated layout: metadata (dense, SIMD-scanned) kept apart from the KV payload (touched on match).
    metas: [BucketMeta; TOTAL_BUCKETS],
    pairs: [[Pair<K, V>; K_NUM_PAIR_PER_BUCKET]; TOTAL_BUCKETS],
    local_depth: usize,
    pub pattern: u64,
    count: usize,
}

impl<K, V> Table<K, V>
where
    K: Eq + Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    pub fn new(local_depth: usize) -> Self {
        Self {
            metas: [BucketMeta::new(); TOTAL_BUCKETS],
            pairs: [[Pair::default(); K_NUM_PAIR_PER_BUCKET]; TOTAL_BUCKETS],
            local_depth,
            pattern: 0,
            count: 0,
        }
    }

    pub fn local_depth(&self) -> usize {
        self.local_depth
    }

    pub fn insert(
        &mut self,
        key: K,
        value: V,
        key_hash: u64,
        meta_hash: u8,
    ) -> Result<bool, bool> {
        let key_pattern = key_hash >> (64 - self.local_depth);
        if key_pattern != self.pattern {
            return Ok(false);
        }

        let duplicate_check = self.get(&key, key_hash, meta_hash);
        if duplicate_check.is_some() {
            return Ok(false);
        }

        let bucket_idx = self.bucket_index(key_hash);
        let neighbor_idx = (bucket_idx + 1) & BUCKET_MASK;

        // Destructure into disjoint field borrows so `metas[a]` and `pairs[b]` never alias.
        let Self { metas, pairs, count, .. } = self;

        let target_count = metas[bucket_idx].count();
        let neighbor_count = metas[neighbor_idx].count();
        let use_neighbor = neighbor_count < target_count;
        let probe = use_neighbor;

        let insert_target_idx = if use_neighbor { neighbor_idx } else { bucket_idx };

        if !metas[insert_target_idx].is_full() {
            if metas[insert_target_idx]
                .insert(&mut pairs[insert_target_idx], key, value, meta_hash, probe)
                .is_ok()
            {
                *count += 1;
                return Ok(true);
            }
        }

        if metas[bucket_idx].is_full() && metas[neighbor_idx].is_full() {
            // Try next neighbor displacement (matching C++ Next_displace)
            let next_neighbor_idx = (bucket_idx + 2) & BUCKET_MASK;
            if let Some(displace_idx) = metas[neighbor_idx].find_org_displacement() {
                if !metas[next_neighbor_idx].is_full() {
                    if let Some((displaced_key, displaced_value, displaced_fp)) =
                        metas[neighbor_idx].get_pair(&pairs[neighbor_idx], displace_idx)
                    {
                        // NB: keep the calls OUTSIDE debug_assert! — its argument is not
                        // evaluated in release builds, which would skip these side effects.
                        let ins_ok = metas[next_neighbor_idx].insert(
                            &mut pairs[next_neighbor_idx], displaced_key, displaced_value, displaced_fp, true
                        ).is_ok();
                        debug_assert!(ins_ok, "Next_displace: dest insert must succeed");
                        // BUGFIX (validity): the displaced entry is an ORG (non-probe)
                        // entry, so it must be removed with probe=false. The original
                        // used true, so remove() silently failed.
                        let rem_ok = metas[neighbor_idx].remove(&pairs[neighbor_idx], &displaced_key, displaced_fp, false);
                        debug_assert!(rem_ok, "Next_displace: source remove must succeed");
                        metas[neighbor_idx].insert_displace(&mut pairs[neighbor_idx], key, value, meta_hash, displace_idx, true);
                        *count += 1;
                        return Ok(true);
                    }
                }
            }

            // Try prev neighbor displacement (matching C++ Prev_displace)
            let prev_neighbor_idx = if bucket_idx == 0 {
                K_NUM_BUCKET - 1
            } else {
                bucket_idx - 1
            };
            if let Some(displace_idx) = metas[bucket_idx].find_probe_displacement() {
                if !metas[prev_neighbor_idx].is_full() {
                    if let Some((displaced_key, displaced_value, displaced_fp)) =
                        metas[bucket_idx].get_pair(&pairs[bucket_idx], displace_idx)
                    {
                        let ins_ok = metas[prev_neighbor_idx].insert(
                            &mut pairs[prev_neighbor_idx], displaced_key, displaced_value, displaced_fp, false
                        ).is_ok();
                        debug_assert!(ins_ok, "Prev_displace: dest insert must succeed");
                        let rem_ok = metas[bucket_idx].remove(&pairs[bucket_idx], &displaced_key, displaced_fp, true);
                        debug_assert!(rem_ok, "Prev_displace: source remove must succeed");
                        metas[bucket_idx].insert_displace(&mut pairs[bucket_idx], key, value, meta_hash, displace_idx, false);
                        *count += 1;
                        return Ok(true);
                    }
                }
            }

            // Try stash (matching C++ Stash_insert)
            let stash_idx = bucket_idx & STASH_MASK;
            for i in 0..STASH_BUCKET {
                let stash_bucket_idx = K_NUM_BUCKET + ((stash_idx + i) & STASH_MASK);
                if !metas[stash_bucket_idx].is_full() {
                    if metas[stash_bucket_idx]
                        .insert(&mut pairs[stash_bucket_idx], key, value, meta_hash, false)
                        .is_ok()
                    {
                        *count += 1;
                        return Ok(true);
                    }
                }
            }

            return Err(true);
        }

        Err(true)
    }

    pub fn get(&self, key: &K, key_hash: u64, meta_hash: u8) -> Option<&V> {
        let key_pattern = key_hash >> (64 - self.local_depth);
        if key_pattern != self.pattern {
            return None;
        }

        let bucket_idx = self.bucket_index(key_hash);

        if let Some(value) = self.metas[bucket_idx].get(&self.pairs[bucket_idx], key, meta_hash, false) {
            return Some(value);
        }

        let neighbor_idx = (bucket_idx + 1) & BUCKET_MASK;
        if let Some(value) = self.metas[neighbor_idx].get(&self.pairs[neighbor_idx], key, meta_hash, true) {
            return Some(value);
        }

        let stash_idx = bucket_idx & STASH_MASK;
        for i in 0..STASH_BUCKET {
            let stash_bucket_idx = K_NUM_BUCKET + ((stash_idx + i) & STASH_MASK);
            if let Some(value) = self.metas[stash_bucket_idx].get(&self.pairs[stash_bucket_idx], key, meta_hash, false) {
                return Some(value);
            }
        }

        None
    }

    pub fn remove(&mut self, key: &K, key_hash: u64, meta_hash: u8) -> bool {
        let bucket_idx = self.bucket_index(key_hash);

        let Self { metas, pairs, count, .. } = self;

        if metas[bucket_idx].remove(&pairs[bucket_idx], key, meta_hash, false) {
            *count -= 1;
            return true;
        }

        let neighbor_idx = (bucket_idx + 1) & BUCKET_MASK;
        if metas[neighbor_idx].remove(&pairs[neighbor_idx], key, meta_hash, true) {
            *count -= 1;
            return true;
        }

        let stash_idx = bucket_idx & STASH_MASK;
        for i in 0..STASH_BUCKET {
            let stash_bucket_idx = K_NUM_BUCKET + ((stash_idx + i) & STASH_MASK);
            if metas[stash_bucket_idx].remove(&pairs[stash_bucket_idx], key, meta_hash, false) {
                *count -= 1;
                return true;
            }
        }

        false
    }

    fn bucket_index(&self, key_hash: u64) -> usize {
        ((key_hash >> K_FINGER_BITS) as usize) & BUCKET_MASK
    }

    /// Single-pass split matching C++ Split() exactly.
    /// Uses Insert4split (no duplicate check, direct slot assignment) and
    /// invalid_array bitmask for bulk slot clearing.
    // Step 6: NO rehash for migrating REGULAR entries — decision from routing bits (step 5), bucket
    // index from physical position + member bit. Step 7 does not change the split rules; it only
    // reads from / writes to the separated metas + pairs arrays instead of interleaved buckets.
    /// By-value split. Thin wrapper over `split_core`; retained for callers wanting an owned
    /// Table and for the no-reservable-slot fallback.
    ///
    /// PORTING NOTE. Returning a whole Table by value is a Rust artefact, not the original's
    /// shape: C++ DASH allocates the child through the allocator and fills it in place via a
    /// pointer, returning `Table<T>*` (DASH's ex_finger.h:1303-1320). The by-value
    /// form forces the caller to reserve a segment-sized return slot in its own frame and then
    /// copy the finished child into place. This is FIDELITY RESTORATION applied uniformly to
    /// every implementation, NOT an optimisation credited to any step.
    pub fn split<S: std::hash::BuildHasher>(&mut self, _key_hash: u64, hash_builder: &S, initial_depth: usize) -> Table<K, V>
    where
        K: std::hash::Hash,
        V: crate::RoutingValue,
    {
        let mut new_table = Table::new(self.local_depth + 1);
        self.split_core(&mut new_table, _key_hash, hash_builder, initial_depth);
        new_table
    }

    /// Split, constructing the child directly in `dst` -- the original's
    /// allocate-then-fill shape. Shares `split_core`, so the two cannot diverge.
    ///
    /// SAFETY: `dst` must be writable, aligned, and not currently live.
    pub unsafe fn split_in_place<S: std::hash::BuildHasher>(&mut self, dst: *mut Table<K, V>,  _key_hash: u64, hash_builder: &S, initial_depth: usize) 
    where
        K: std::hash::Hash,
        V: crate::RoutingValue,
    {
        std::ptr::write(dst, Table::new(self.local_depth + 1));
        self.split_core(&mut *dst, _key_hash, hash_builder, initial_depth);
    }

    /// The one and only implementation of split's migration logic.
    fn split_core<S: std::hash::BuildHasher>(&mut self, new_table: &mut Table<K, V>,  _key_hash: u64, hash_builder: &S, initial_depth: usize) 
    where
        K: std::hash::Hash,
        V: crate::RoutingValue,
    {
        let new_depth = self.local_depth + 1;
        new_table.local_depth = new_depth;
        let new_pattern = (self.pattern << 1) | 1;
        let old_pattern = self.pattern << 1;

        let mut invalid_array = [0u32; TOTAL_BUCKETS];

        // Regular buckets: single pass scan + Insert4split into new table
        for i in 0..K_NUM_BUCKET {
            let mask = self.metas[i].get_bitmap();
            let mut invalid_mask: u32 = 0;
            for j in 0..K_NUM_PAIR_PER_BUCKET {
                if (mask & (1 << j)) != 0 {
                    let value = self.pairs[i][j].value;
                    // Step 5: stay/migrate decision from routing bits (no rehash), falling back
                    // to a rehash only when the depth exceeds the stored routing bits.
                    let migrates = match value.should_migrate(self.local_depth, initial_depth) {
                        Some(m) => m,
                        None => {
                            let key = self.pairs[i][j].key;
                            let key_hash = Self::hash_key_with(hash_builder, &key);
                            (key_hash >> (64 - new_depth)) == new_pattern
                        }
                    };
                    if migrates {
                        invalid_mask |= 1 << j;
                        let key = self.pairs[i][j].key;
                        let meta_hash = self.metas[i].finger_array[j];
                        // Step 6: bucket index in the new table is the entry's primary bucket,
                        // recovered from its physical position — NO rehash. An org entry
                        // (member bit 0) sits in its primary bucket i; a probe entry
                        // (member bit 1) was displaced from primary i-1.
                        let is_probe = (self.metas[i].get_member() >> j) & 1 == 1;
                        let y = if is_probe {
                            if i == 0 { K_NUM_BUCKET - 1 } else { i - 1 }
                        } else {
                            i
                        };
                        new_table.place_split(y, key, value, meta_hash);
                        self.count -= 1;
                    }
                }
            }
            invalid_array[i] = invalid_mask;
        }

        // Stash buckets — the stay/migrate decision still uses routing bits, but the bucket
        // index MUST be rehashed (a stash entry's physical position doesn't identify its
        // primary bucket).
        for i in 0..STASH_BUCKET {
            let bi = K_NUM_BUCKET + i;
            let mask = self.metas[bi].get_bitmap();
            let mut invalid_mask: u32 = 0;
            for j in 0..K_NUM_PAIR_PER_BUCKET {
                if (mask & (1 << j)) != 0 {
                    let value = self.pairs[bi][j].value;
                    let migrates = match value.should_migrate(self.local_depth, initial_depth) {
                        Some(m) => m,
                        None => {
                            let key = self.pairs[bi][j].key;
                            let key_hash = Self::hash_key_with(hash_builder, &key);
                            (key_hash >> (64 - new_depth)) == new_pattern
                        }
                    };
                    if migrates {
                        invalid_mask |= 1 << j;
                        let key = self.pairs[bi][j].key;
                        let key_hash = Self::hash_key_with(hash_builder, &key);
                        let meta_hash = self.metas[bi].finger_array[j];
                        let y = ((key_hash >> K_FINGER_BITS) as usize) & BUCKET_MASK;
                        new_table.place_split(y, key, value, meta_hash);
                        self.count -= 1;
                    }
                }
            }
            invalid_array[bi] = invalid_mask;
        }

        new_table.pattern = new_pattern;
        self.pattern = old_pattern;
        self.local_depth = new_depth;

        // Bulk-clear old table slots using bitmask (matching C++ exactly)
        for i in 0..TOTAL_BUCKETS {
            self.metas[i].bulk_clear(invalid_array[i]);
        }

    }

    /// Place a migrating entry into this (new) table at primary bucket `y`: choose y or its
    /// neighbor by load (matching C++), else overflow to stash. Identical policy to step 6 — only
    /// factored out because both split loops share it. `#[inline(always)]` so the step 6 → step 7
    /// delta stays purely the storage layout (step 6 had this block inline in both loops).
    #[inline(always)]
    fn place_split(&mut self, y: usize, key: K, value: V, meta_hash: u8) {
        let neighbor_idx = (y + 1) & BUCKET_MASK;
        let (insert_idx, probe) = if self.metas[y].count() <= self.metas[neighbor_idx].count() {
            (y, false)
        } else {
            (neighbor_idx, true)
        };
        if self.metas[insert_idx].count() < K_NUM_PAIR_PER_BUCKET {
            let Self { metas, pairs, count, .. } = self;
            metas[insert_idx].insert4split(&mut pairs[insert_idx], key, value, meta_hash, probe);
            *count += 1;
        } else {
            self.stash_insert(y, key, value, meta_hash);
        }
    }

    /// Stash insertion for entries that overflow during split.
    fn stash_insert(&mut self, bucket_idx: usize, key: K, value: V, meta_hash: u8) {
        let stash_idx = bucket_idx & STASH_MASK;
        let Self { metas, pairs, count, .. } = self;
        for i in 0..STASH_BUCKET {
            let si = K_NUM_BUCKET + ((stash_idx + i) & STASH_MASK);
            if metas[si].count() < K_NUM_PAIR_PER_BUCKET {
                metas[si].insert4split(&mut pairs[si], key, value, meta_hash, false);
                *count += 1;
                return;
            }
        }
        // Both stash buckets full during split. A silent drop would lose a key, so make
        // it loud in ALL builds (we run correctness in release). If this ever fires the
        // split-overflow handling needs the full C++ displacement fallback.
        panic!("split stash overflow: entry would be dropped (bucket_idx={bucket_idx})");
    }

    // Rehash a key during split using the map's configured hasher (threaded in).
    fn hash_key_with<S: std::hash::BuildHasher>(hash_builder: &S, key: &K) -> u64
    where
        K: std::hash::Hash,
    {
        use std::hash::Hasher;
        let mut hasher = hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    pub fn len(&self) -> usize {
        self.count
    }

    /// DEBUG: validate every bucket and that segment count == sum of bucket counts.
    pub fn debug_check(&self) -> Result<(), String> {
        let mut sum = 0usize;
        for i in 0..TOTAL_BUCKETS {
            self.metas[i].debug_check().map_err(|e| format!("bucket {i}: {e}"))?;
            sum += self.metas[i].count();
        }
        if sum != self.count {
            return Err(format!("segment count {} != sum bucket counts {}", self.count, sum));
        }
        Ok(())
    }

    /// DEBUG: which bucket (0..66) physically holds `key`, if any.
    pub fn debug_find(&self, key: &K) -> Option<usize>
    where K: PartialEq {
        (0..TOTAL_BUCKETS).find(|&b| self.metas[b].debug_has(&self.pairs[b], key))
    }
    pub fn debug_meta(&self) -> (u64, usize) { (self.pattern, self.local_depth) }

    /// Per-bucket fill histogram for this segment: out[i] = #buckets with i occupied
    /// slots (0..=14). Covers all 64+2 buckets so it matches bucket_count()'s 66/segment.
    pub fn bucket_fill_histogram(&self) -> Vec<u64> {
        let mut out = vec![0u64; 15];
        for i in 0..TOTAL_BUCKETS {
            let c = self.metas[i].count().min(14);
            out[c] += 1;
        }
        out
    }
}

impl<K, V> Clone for Table<K, V>
where
    K: Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    fn clone(&self) -> Self {
        Self {
            metas: self.metas,
            pairs: self.pairs,
            local_depth: self.local_depth,
            pattern: self.pattern,
            count: self.count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_insert_get() {
        let mut table = Table::new(1);
        assert!(table.insert(1u64, 10u64, 0x12345678, 42).is_ok());
        assert_eq!(table.get(&1u64, 0x12345678, 42), Some(&10u64));
    }

    /// Step 7 layout proof: the pairs array is physically separate from the metas array, and each
    /// BucketMeta occupies exactly 32 bytes (so 66 metas span ~2 KiB, scanned without touching
    /// the ~14.8 KiB pairs region on a fingerprint miss).
    #[test]
    fn soa_layout_is_separated() {
        let table: Table<u64, u64> = Table::new(1);
        assert_eq!(std::mem::size_of::<BucketMeta>(), 32);
        // adjacent metas are 32 bytes apart (dense array)
        let m0 = &table.metas[0] as *const _ as usize;
        let m1 = &table.metas[1] as *const _ as usize;
        assert_eq!(m1 - m0, 32);
        // adjacent pairs rows are 14 * size_of::<Pair<u64,u64>>() = 14*16 = 224 bytes apart
        let p0 = &table.pairs[0] as *const _ as usize;
        let p1 = &table.pairs[1] as *const _ as usize;
        assert_eq!(p1 - p0, 14 * std::mem::size_of::<Pair<u64, u64>>());
        // metas and pairs live in disjoint regions (not interleaved per bucket)
        let metas_lo = &table.metas[0] as *const _ as usize;
        let pairs_lo = &table.pairs[0] as *const _ as usize;
        assert!(metas_lo.abs_diff(pairs_lo) >= std::mem::size_of::<[BucketMeta; TOTAL_BUCKETS]>());
    }
}
