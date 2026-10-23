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

// Step 9: write-intent (RFO) software prefetch. In the separated layout a bucket's metadata and
// its KV pairs live on different cache lines, so an insert pays an extra RFO miss on the pair
// line that the interleaved layout avoided. Issuing an exclusive-state prefetch (`_MM_HINT_ET0`)
// for the home+neighbor pair AND meta lines at the top of insert hides that miss behind the
// dup-check + probe. x86_64 only; a no-op elsewhere so the crate still compiles.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_w(ptr: *const u8) {
    unsafe {
        use std::arch::x86_64::*;
        _mm_prefetch::<{ _MM_HINT_ET0 }>(ptr as *const i8);
    }
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn prefetch_w(_ptr: *const u8) {}

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

        let bucket_idx = self.bucket_index(key_hash);
        let neighbor_idx = (bucket_idx + 1) & BUCKET_MASK;

        // Step 9: write-intent prefetch the home+neighbor pair AND meta lines now, so the RFO miss
        // on the (separate-cache-line) pair region is in flight during the dup-check + probe.
        unsafe {
            prefetch_w(self.pairs.as_ptr().add(bucket_idx) as *const u8);
            prefetch_w(self.pairs.as_ptr().add(neighbor_idx) as *const u8);
            prefetch_w(self.metas.as_ptr().add(bucket_idx) as *const u8);
            prefetch_w(self.metas.as_ptr().add(neighbor_idx) as *const u8);
        }

        let duplicate_check = self.get(&key, key_hash, meta_hash);
        if duplicate_check.is_some() {
            return Ok(false);
        }

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

    /// Step 8 — POSITIONAL BLIND-COPY SPLIT + STASH RECOMPACTION. Single change vs
    /// step 7: instead of rebuilding the child by re-inserting each migrant, the child starts as a
    /// byte-for-byte copy of the parent (metas + pairs; the ~15 KB pairs block lowers to one
    /// memcpy), then each regular occupied slot is DROPPED from exactly one side — migrants
    /// from the parent, survivors from the child — leaving every entry at its original slot.
    /// This is lookup-correct because the in-segment bucket index `(hash>>8)&63` is
    /// depth-invariant (survivor and migrant keep the same bucket index, and member/probe
    /// status is unchanged), so no rehash and no re-placement is needed for regular buckets.
    /// The win is at scale: the old rebuild's scattered re-insertion cache-misses once the
    /// arena leaves cache; the sequential copy barely moves (measured −42% split_p50 at 10M).
    ///
    /// Stashes CANNOT be positionally split (a stash slot's position doesn't encode its home
    /// bucket), and a blind copy would hand the child ~half a full stash. So both stashes are
    /// emptied and each occupant re-homed into its destination table's regular buckets
    /// (recompaction) — which also packs better than the old rebuild.
    /// By-value split. Thin wrapper over `split_core`; retained for callers wanting an owned
    /// Table and for the no-reservable-slot fallback.
    ///
    /// PORTING NOTE. Returning a whole Table by value is a Rust artefact, not the original's
    /// shape: C++ DASH allocates the child through the allocator and fills it in place via a
    /// pointer (DASH's ex_finger.h:1303-1320). Fidelity restoration applied uniformly, NOT an
    /// optimisation credited to step 8. NOTE step 8 also changes split
    /// MECHANICS (positional blind copy); that remains its contribution and is measured on
    /// top of the already-normalised construction shape.
    pub fn split<S: std::hash::BuildHasher>(&mut self, _key_hash: u64, hash_builder: &S, initial_depth: usize) -> Table<K, V>
    where
        K: std::hash::Hash,
        V: crate::RoutingValue,
    {
        let mut new_table = Table::new(self.local_depth + 1);
        new_table.metas = self.metas;
        new_table.pairs = self.pairs;
        self.split_core(&mut new_table, _key_hash, hash_builder, initial_depth);
        new_table
    }

    /// Split, constructing the child directly in `dst`. Shares `split_core` with the
    /// by-value path, so the two cannot diverge.
    ///
    /// SAFETY: `dst` must be writable, aligned, and not currently live.
    pub unsafe fn split_in_place<S: std::hash::BuildHasher>(&mut self, dst: *mut Table<K, V>, _key_hash: u64, hash_builder: &S, initial_depth: usize)
    where
        K: std::hash::Hash,
        V: crate::RoutingValue,
    {
        // Copying the whole parent both performs the positional blind copy AND fully
        // initialises dst, so the following &mut *dst is valid.
        std::ptr::copy_nonoverlapping(self as *const Table<K, V>, dst, 1);
        let child = &mut *dst;
        child.local_depth = self.local_depth + 1;
        self.split_core(child, _key_hash, hash_builder, initial_depth);
    }

    /// The one and only implementation of this step's split logic. `new_table` must
    /// already hold the positional copy of the parent's planes.
    fn split_core<S: std::hash::BuildHasher>(&mut self, new_table: &mut Table<K, V>, _key_hash: u64, hash_builder: &S, initial_depth: usize)
    where
        K: std::hash::Hash,
        V: crate::RoutingValue,
    {
        let new_depth = self.local_depth + 1;
        new_table.local_depth = new_depth;
        let new_pattern = (self.pattern << 1) | 1;
        let old_pattern = self.pattern << 1;

        // (1) Counts before. The step 7 layout has no cached stash_count, so derive it from the two stashes.
        let count_before = self.count;
        let stash_before = self.metas[K_NUM_BUCKET].count() + self.metas[K_NUM_BUCKET + 1].count();
        let regular_before = count_before - stash_before;

        // (2) Child already holds the positional copy of the parent (done by the caller).

        // (3) Regular buckets: positional clear. Drop migrants from the parent, survivors from
        // the child; both stay in the same bucket index (depth-invariant) — no rehash, no move.
        let mut new_reg = 0usize;
        for bi in 0..K_NUM_BUCKET {
            let mut slot_mask = self.metas[bi].get_bitmap();
            while slot_mask != 0 {
                let j = slot_mask.trailing_zeros() as usize;
                slot_mask &= slot_mask - 1;

                let value = self.pairs[bi][j].value;
                let migrates = match value.should_migrate(self.local_depth, initial_depth) {
                    Some(m) => m,
                    None => {
                        let key = self.pairs[bi][j].key;
                        let kh = Self::hash_key_with(hash_builder, &key);
                        (kh >> (64 - new_depth)) == new_pattern
                    }
                };

                if migrates {
                    self.metas[bi].clear_slot(j);       // drop from parent (survivors stay)
                    new_reg += 1;
                } else {
                    new_table.metas[bi].clear_slot(j);  // drop from child (migrants stay)
                }
            }
        }

        // (4) Snapshot parent-stash occupants, rehashing each (the step 7 stash stores no key hash).
        let mut items: [(K, V, u64); STASH_BUCKET * K_NUM_PAIR_PER_BUCKET] =
            [(K::default(), V::default(), 0u64); STASH_BUCKET * K_NUM_PAIR_PER_BUCKET];
        let mut nitems = 0usize;
        for si in 0..STASH_BUCKET {
            let sbi = K_NUM_BUCKET + si;
            let mut slot_mask = self.metas[sbi].get_bitmap();
            while slot_mask != 0 {
                let j = slot_mask.trailing_zeros() as usize;
                slot_mask &= slot_mask - 1;
                let key = self.pairs[sbi][j].key;
                let value = self.pairs[sbi][j].value;
                let kh = Self::hash_key_with(hash_builder, &key);
                items[nitems] = (key, value, kh);
                nitems += 1;
            }
        }

        // (5) Clear BOTH stashes in parent and child (metadata is the source of truth; the
        // copied pair bytes are left as unreachable garbage).
        for si in 0..STASH_BUCKET {
            let sbi = K_NUM_BUCKET + si;
            self.metas[sbi] = BucketMeta::new();
            new_table.metas[sbi] = BucketMeta::new();
        }

        // (6) Commit depths/patterns and regular-only counts BEFORE reinserting (so the stash
        // fallback in reinsert uses the new depth; and counts are the regular-only totals that
        // reinsert then bumps).
        new_table.pattern = new_pattern;
        self.pattern = old_pattern;
        self.local_depth = new_depth;
        let old_reg = regular_before - new_reg;
        self.count = old_reg;
        new_table.count = new_reg;

        // (7) Re-home each stash occupant into the parent or child by its rehashed routing.
        for k in 0..nitems {
            let (key, value, kh) = items[k];
            let meta_hash = (kh & 0xFF) as u8;
            if (kh >> (64 - new_depth)) == new_pattern {
                new_table.reinsert_after_split(key, value, kh, meta_hash);
            } else {
                self.reinsert_after_split(key, value, kh, meta_hash);
            }
        }

        // (8) Nothing lost or duplicated — hard assert (we run correctness in release).
        assert_eq!(self.count + new_table.count, count_before,
            "split lost/gained entries: before={} old={} new={}", count_before, self.count, new_table.count);

    }

    /// Re-home a known-unique entry pulled from a cleared stash into THIS table's regular
    /// buckets: the less-full of home/neighbor (matching C++), else the stash. Uses `insert`
    /// (not `insert4split`) because positional clearing leaves holes, so the slot must come
    /// from `find_empty_slot`, not `count`. `count` was pre-set to the regular-only total.
    fn reinsert_after_split(&mut self, key: K, value: V, kh: u64, meta_hash: u8) {
        let bi = ((kh >> K_FINGER_BITS) as usize) & BUCKET_MASK;
        let neighbor = (bi + 1) & BUCKET_MASK;
        let Self { metas, pairs, count, .. } = self;
        let (first, second, p1, p2) = if metas[bi].count() <= metas[neighbor].count() {
            (bi, neighbor, false, true)
        } else {
            (neighbor, bi, true, false)
        };
        if metas[first].insert(&mut pairs[first], key, value, meta_hash, p1).is_ok() {
            *count += 1;
            return;
        }
        if metas[second].insert(&mut pairs[second], key, value, meta_hash, p2).is_ok() {
            *count += 1;
            return;
        }
        // Home + neighbor full just after a split: fall back to stash. Capacity (2×14) ≥ the
        // ≤28 entries re-homed per table, so this cannot overflow — assert if it ever does.
        let stash_idx = bi & STASH_MASK;
        for i in 0..STASH_BUCKET {
            let si = K_NUM_BUCKET + ((stash_idx + i) & STASH_MASK);
            if metas[si].insert(&mut pairs[si], key, value, meta_hash, false).is_ok() {
                *count += 1;
                return;
            }
        }
        panic!("split recompaction: stash overflow (impossible)");
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
