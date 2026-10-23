// Table/Segment implementation for Dash extendible hashing

use crate::bucket::{Bucket, K_NUM_PAIR_PER_BUCKET};

const K_NUM_BUCKET: usize = 64;
const STASH_BUCKET: usize = 2;
const BUCKET_MASK: usize = K_NUM_BUCKET - 1;
const STASH_MASK: usize = STASH_BUCKET - 1;
const K_FINGER_BITS: usize = 8;

pub struct Table<K, V> {
    buckets: [Bucket<K, V>; K_NUM_BUCKET + STASH_BUCKET],
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
        let buckets = std::array::from_fn(|_| Bucket::new());
        Self {
            buckets,
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
        
        let target_count = self.buckets[bucket_idx].count();
        let neighbor_count = self.buckets[neighbor_idx].count();
        let use_neighbor = neighbor_count < target_count;
        let probe = use_neighbor;

        let insert_target_idx = if use_neighbor { neighbor_idx } else { bucket_idx };
        
        if !self.buckets[insert_target_idx].is_full() {
            match self.buckets[insert_target_idx].insert(key, value, meta_hash, probe) {
                Ok(_) => {
                    self.count += 1;
                    return Ok(true);
                }
                Err(_) => {}
            }
        }

        if self.buckets[bucket_idx].is_full() && self.buckets[neighbor_idx].is_full() {
            // Try next neighbor displacement (matching C++ Next_displace)
            let next_neighbor_idx = (bucket_idx + 2) & BUCKET_MASK;
            if let Some(displace_idx) = self.buckets[neighbor_idx].find_org_displacement() {
                if !self.buckets[next_neighbor_idx].is_full() {
                    if let Some((displaced_key, displaced_value, displaced_fp)) = 
                        self.buckets[neighbor_idx].get_pair(displace_idx) {
                        // NB: keep the calls OUTSIDE debug_assert! — its argument is not
                        // evaluated in release builds, which would skip these side effects.
                        let ins_ok = self.buckets[next_neighbor_idx].insert(
                            displaced_key, displaced_value, displaced_fp, true
                        ).is_ok();
                        debug_assert!(ins_ok, "Next_displace: dest insert must succeed");
                        // BUGFIX (validity): the displaced entry is an ORG (non-probe)
                        // entry, so it must be removed with probe=false. The original
                        // used true, so remove() silently failed.
                        let rem_ok = self.buckets[neighbor_idx].remove(&displaced_key, displaced_fp, false);
                        debug_assert!(rem_ok, "Next_displace: source remove must succeed");
                        self.buckets[neighbor_idx].insert_displace(key, value, meta_hash, displace_idx, true);
                        self.count += 1;
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
            if let Some(displace_idx) = self.buckets[bucket_idx].find_probe_displacement() {
                if !self.buckets[prev_neighbor_idx].is_full() {
                    if let Some((displaced_key, displaced_value, displaced_fp)) = 
                        self.buckets[bucket_idx].get_pair(displace_idx) {
                        let ins_ok = self.buckets[prev_neighbor_idx].insert(
                            displaced_key, displaced_value, displaced_fp, false
                        ).is_ok();
                        debug_assert!(ins_ok, "Prev_displace: dest insert must succeed");
                        let rem_ok = self.buckets[bucket_idx].remove(&displaced_key, displaced_fp, true);
                        debug_assert!(rem_ok, "Prev_displace: source remove must succeed");
                        self.buckets[bucket_idx].insert_displace(key, value, meta_hash, displace_idx, false);
                        self.count += 1;
                        return Ok(true);
                    }
                }
            }

            // Try stash (matching C++ Stash_insert)
            let stash_idx = bucket_idx & STASH_MASK;
            for i in 0..STASH_BUCKET {
                let stash_bucket_idx = K_NUM_BUCKET + ((stash_idx + i) & STASH_MASK);
                if !self.buckets[stash_bucket_idx].is_full() {
                    match self.buckets[stash_bucket_idx].insert(key, value, meta_hash, false) {
                        Ok(_) => {
                            self.count += 1;
                            return Ok(true);
                        }
                        Err(_) => continue,
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
        
        if let Some(value) = self.buckets[bucket_idx].get(key, meta_hash, false) {
            return Some(value);
        }
        
        let neighbor_idx = (bucket_idx + 1) & BUCKET_MASK;
        if let Some(value) = self.buckets[neighbor_idx].get(key, meta_hash, true) {
            return Some(value);
        }
        
        let stash_idx = bucket_idx & STASH_MASK;
        for i in 0..STASH_BUCKET {
            let stash_bucket_idx = K_NUM_BUCKET + ((stash_idx + i) & STASH_MASK);
            if let Some(value) = self.buckets[stash_bucket_idx].get(key, meta_hash, false) {
                return Some(value);
            }
        }
        
        None
    }

    pub fn remove(&mut self, key: &K, key_hash: u64, meta_hash: u8) -> bool {
        let bucket_idx = self.bucket_index(key_hash);
        
        if self.buckets[bucket_idx].remove(key, meta_hash, false) {
            self.count -= 1;
            return true;
        }
        
        let neighbor_idx = (bucket_idx + 1) & BUCKET_MASK;
        if self.buckets[neighbor_idx].remove(key, meta_hash, true) {
            self.count -= 1;
            return true;
        }
        
        let stash_idx = bucket_idx & STASH_MASK;
        for i in 0..STASH_BUCKET {
            let stash_bucket_idx = K_NUM_BUCKET + ((stash_idx + i) & STASH_MASK);
            if self.buckets[stash_bucket_idx].remove(key, meta_hash, false) {
                self.count -= 1;
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
    // Step 4: split rehashes keys with the MAP's configured hasher (threaded in), not a
    // hardcoded DefaultHasher. Required for FoldHash (or any non-SipHash) to be correct.
    /// By-value split. Thin wrapper over `split_core`; retained for callers wanting an owned
    /// Table and for the no-reservable-slot fallback.
    ///
    /// PORTING NOTE. Returning a whole Table by value is a Rust artefact, not the original's
    /// shape: C++ DASH allocates the child through the allocator and fills it in place via a
    /// pointer, returning `Table<T>*` (DASH's ex_finger.h:1303-1320). The by-value
    /// form forces the caller to reserve a segment-sized return slot in its own frame and
    /// then copy the finished child into place. This is FIDELITY RESTORATION, not one of our
    /// optimisations: it is applied uniformly to every implementation, so no variant gains
    /// from it and its cost is not attributed to any step.
    pub fn split<S: std::hash::BuildHasher>(&mut self, _key_hash: u64, hash_builder: &S) -> Table<K, V>
    where
        K: std::hash::Hash,
    {
        let mut new_table = Table::new(self.local_depth + 1);
        self.split_core(&mut new_table, hash_builder);
        new_table
    }

    /// Split, constructing the child directly in `dst` -- the original's allocate-then-fill
    /// shape. Shares `split_core` with the by-value path, so the two cannot diverge.
    ///
    /// SAFETY: `dst` must be writable, aligned for `Table<K, V>`, and not currently live.
    pub unsafe fn split_in_place<S: std::hash::BuildHasher>(
        &mut self, dst: *mut Table<K, V>, hash_builder: &S)
    where
        K: std::hash::Hash,
    {
        std::ptr::write(dst, Table::new(self.local_depth + 1));
        self.split_core(&mut *dst, hash_builder);
    }

    /// The one and only implementation of split's migration logic.
    fn split_core<S: std::hash::BuildHasher>(&mut self, new_table: &mut Table<K, V>, hash_builder: &S)
    where
        K: std::hash::Hash,
    {
        let new_depth = self.local_depth + 1;
        new_table.local_depth = new_depth;
        let new_pattern = (self.pattern << 1) | 1;
        let old_pattern = self.pattern << 1;

        let mut invalid_array = [0u32; K_NUM_BUCKET + STASH_BUCKET];

        // Regular buckets: single pass scan + Insert4split into new table
        for i in 0..K_NUM_BUCKET {
            let mask = self.buckets[i].get_bitmap();
            let mut invalid_mask: u32 = 0;
            for j in 0..K_NUM_PAIR_PER_BUCKET {
                if (mask & (1 << j)) != 0 {
                    let key = self.buckets[i].pairs[j].key;
                    let value = self.buckets[i].pairs[j].value;
                    let key_hash = Self::hash_key_with(hash_builder, &key);
                    if (key_hash >> (64 - new_depth)) == new_pattern {
                        invalid_mask |= 1 << j;
                        let meta_hash = self.buckets[i].finger_array[j];
                        let y = ((key_hash >> K_FINGER_BITS) as usize) & BUCKET_MASK;
                        let neighbor_idx = (y + 1) & BUCKET_MASK;
                        let (insert_idx, probe) = if new_table.buckets[y].count() <= new_table.buckets[neighbor_idx].count() {
                            (y, false)
                        } else {
                            (neighbor_idx, true)
                        };
                        if new_table.buckets[insert_idx].count() < K_NUM_PAIR_PER_BUCKET {
                            new_table.buckets[insert_idx].insert4split(key, value, meta_hash, probe);
                            new_table.count += 1;
                        } else {
                            new_table.stash_insert(y, key, value, key_hash, meta_hash);
                        }
                        self.count -= 1;
                    }
                }
            }
            invalid_array[i] = invalid_mask;
        }

        // Stash buckets
        for i in 0..STASH_BUCKET {
            let bi = K_NUM_BUCKET + i;
            let mask = self.buckets[bi].get_bitmap();
            let mut invalid_mask: u32 = 0;
            for j in 0..K_NUM_PAIR_PER_BUCKET {
                if (mask & (1 << j)) != 0 {
                    let key = self.buckets[bi].pairs[j].key;
                    let value = self.buckets[bi].pairs[j].value;
                    let key_hash = Self::hash_key_with(hash_builder, &key);
                    if (key_hash >> (64 - new_depth)) == new_pattern {
                        invalid_mask |= 1 << j;
                        let meta_hash = self.buckets[bi].finger_array[j];
                        let y = ((key_hash >> K_FINGER_BITS) as usize) & BUCKET_MASK;
                        let neighbor_idx = (y + 1) & BUCKET_MASK;
                        let (insert_idx, probe) = if new_table.buckets[y].count() <= new_table.buckets[neighbor_idx].count() {
                            (y, false)
                        } else {
                            (neighbor_idx, true)
                        };
                        if new_table.buckets[insert_idx].count() < K_NUM_PAIR_PER_BUCKET {
                            new_table.buckets[insert_idx].insert4split(key, value, meta_hash, probe);
                            new_table.count += 1;
                        } else {
                            new_table.stash_insert(y, key, value, key_hash, meta_hash);
                        }
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
        for i in 0..(K_NUM_BUCKET + STASH_BUCKET) {
            self.buckets[i].bulk_clear(invalid_array[i]);
        }

    }

    /// Stash insertion for entries that overflow during split.
    fn stash_insert(&mut self, bucket_idx: usize, key: K, value: V, _key_hash: u64, meta_hash: u8) {
        let stash_idx = bucket_idx & STASH_MASK;
        for i in 0..STASH_BUCKET {
            let si = K_NUM_BUCKET + ((stash_idx + i) & STASH_MASK);
            if self.buckets[si].count() < K_NUM_PAIR_PER_BUCKET {
                self.buckets[si].insert4split(key, value, meta_hash, false);
                self.count += 1;
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
        for (i, b) in self.buckets.iter().enumerate() {
            b.debug_check().map_err(|e| format!("bucket {i}: {e}"))?;
            sum += b.count();
        }
        if sum != self.count {
            return Err(format!("segment count {} != sum bucket counts {}", self.count, sum));
        }
        Ok(())
    }

    /// DEBUG: which bucket (0..66) physically holds `key`, if any.
    pub fn debug_find(&self, key: &K) -> Option<usize>
    where K: PartialEq {
        (0..K_NUM_BUCKET + STASH_BUCKET).find(|&b| self.buckets[b].debug_has(key))
    }
    pub fn debug_meta(&self) -> (u64, usize) { (self.pattern, self.local_depth) }

    /// Per-bucket fill histogram for this segment: out[i] = #buckets with i occupied
    /// slots (0..=14). Covers all 64+2 buckets so it matches bucket_count()'s 66/segment.
    pub fn bucket_fill_histogram(&self) -> Vec<u64> {
        let mut out = vec![0u64; 15];
        for b in self.buckets.iter() {
            let c = b.count().min(14);
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
            buckets: self.buckets.clone(),
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
}
