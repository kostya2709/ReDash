use crate::directory::Directory;
use crate::allocator::{TableAllocator, DefaultAllocator, PoolAllocator};
use std::hash::{Hash, Hasher, BuildHasher, BuildHasherDefault};
use std::collections::hash_map::DefaultHasher;

// Constants matching the C++ implementation
const K_NUM_PAIR_PER_BUCKET: usize = 14;
const K_FINGER_BITS: usize = 8;
const K_MASK: u64 = (1 << K_FINGER_BITS) - 1;
const K_NUM_BUCKET: usize = 64;
const BUCKET_MASK: usize = K_NUM_BUCKET - 1;

/// SipHash (std DefaultHasher) — the steps 0–3 hasher, kept for the comparison.
pub type SipHashBuilder = BuildHasherDefault<DefaultHasher>;
/// Step 4 default hasher: FoldHash with a FIXED seed (deterministic). The comparison winner
/// the ladder continues with; other hashers can be swapped in via with_hasher.
pub type DefaultHashBuilder = foldhash::fast::FixedState;

/// Dash Extendible Hash Map. Generic over allocator `A` and hash builder `S`. Step 4's
/// default is FoldHash; split() threads `S` so any hasher is correct.
pub struct DashExtendibleHashMap<K, V, A: TableAllocator<K, V> = DefaultAllocator, S: BuildHasher = DefaultHashBuilder> {
    directory: Directory<K, V, A>,
    hash_builder: S,
}

// `new()` / `with_pool_allocator()` default to FoldHash (the step 4 change).
impl<K, V> DashExtendibleHashMap<K, V, DefaultAllocator, DefaultHashBuilder>
where
    K: Hash + Eq + Clone + Copy + Default + std::fmt::Debug,
    V: Clone + Copy + Default,
{
    /// Create with the default allocator + FoldHash.
    pub fn new() -> Self {
        Self {
            directory: Directory::new(),
            hash_builder: DefaultHashBuilder::default(),
        }
    }
}

impl<K, V> DashExtendibleHashMap<K, V, PoolAllocator<K, V>, DefaultHashBuilder>
where
    K: Hash + Eq + Clone + Copy + Default + std::fmt::Debug,
    V: Clone + Copy + Default,
{
    /// Create with a pre-faulted pool allocator + FoldHash, sized for `capacity` segments.
    pub fn with_pool_allocator(capacity: usize) -> Self {
        Self {
            directory: Directory::with_allocator(PoolAllocator::new(capacity)),
            hash_builder: DefaultHashBuilder::default(),
        }
    }

    /// # of pool-exhaustion fallbacks (assert 0 in a clean perf run).
    pub fn pool_exhausted_count(&self) -> usize { self.directory.exhausted_count() }
}

// `with_hasher` on the default allocator (API parity with steps 0–2; seam for step 4).
impl<K, V, S> DashExtendibleHashMap<K, V, DefaultAllocator, S>
where
    K: Hash + Eq + Clone + Copy + Default + std::fmt::Debug,
    V: Clone + Copy + Default,
    S: BuildHasher,
{
    pub fn with_hasher(hash_builder: S) -> Self {
        Self { directory: Directory::new(), hash_builder }
    }
}

impl<K, V, A, S> DashExtendibleHashMap<K, V, A, S>
where
    K: Hash + Eq + Clone + Copy + Default + std::fmt::Debug,
    V: Clone + Copy + Default,
    A: TableAllocator<K, V>,
    S: BuildHasher,
{
    /// Create with an explicit allocator + hash builder.
    pub fn with_allocator_and_hasher(allocator: A, hash_builder: S) -> Self {
        Self {
            directory: Directory::with_allocator(allocator),
            hash_builder,
        }
    }

    /// As `with_allocator_and_hasher`, but with an explicit initial directory size.
    /// Benchmarking only -- see `Directory::with_allocator_and_cap`. Lets step 4 be compared
    /// against the layout variants at a matched starting configuration (64 segments).
    pub fn with_allocator_hasher_and_cap(allocator: A, hash_builder: S, init_cap: usize) -> Self {
        Self {
            directory: Directory::with_allocator_and_cap(allocator, init_cap),
            hash_builder,
        }
    }

    /// Insert a key-value pair
    /// Returns true if inserted, false if key already exists
    pub fn insert(&mut self, key: K, value: V) -> bool {
        let key_hash = self.hash_key(&key);
        let meta_hash = (key_hash & K_MASK) as u8;
        let mut retry_count = 0;
        
        loop {
            let global_depth = self.directory.global_depth();
            let dir_entry_idx = (key_hash >> (64 - global_depth)) as usize;
            
            let needs_split = {
                let table = self.directory.table_mut(dir_entry_idx);

                let table_local_depth = table.local_depth();
                let table_pattern = table.pattern;
                let key_pattern = key_hash >> (64 - table_local_depth);
                if key_pattern != table_pattern {
                    retry_count += 1;
                    if retry_count > 10 {
                        return false;
                    }
                    continue;
                }

                match table.insert(key.clone(), value.clone(), key_hash, meta_hash) {
                    Ok(inserted) => {
                        if !inserted {
                            let exists = self.directory.table_ref(dir_entry_idx)
                                .get(&key, key_hash, meta_hash).is_some();
                            if !exists {
                                retry_count += 1;
                                if retry_count > 10 {
                                    return false;
                                }
                                continue;
                            }
                        }
                        return inserted;
                    }
                    Err(split) => split,
                }
            };

            if needs_split {
                let table = self.directory.table_ref(dir_entry_idx);
                let local_depth = table.local_depth();
                let pattern = table.pattern;

                let new_pattern = (pattern << 1) | 1;
                let old_pattern = pattern << 1;

                if local_depth < global_depth {
                    // Original's allocate-then-fill shape; by-value only as fallback.
                    if !self.directory.split_into_slot_and_update(
                        dir_entry_idx, key_hash, &self.hash_builder, local_depth)
                    {
                        let new_table = self.directory.table_mut(dir_entry_idx).split(key_hash, &self.hash_builder);
                        self.directory.update_directory_after_split(
                            dir_entry_idx, new_table, old_pattern, new_pattern, local_depth
                        );
                    }
                } else {
                    let old_dir_entry_idx = dir_entry_idx;
                    let pre = self.directory.split_into_slot_at(old_dir_entry_idx, key_hash, &self.hash_builder);
                    let owned = if pre.is_none() {
                        Some(self.directory.table_mut(old_dir_entry_idx).split(key_hash, &self.hash_builder))
                    } else { None };
                    self.directory.double_directory();
                    let doubled_idx_new = old_dir_entry_idx * 2 + 1;
                    match (pre, owned) {
                        (Some(ptr), _) => self.directory.set_allocated_table(doubled_idx_new, ptr),
                        (None, Some(t)) => self.directory.allocate_and_set_table(doubled_idx_new, t),
                        _ => unreachable!("exactly one split path runs"),
                    }
                }
                retry_count += 1;
                if retry_count > 10 {
                    return false;
                }
                continue;
            }

            return false;
        }
    }

    /// Get a value by key
    pub fn get(&self, key: &K) -> Option<V> 
    where 
        V: Copy,
    {
        let key_hash = self.hash_key(key);
        let meta_hash = (key_hash & K_MASK) as u8;
        
        let dir_entry_idx = (key_hash >> (64 - self.directory.global_depth())) as usize;
        self.directory.table_ref(dir_entry_idx).get(key, key_hash, meta_hash).copied()
    }

    /// Remove a key-value pair
    pub fn remove(&mut self, key: &K) -> bool {
        let key_hash = self.hash_key(key);
        let meta_hash = (key_hash & K_MASK) as u8;
        let dir_entry_idx = (key_hash >> (64 - self.directory.global_depth())) as usize;
        self.directory.table_mut(dir_entry_idx).remove(key, key_hash, meta_hash)
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.directory.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get bucket count (for tracking rehashes)
    pub fn bucket_count(&self) -> usize {
        self.directory.bucket_count()
    }

    /// O(1) count of unique segments (instrumentation; harness polls for split detection).
    pub fn segment_count(&self) -> usize {
        self.directory.segment_count()
    }

    /// Get global depth of the directory
    pub fn global_depth(&self) -> usize {
        self.directory.global_depth()
    }

    /// Get directory size (number of directory entries)
    pub fn directory_size(&self) -> usize {
        self.directory.table_count()
    }

    fn hash_key(&self, key: &K) -> u64 {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Bucket fill-rate histogram across all unique segments.
    /// `out[i]` = number of buckets with exactly `i` occupied slots (0..=14).
    pub fn dash_map_fill_histogram(&self) -> Vec<u64> {
        self.directory.fill_histogram()
    }

    /// DEBUG: full structural validation. Per-bucket count/bitmap invariants,
    /// per-segment count == sum bucket counts, and total live entries == len().
    pub fn debug_validate(&self) -> Result<(), String> {
        let total = self.directory.debug_validate()?;
        if total != self.len() {
            return Err(format!("sum of segment counts {} != len() {}", total, self.len()));
        }
        Ok(())
    }

    /// DEBUG: for a (presumably missing) key, report routing vs physical location.
    /// Returns (routed_dir_idx, routed_pattern, routed_local_depth, found_by_get,
    ///          physical: Option<(dir_idx, bucket, pattern, local_depth)>).
    pub fn debug_locate(&self, key: &K) -> (usize, u64, usize, bool,
                                            Option<(usize, usize, u64, usize)>)
    where K: PartialEq {
        let key_hash = self.hash_key(key);
        let gd = self.directory.global_depth();
        let routed = (key_hash >> (64 - gd)) as usize;
        let (rp, rd) = self.directory.table_ref(routed).debug_meta();
        let by_get = self.get(key).is_some();
        let phys = self.directory.debug_find_any(key);
        (routed, rp, rd, by_get, phys)
    }
}

impl<K, V> Default for DashExtendibleHashMap<K, V, DefaultAllocator, DefaultHashBuilder>
where
    K: Hash + Eq + Clone + Copy + Default + std::fmt::Debug,
    V: Clone + Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_get() {
        let mut map = DashExtendibleHashMap::new();
        assert!(map.insert(1u64, 10u64));
        assert_eq!(map.get(&1), Some(10));
    }

    #[test]
    fn test_duplicate_insert() {
        let mut map = DashExtendibleHashMap::new();
        assert!(map.insert(1u64, 10u64));
        assert!(!map.insert(1u64, 20u64)); // Duplicate
        assert_eq!(map.get(&1), Some(10)); // Original value
    }

    #[test]
    fn test_remove() {
        let mut map = DashExtendibleHashMap::new();
        map.insert(1u64, 10u64);
        assert!(map.remove(&1));
        assert_eq!(map.get(&1), None);
    }

    #[test]
    fn test_many_inserts() {
        let mut map = DashExtendibleHashMap::new();
        for i in 0..1000 {
            assert!(map.insert(i, i * 2));
        }
        for i in 0..1000 {
            assert_eq!(map.get(&i), Some(i * 2));
        }
    }

    #[test]
    fn test_len() {
        let mut map = DashExtendibleHashMap::new();
        assert_eq!(map.len(), 0);
        map.insert(1u64, 10u64);
        assert_eq!(map.len(), 1);
        map.insert(2u64, 20u64);
        assert_eq!(map.len(), 2);
        map.remove(&1);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_bucket_count_tracking() {
        let mut map = DashExtendibleHashMap::new();
        let initial_count = map.bucket_count();
        
        for i in 0..1000 {
            map.insert(i, i * 2);
        }
        
        assert!(map.bucket_count() >= initial_count);
    }
}
