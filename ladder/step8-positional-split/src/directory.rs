// Directory implementation for Dash extendible hashing.
// Step 3: raw `*mut Table` directory (from step 2) + an allocator abstraction. Segment
// memory now comes from a `TableAllocator`; the directory frees pointers on drop ONLY if
// the allocator does not own its memory (DefaultAllocator), so a pool-owned arena isn't
// double-freed.
use crate::table::Table;
use crate::allocator::{TableAllocator, DefaultAllocator};

pub struct Directory<K, V, A: TableAllocator<K, V> = DefaultAllocator> {
    tables: Vec<*mut Table<K, V>>,
    global_depth: usize,
    allocator: A,
    unique_tables: usize,
}

impl<K, V> Directory<K, V, DefaultAllocator>
where
    K: Eq + Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    pub fn new() -> Self {
        Self::with_allocator(DefaultAllocator)
    }
}

impl<K, V, A: TableAllocator<K, V>> Directory<K, V, A>
where
    K: Eq + Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    pub fn with_allocator(mut allocator: A) -> Self {
        let initial_depth = 1;
        let initial_size = 1 << initial_depth;
        let mut tables = Vec::with_capacity(initial_size);
        for i in 0..initial_size {
            let mut table = Table::new(initial_depth);
            table.pattern = i as u64; // pattern == directory index (matching C++)
            tables.push(allocator.allocate(table));
        }
        Self { tables, global_depth: initial_depth, allocator, unique_tables: initial_size }
    }

    /// # of pool-exhaustion fallbacks (0 for DefaultAllocator / a well-sized pool).
    pub fn exhausted_count(&self) -> usize { self.allocator.exhausted_count() }

    #[inline(always)]
    pub fn global_depth(&self) -> usize {
        self.global_depth
    }

    /// Immutable segment access (no Rc clone, no borrow check, no bounds check).
    /// `pub(crate)`: directory entries alias, so this must not escape the crate.
    #[inline(always)]
    pub(crate) fn table_ref(&self, idx: usize) -> &Table<K, V> {
        let idx = idx & ((1 << self.global_depth) - 1);
        // SAFETY: idx masked to directory size; pointer is valid (we own the box).
        unsafe { &**self.tables.get_unchecked(idx) }
    }

    /// Mutable segment access. Takes `&mut self` (NOT `&self`) so the borrow checker
    /// forbids minting two aliasing `&mut Table` — even though directory entries alias
    /// the same box, the exclusive `&mut self` makes that sound. Callers must let the
    /// returned reference die before any other directory access (NLL enforces this).
    #[inline(always)]
    pub(crate) fn table_mut(&mut self, idx: usize) -> &mut Table<K, V> {
        let idx = idx & ((1 << self.global_depth) - 1);
        // SAFETY: &mut self guarantees no other live borrow of any segment.
        unsafe { &mut **self.tables.get_unchecked(idx) }
    }

    /// Split writing the child straight into an allocator slot -- the C++ original's
    /// allocate-then-fill shape. False when no slot is reservable (caller falls back).
    pub fn split_into_slot_and_update<S: std::hash::BuildHasher>(
        &mut self, dir_entry_idx: usize, key_hash: u64, hash_builder: &S,
        initial_depth: usize, old_local_depth: usize) -> bool
    where
        K: std::hash::Hash,
        V: crate::RoutingValue,
    {
        let Some(dst) = self.allocator.allocate_uninit() else { return false };
        let table = self.table_mut(dir_entry_idx) as *mut Table<K, V>;
        // SAFETY: dst is a fresh, aligned, non-live slot; table is a distinct live segment.
        unsafe { (*table).split_in_place(dst, key_hash, hash_builder, initial_depth); }
        self.unique_tables += 1;
        let new_local_depth = old_local_depth + 1;
        let depth_diff = self.global_depth - new_local_depth;
        self.splice_child(dir_entry_idx, dst, depth_diff, new_local_depth);
        true
    }

    /// Doubling-path variant.
    pub fn split_into_slot_at<S: std::hash::BuildHasher>(
        &mut self, src_idx: usize, key_hash: u64, hash_builder: &S,
        initial_depth: usize) -> Option<*mut Table<K, V>>
    where
        K: std::hash::Hash,
        V: crate::RoutingValue,
    {
        let dst = self.allocator.allocate_uninit()?;
        let table = self.table_mut(src_idx) as *mut Table<K, V>;
        unsafe { (*table).split_in_place(dst, key_hash, hash_builder, initial_depth); }
        Some(dst)
    }

    /// Place an already-constructed child and count it.
    pub fn set_allocated_table(&mut self, idx: usize, ptr: *mut Table<K, V>) {
        self.set_table(idx, ptr);
        self.unique_tables += 1;
    }

    pub fn update_directory_after_split(&mut self, dir_entry_idx: usize, new_table: Table<K, V>, _old_pattern: u64, _new_pattern: u64, old_local_depth: usize) {
        let new_local_depth = old_local_depth + 1;
        let depth_diff = self.global_depth - new_local_depth;

        let new_ptr = self.allocator.allocate(new_table);
        self.unique_tables += 1; // one new segment

        self.splice_child(dir_entry_idx, new_ptr, depth_diff, new_local_depth);
    }

    /// Shared directory-splice logic used by the by-value and in-place paths.
    fn splice_child(&mut self, dir_entry_idx: usize, new_ptr: *mut Table<K, V>,
                    depth_diff: usize, new_local_depth: usize) {
        if depth_diff == 0 {
            if dir_entry_idx % 2 == 0 {
                let update_idx = dir_entry_idx + 1;
                if update_idx < self.tables.len() {
                    self.tables[update_idx] = new_ptr;
                }
            } else if dir_entry_idx < self.tables.len() {
                self.tables[dir_entry_idx] = new_ptr;
            }
        } else {
            let chunk_size = 1 << (self.global_depth - (new_local_depth - 1));
            let aligned_idx = dir_entry_idx - (dir_entry_idx % chunk_size);
            let base = chunk_size / 2;
            for i in 0..base {
                let idx = aligned_idx + base + i;
                if idx < self.tables.len() {
                    self.tables[idx] = new_ptr;
                }
            }
        }
    }

    pub fn double_directory(&mut self) {
        let old_size = self.tables.len();
        let mut new_tables = Vec::with_capacity(old_size * 2);
        for i in 0..old_size {
            let p = self.tables[i]; // pointer copy (like C++ dd[2i]=dd[2i+1]=d[i])
            new_tables.push(p);
            new_tables.push(p);
        }
        self.tables = new_tables;
        self.global_depth += 1;
    }

    pub fn len(&self) -> usize {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let mut total = 0;
        for &p in &self.tables {
            if seen.insert(p) {
                total += unsafe { &*p }.len();
            }
        }
        total
    }

    /// O(1) count of unique segments (instrumentation; harness polls for split detection).
    pub fn segment_count(&self) -> usize {
        self.unique_tables
    }

    pub fn bucket_count(&self) -> usize {
        self.unique_tables * (64 + 2) // O(1) via the cached segment count
    }

    /// Bucket fill histogram aggregated across unique segments (0..=14 occupied slots).
    pub fn fill_histogram(&self) -> Vec<u64> {
        use std::collections::HashSet;
        let mut out = vec![0u64; 15];
        let mut seen = HashSet::new();
        for &p in &self.tables {
            if seen.insert(p) {
                for (i, c) in unsafe { &*p }.bucket_fill_histogram().into_iter().enumerate() {
                    out[i] += c;
                }
            }
        }
        out
    }

    /// DEBUG: validate every unique segment AND the directory routing structure;
    /// return total live entries across unique segments.
    pub fn debug_validate(&self) -> Result<usize, String> {
        use std::collections::HashSet;
        let gd = self.global_depth;
        for (idx, &p) in self.tables.iter().enumerate() {
            let t = unsafe { &*p };
            let (pat, ld) = t.debug_meta();
            let expected = (idx >> (gd - ld)) as u64;
            if pat != expected {
                return Err(format!("dir[{idx}] -> pattern {pat} != expected {expected} (gd={gd} ld={ld})"));
            }
        }
        let mut seen = HashSet::new();
        let mut total = 0usize;
        for &p in &self.tables {
            if seen.insert(p) {
                let t = unsafe { &*p };
                t.debug_check()?;
                total += t.len();
            }
        }
        Ok(total)
    }

    /// DEBUG: scan all unique segments for `key`; return (dir_idx, bucket, pattern, local_depth).
    pub fn debug_find_any(&self, key: &K) -> Option<(usize, usize, u64, usize)>
    where K: PartialEq {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for (idx, &p) in self.tables.iter().enumerate() {
            if seen.insert(p) {
                let t = unsafe { &*p };
                if let Some(b) = t.debug_find(key) {
                    let (pat, d) = t.debug_meta();
                    return Some((idx, b, pat, d));
                }
            }
        }
        None
    }

    // Private: overwriting a directory slot with a foreign pointer can leak the box it
    // replaced. Only the controlled `allocate_and_set_table` path uses it, and
    // only on the doubling path where the overwritten alias is still referenced elsewhere.
    fn set_table(&mut self, idx: usize, ptr: *mut Table<K, V>) {
        let idx = idx & ((1 << self.global_depth) - 1);
        if idx < self.tables.len() {
            self.tables[idx] = ptr;
        }
    }

    /// Allocate a new segment and install it (directory-doubling path).
    pub fn allocate_and_set_table(&mut self, idx: usize, table: Table<K, V>) {
        let ptr = self.allocator.allocate(table);
        self.set_table(idx, ptr);
        self.unique_tables += 1;
    }

    pub fn table_count(&self) -> usize {
        self.tables.len()
    }
}

impl<K, V> Default for Directory<K, V, DefaultAllocator>
where
    K: Eq + Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

// Free each unique segment exactly once — but ONLY when the allocator does not own its
// memory. A PoolAllocator frees its whole arena in its own Drop; freeing here too would
// double-free. (The allocator field drops after this impl, so the pool frees last.)
impl<K, V, A: TableAllocator<K, V>> Drop for Directory<K, V, A> {
    fn drop(&mut self) {
        if self.allocator.owns_memory() {
            return;
        }
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for &p in &self.tables {
            if seen.insert(p) {
                unsafe { drop(Box::from_raw(p)); }
            }
        }
    }
}
