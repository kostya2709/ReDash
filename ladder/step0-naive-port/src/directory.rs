// Directory implementation for Dash extendible hashing
// Uses Rc<RefCell<Table>> to match C++ pointer semantics (copy pointers, not tables)

use crate::table::Table;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Directory<K, V> {
    tables: Vec<Rc<RefCell<Table<K, V>>>>,  // Reference-counted pointers (like C++ Table*)
    global_depth: usize,
    // INSTRUMENTATION (all steps, not an optimization): cached count of unique
    // segments, so segment_count()/bucket_count() are O(1) and the harness can poll
    // per-op for exact split detection. Incremented on the two segment-creating paths.
    unique_tables: usize,
}

impl<K, V> Directory<K, V>
where
    K: Eq + Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    pub fn new() -> Self {
        let initial_depth = 1;
        let initial_size = 1 << initial_depth;
        let mut tables = Vec::with_capacity(initial_size);

        for i in 0..initial_size {
            let mut table = Table::new(initial_depth);
            table.pattern = i as u64;  // Set pattern to directory index (matching C++)
            tables.push(Rc::new(RefCell::new(table)));  // Wrap in Rc<RefCell>
        }

        Self {
            tables,
            global_depth: initial_depth,
            unique_tables: initial_size,
        }
    }

    /// O(1) count of unique segments (instrumentation). Each split creates exactly one.
    pub fn segment_count(&self) -> usize {
        self.unique_tables
    }

    pub fn global_depth(&self) -> usize {
        self.global_depth
    }

    pub fn get_table(&self, idx: usize) -> Option<Rc<RefCell<Table<K, V>>>> {
        let idx = idx & ((1 << self.global_depth) - 1);
        self.tables.get(idx).map(|t| Rc::clone(t))  // Clone Rc (just increments ref count)
    }

    pub fn get_table_mut(&mut self, idx: usize) -> Option<Rc<RefCell<Table<K, V>>>> {
        let idx = idx & ((1 << self.global_depth) - 1);
        self.tables.get(idx).map(|t| Rc::clone(t))  // Clone Rc (just increments ref count)
    }

    pub fn update_directory_after_split(&mut self, dir_entry_idx: usize, new_table: Table<K, V>, _old_pattern: u64, _new_pattern: u64, old_local_depth: usize) {
        let new_local_depth = old_local_depth + 1;
        let depth_diff = self.global_depth - new_local_depth;

        let new_table_rc = Rc::new(RefCell::new(new_table));
        self.unique_tables += 1; // exactly one new segment (aliased into >=1 dir entries)

        if depth_diff == 0 {
            if dir_entry_idx % 2 == 0 {
                let update_idx = dir_entry_idx + 1;
                if update_idx < self.tables.len() {
                    self.tables[update_idx] = Rc::clone(&new_table_rc);
                }
            } else {
                if dir_entry_idx < self.tables.len() {
                    self.tables[dir_entry_idx] = Rc::clone(&new_table_rc);
                }
            }
        } else {
            let chunk_size = 1 << (self.global_depth - (new_local_depth - 1));
            let aligned_idx = dir_entry_idx - (dir_entry_idx % chunk_size);
            let base = chunk_size / 2;
            
            for i in 0..base {
                let idx = aligned_idx + base + i;
                if idx < self.tables.len() {
                    self.tables[idx] = Rc::clone(&new_table_rc);
                }
            }
        }
    }

    pub fn double_directory(&mut self) {
        let old_size = self.tables.len();
        let new_size = old_size * 2;
        let mut new_tables = Vec::with_capacity(new_size);
        
        for i in 0..old_size {
            let table_ptr = &self.tables[i];
            new_tables.push(Rc::clone(table_ptr));
            new_tables.push(Rc::clone(table_ptr));
        }
        
        self.tables = new_tables;
        self.global_depth += 1;
    }

    pub fn len(&self) -> usize {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let mut total = 0;
        for table_rc in &self.tables {
            let ptr = Rc::as_ptr(table_rc);
            if seen.insert(ptr) {
                total += table_rc.borrow().len();
            }
        }
        total
    }

    pub fn bucket_count(&self) -> usize {
        self.unique_tables * (64 + 2) // O(1) via the cached segment count
    }

    /// Bucket fill histogram aggregated across unique segments (0..=14 occupied slots).
    pub fn fill_histogram(&self) -> Vec<u64> {
        use std::collections::HashSet;
        let mut out = vec![0u64; 15];
        let mut seen = HashSet::new();
        for table_rc in &self.tables {
            let ptr = Rc::as_ptr(table_rc);
            if seen.insert(ptr) {
                for (i, c) in table_rc.borrow().bucket_fill_histogram().into_iter().enumerate() {
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
        // Routing structure: every directory entry must point to a segment whose pattern
        // matches that index's top (global_depth - local_depth) bits. Catches mis-pointed
        // entries after split/doubling.
        for (idx, table_rc) in self.tables.iter().enumerate() {
            let t = table_rc.borrow();
            let (pat, ld) = t.debug_meta();
            let expected = (idx >> (gd - ld)) as u64;
            if pat != expected {
                return Err(format!("dir[{idx}] -> pattern {pat} != expected {expected} (gd={gd} ld={ld})"));
            }
        }
        let mut seen = HashSet::new();
        let mut total = 0usize;
        for table_rc in &self.tables {
            let ptr = Rc::as_ptr(table_rc);
            if seen.insert(ptr) {
                let t = table_rc.borrow();
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
        for (idx, table_rc) in self.tables.iter().enumerate() {
            let ptr = Rc::as_ptr(table_rc);
            if seen.insert(ptr) {
                let t = table_rc.borrow();
                if let Some(b) = t.debug_find(key) {
                    let (p, d) = t.debug_meta();
                    return Some((idx, b, p, d));
                }
            }
        }
        None
    }

    fn set_table(&mut self, idx: usize, table_rc: Rc<RefCell<Table<K, V>>>) {
        let idx = idx & ((1 << self.global_depth) - 1);
        if idx < self.tables.len() {
            self.tables[idx] = table_rc;
        }
    }

    /// Doubling-split path: wrap the new segment, install it, and count it (one place).
    pub fn allocate_and_set_table(&mut self, idx: usize, table: Table<K, V>) {
        let rc = Rc::new(RefCell::new(table));
        self.set_table(idx, rc);
        self.unique_tables += 1;
    }

    pub fn table_count(&self) -> usize {
        self.tables.len()
    }
}

impl<K, V> Default for Directory<K, V>
where
    K: Eq + Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}
