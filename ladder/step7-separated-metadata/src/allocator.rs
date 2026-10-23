// Table allocator abstraction (step 3 — the first optimization beyond the C++ core design).
// Returns raw *mut Table. The directory frees pointers on drop ONLY if the allocator
// does not own its memory (owns_memory()==false) — this owner abstraction has to exist
// before the pool can be introduced, so Drop doesn't always assume Box::from_raw.
//
// DefaultAllocator: Box::into_raw passthrough (== step 2 behaviour).
// PoolAllocator: one contiguous arena, cache-line-aligned slots, pages pre-faulted at
//   construction; allocate() does a sequential ptr::write into a pre-faulted slot,
//   eliminating the page faults a fresh Box would incur during a split.

use crate::table::Table;
use std::alloc::{self, Layout};

const CACHE_LINE: usize = 64; // x86_64

pub trait TableAllocator<K, V> {
    /// Allocate + initialize a segment; pointer valid until the allocator drops.
    /// The caller must NOT free it directly.
    fn allocate(&mut self, table: Table<K, V>) -> *mut Table<K, V>;

    /// True if the allocator manages (and frees) its own memory. If false, the
    /// Directory must free segment pointers on drop via Box::from_raw.
    fn owns_memory(&self) -> bool { false }

    /// Reserve storage WITHOUT initialising it, so a split can construct the child directly
    /// in its final home -- the C++ original's allocate-then-fill shape. None means no
    /// reservable slot and the caller falls back to the by-value path.
    fn allocate_uninit(&mut self) -> Option<*mut Table<K, V>> { None }

    /// # of pool-exhaustion fallbacks (0 for DefaultAllocator). Nonzero in a pool run
    /// means the pool was undersized and cold allocations contaminated timing.
    fn exhausted_count(&self) -> usize { 0 }
}

/// Passthrough: each segment is its own Box (identical to step 2).
pub struct DefaultAllocator;

impl<K, V> TableAllocator<K, V> for DefaultAllocator {
    fn allocate(&mut self, table: Table<K, V>) -> *mut Table<K, V> {
        Box::into_raw(Box::new(table))
    }
}

/// Contiguous pre-faulted arena. Hands out cache-line-aligned slots from a free stack.
pub struct PoolAllocator<K, V> {
    free_list: Vec<*mut Table<K, V>>,
    arena: *mut u8,
    arena_layout: Layout,
    overflow: Vec<*mut Table<K, V>>,
    slot_size: usize,
    capacity: usize,
    exhausted: usize,
}

impl<K, V> PoolAllocator<K, V>
where
    K: Eq + Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    pub fn new(capacity: usize) -> Self {
        let table_size = std::mem::size_of::<Table<K, V>>();
        // Align slots to at least the cache line AND the type's own alignment (generic
        // soundness: align_of::<Table>() could exceed 64 for some K/V). Use checked math
        // so a huge capacity can't wrap the size past Layout's limit.
        let align = CACHE_LINE.max(std::mem::align_of::<Table<K, V>>());
        let slot_size = table_size.checked_next_multiple_of(align).expect("slot size overflow");
        let total_size = slot_size.checked_mul(capacity.max(1)).expect("arena size overflow");
        let layout = Layout::from_size_align(total_size, align)
            .expect("invalid layout for pool arena");

        let arena = unsafe { alloc::alloc_zeroed(layout) };
        if arena.is_null() {
            alloc::handle_alloc_error(layout);
        }

        // Pre-fault every slot by writing an empty table into it.
        let mut free_list = Vec::with_capacity(capacity);
        for i in 0..capacity {
            let ptr = unsafe { arena.add(i * slot_size) } as *mut Table<K, V>;
            unsafe { std::ptr::write(ptr, Table::<K, V>::new(0)); }
            free_list.push(ptr);
        }

        Self { free_list, arena, arena_layout: layout, overflow: Vec::new(), slot_size, capacity, exhausted: 0 }
    }
}

impl<K, V> TableAllocator<K, V> for PoolAllocator<K, V> {
    fn owns_memory(&self) -> bool { true }

    fn allocate_uninit(&mut self) -> Option<*mut Table<K, V>> {
        self.free_list.pop()
    }
    fn exhausted_count(&self) -> usize { self.exhausted }

    fn allocate(&mut self, table: Table<K, V>) -> *mut Table<K, V> {
        if let Some(ptr) = self.free_list.pop() {
            unsafe { std::ptr::write(ptr, table); } // sequential write into pre-faulted slot
            ptr
        } else {
            // Say so at the true allocation site. The benchmark used to INFER exhaustion by
            // recomputing the pool formula from the split count, so a mis-calibrated formula
            // could hide the very thing it was checking for. A run that reaches here is timing
            // per-split heap allocation, not the pool, so its numbers are invalid.
            if self.exhausted == 0 {
                eprintln!("!! POOL EXHAUSTED: allocator fell back to heap -- results INVALID");
                if std::env::var_os("DASH_STRICT_POOL").is_some() {
                    eprintln!("!! DASH_STRICT_POOL set -- aborting rather than emitting invalid numbers");
                    std::process::abort();
                }
            }
            self.exhausted += 1;
            let ptr = Box::into_raw(Box::new(table)); // cold fallback
            self.overflow.push(ptr);
            ptr
        }
    }
}

impl<K, V> Drop for PoolAllocator<K, V> {
    fn drop(&mut self) {
        // Drop every arena slot in place (Table has no heap fields, but be correct),
        // free the arena, then free overflow boxes.
        for i in 0..self.capacity {
            let ptr = unsafe { self.arena.add(i * self.slot_size) } as *mut Table<K, V>;
            unsafe { std::ptr::drop_in_place(ptr); }
        }
        unsafe { alloc::dealloc(self.arena, self.arena_layout); }
        for &ptr in &self.overflow {
            unsafe { drop(Box::from_raw(ptr)); }
        }
    }
}
