// Step 3 (first optimization beyond the C++ core design) — step 2 + pool allocator.
// Single change vs step 2: segment memory comes from a pre-faulted PoolAllocator (contiguous
// arena, cache-line-aligned slots) instead of per-segment Box::into_raw, eliminating
// page faults during splits. Introduces the TableAllocator owner abstraction so the
// directory's Drop frees pointers only when the allocator does NOT own its memory.
// Unused helpers are kept so each step's source matches the code that was measured.
#![allow(dead_code)]

mod allocator;
mod bucket;
mod directory;
mod table;
mod hash;

pub use hash::{DashExtendibleHashMap, SipHashBuilder};
pub use allocator::{DefaultAllocator, PoolAllocator, TableAllocator};
