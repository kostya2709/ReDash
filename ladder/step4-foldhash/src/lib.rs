// Step 4 — hasher selection: default hasher SipHash -> FoldHash. In the paper this step is
// "the baseline" that the headline results are stated against.
// Single change vs step 3: the configured BuildHasher is now threaded through split()
// (steps 0–3 hardcoded DefaultHasher, only accidentally correct for SipHash), and the
// default becomes FoldHash (the comparison winner). SipHash and other hashers are
// swappable via with_hasher / with_allocator_and_hasher for the comparison side-table.
// Unused helpers are kept so each step's source matches the code that was measured.
#![allow(dead_code)]

mod allocator;
mod bucket;
mod directory;
mod table;
mod hash;

pub use hash::{DashExtendibleHashMap, DefaultHashBuilder, SipHashBuilder};
pub use allocator::{DefaultAllocator, PoolAllocator, TableAllocator};
