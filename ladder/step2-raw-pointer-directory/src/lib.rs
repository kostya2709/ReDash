// Step 2 (C++ parity reached) — step 1 + raw-pointer/UnsafeCell directory.
// Single change vs step 1: the directory now holds raw `*mut Table` (C++ `Table*`) instead
// of Rc<RefCell<Table>> — no refcount, no borrow check, interior mutability via raw
// pointers (single-threaded), Box-allocated, freed on Drop. After this step the
// core design matches C++ DASH (except hasher + allocator).
// Unused helpers are kept so each step's source matches the code that was measured.
#![allow(dead_code)]

mod bucket;
mod directory;
mod table;
mod hash;

pub use hash::{DashExtendibleHashMap, SipHashBuilder};
