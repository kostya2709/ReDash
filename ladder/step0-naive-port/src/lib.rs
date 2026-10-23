// Step 0 — naive port: a clean, bug-free, C++-faithful port of DASH.
// Rc<RefCell> directory, SipHash, no pool, scalar fingerprint, no routing bits,
// rehash-on-split, inline uniqueness check (matches C++ DASH `unique_check`).
// This is the unoptimised-but-correct starting point of the ablation ladder.
// Unused helpers are kept so each step's source matches the code that was measured.
#![allow(dead_code)]

mod bucket;
mod directory;
mod table;
mod hash;

pub use hash::{DashExtendibleHashMap, SipHashBuilder};
