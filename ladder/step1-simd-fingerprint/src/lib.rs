// Step 1 (C++ parity) — step 0 + SIMD fingerprint comparison.
// The only change vs step 0: the bucket fingerprint scan in get()/remove() uses an SSE2
// 16-way compare (matching C++ DASH's SSE_CMP8) instead of a scalar loop. Everything
// else (Rc<RefCell> directory, SipHash, no pool, rehash-on-split) is unchanged.
// This is a PARITY fix (C++ uses SIMD by default), not a novel optimization.
// Unused helpers are kept so each step's source matches the code that was measured.
#![allow(dead_code)]

mod bucket;
mod directory;
mod table;
mod hash;

pub use hash::{DashExtendibleHashMap, SipHashBuilder};
