// Step 3 correctness: the shared suite via the POOL allocator (exercises the pre-faulted
// arena, segment reuse on split/doubling, and the owns_memory() Drop path).
use redash_step3::{DashExtendibleHashMap, PoolAllocator, SipHashBuilder};

fn mk(cap: usize) -> DashExtendibleHashMap<u64, u64, PoolAllocator<u64, u64>, SipHashBuilder> {
    // ~cap/600 segments needed; oversize a bit. Exhaustion falls back to Box (correct).
    let segments = (cap / 600).max(64) + 128;
    DashExtendibleHashMap::with_pool_allocator(segments)
}

redash_testsuite::dash_correctness_suite!(mk);
