// Step 9 correctness: shared suite under BOTH hashers, validating the threaded-hasher split
// is correct for FoldHash (the default since step 4) and SipHash. Uses the pool allocator.
use redash_step9::{DashExtendibleHashMap, PoolAllocator, DefaultHashBuilder, SipHashBuilder};

mod foldhash_default {
    use super::*;
    fn mk(cap: usize) -> DashExtendibleHashMap<u64, u64, PoolAllocator<u64, u64>, DefaultHashBuilder> {
        let segs = (cap / 600).max(64) + 128;
        DashExtendibleHashMap::with_pool_allocator(segs) // FoldHash
    }
    redash_testsuite::dash_correctness_suite!(mk);
}
mod siphash {
    use super::*;
    fn mk(cap: usize) -> DashExtendibleHashMap<u64, u64, PoolAllocator<u64, u64>, SipHashBuilder> {
        let segs = (cap / 600).max(64) + 128;
        DashExtendibleHashMap::with_allocator_and_hasher(PoolAllocator::new(segs), SipHashBuilder::default())
    }
    redash_testsuite::dash_correctness_suite!(mk);
}
