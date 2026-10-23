// Step 2 (raw-pointer directory) correctness: the shared suite, instantiated with step 2's
// constructor. Identical suite runs on every step — see testsuite/ (redash_testsuite).
use redash_step2::DashExtendibleHashMap;

// Steps 0–2 have no pool, so the capacity hint is ignored.
fn mk(_cap: usize) -> DashExtendibleHashMap<u64, u64> {
    DashExtendibleHashMap::new()
}

redash_testsuite::dash_correctness_suite!(mk);
