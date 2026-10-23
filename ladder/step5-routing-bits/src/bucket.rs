use std::marker::PhantomData;

pub(crate) const K_NUM_PAIR_PER_BUCKET: usize = 14;
const COUNT_MASK: u32 = 0xF;

// Step 1 (C++ parity): SIMD fingerprint comparison, matching C++ DASH's SSE_CMP8.
// Compares all 16 fingerprint bytes against `meta_hash` in one instruction and returns
// a 16-bit mask (bit i set if finger_array[i] == meta_hash). x86_64 uses SSE2; other
// targets fall back to a scalar loop so the crate still compiles everywhere.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[inline(always)]
fn sse_cmp8(finger_array: &[u8; 18], meta_hash: u8) -> u32 {
    unsafe {
        use std::arch::x86_64::*;
        let key_data = _mm_set1_epi8(meta_hash as i8);
        let seg_data = _mm_loadu_si128(finger_array.as_ptr() as *const __m128i);
        let rv_mask = _mm_cmpeq_epi8(seg_data, key_data);
        _mm_movemask_epi8(rv_mask) as u32
    }
}

// Scalar fallback for non-x86 AND for Miri (which can't execute SIMD intrinsics).
#[cfg(any(not(target_arch = "x86_64"), miri))]
#[inline(always)]
fn sse_cmp8(finger_array: &[u8; 18], meta_hash: u8) -> u32 {
    let mut mask = 0u32;
    for i in 0..16 {
        if finger_array[i] == meta_hash { mask |= 1 << i; }
    }
    mask
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Pair<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

#[repr(C)]
pub struct Bucket<K, V> {
    version_lock: u32,
    bitmap: u32,
    pub(crate) finger_array: [u8; 18],
    overflow_bitmap: u8,
    overflow_index: u8,
    overflow_member: u8,
    overflow_count: u8,
    unused: [u8; 2],
    pub(crate) pairs: [Pair<K, V>; K_NUM_PAIR_PER_BUCKET],
    _phantom: PhantomData<(K, V)>,
}

impl<K, V> Bucket<K, V>
where
    K: Eq + Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    pub fn new() -> Self {
        Self {
            version_lock: 0,
            bitmap: 0,
            finger_array: [0; 18],
            overflow_bitmap: 0,
            overflow_index: 0,
            overflow_member: 0,
            overflow_count: 0,
            unused: [0; 2],
            pairs: [Pair { key: K::default(), value: V::default() }; K_NUM_PAIR_PER_BUCKET],
            _phantom: PhantomData,
        }
    }

    pub fn count(&self) -> usize {
        (self.bitmap & COUNT_MASK) as usize
    }

    /// DEBUG: structural invariants — count nibble matches the allocation bitmap,
    /// no member bit without an allocation bit, count in range.
    pub(crate) fn debug_check(&self) -> Result<(), String> {
        let alloc = self.get_bitmap();          // 14-bit allocation bitmap
        let member = self.get_member();         // 14-bit member/probe bitmap
        let count = self.count();
        if count != alloc.count_ones() as usize {
            return Err(format!("count {} != alloc popcount {}", count, alloc.count_ones()));
        }
        if member & !alloc != 0 {
            return Err("member bit set without allocation bit".into());
        }
        if count > K_NUM_PAIR_PER_BUCKET {
            return Err(format!("count {} > {}", count, K_NUM_PAIR_PER_BUCKET));
        }
        Ok(())
    }

    /// DEBUG: is `key` physically present in any allocated slot (ignoring meta/probe filters)?
    pub(crate) fn debug_has(&self, key: &K) -> bool
    where K: PartialEq {
        let bm = self.get_bitmap();
        for i in 0..K_NUM_PAIR_PER_BUCKET {
            if (bm & (1 << i)) != 0 && self.pairs[i].key == *key { return true; }
        }
        false
    }

    pub(crate) fn get_bitmap(&self) -> u32 {
        (self.bitmap >> 18) & 0x3FFF
    }

    fn get_member(&self) -> u32 {
        (self.bitmap >> 4) & 0x3FFF
    }

    fn get_inverse_member(&self) -> u32 {
        !self.get_member() & 0x3FFF
    }

    /// Matches C++ __builtin_ctz(~bitmap)
    fn find_empty_slot(&self) -> Option<usize> {
        if self.count() >= K_NUM_PAIR_PER_BUCKET {
            return None;
        }
        let mask = (!self.get_bitmap()) & 0x3FFF;
        if mask == 0 {
            return None;
        }
        Some(mask.trailing_zeros() as usize)
    }

    pub fn insert(&mut self, key: K, value: V, meta_hash: u8, probe: bool) -> Result<bool, ()> {
        if let Some(slot) = self.find_empty_slot() {
            self.pairs[slot] = Pair { key, value };
            self.finger_array[slot] = meta_hash;
            let mut new_bitmap = self.bitmap | (1 << (slot + 18));
            if probe {
                new_bitmap |= 1 << (slot + 4);
            }
            new_bitmap += 1;
            self.bitmap = new_bitmap;
            Ok(true)
        } else {
            Err(())
        }
    }

    /// Matching C++ Insert4split: no dup check, places at count position directly.
    pub fn insert4split(&mut self, key: K, value: V, meta_hash: u8, probe: bool) {
        let slot = self.count();
        self.pairs[slot] = Pair { key, value };
        self.finger_array[slot] = meta_hash;
        let mut new_bitmap = self.bitmap | (1 << (slot + 18));
        if probe {
            new_bitmap |= 1 << (slot + 4);
        }
        new_bitmap += 1;
        self.bitmap = new_bitmap;
    }

    /// Bulk-clear slots using bitmask, matching C++ post-split cleanup.
    pub fn bulk_clear(&mut self, invalid_mask: u32) {
        if invalid_mask == 0 {
            return;
        }
        let count = invalid_mask.count_ones();
        self.bitmap = self.bitmap & (!(invalid_mask << 18)) & (!(invalid_mask << 4));
        self.bitmap -= count;
    }

    pub fn get(&self, key: &K, meta_hash: u8, probe: bool) -> Option<&V> {
        let bitmap = self.get_bitmap();
        let member = self.get_member();
        let mask = if probe { bitmap & member } else { bitmap & !member };
        // SIMD: one compare yields all fingerprint matches; AND with the valid+probe mask,
        // then full key comparison only on matching slots.
        let mut candidates = sse_cmp8(&self.finger_array, meta_hash) & mask & 0x3FFF;
        while candidates != 0 {
            let i = candidates.trailing_zeros() as usize;
            if self.pairs[i].key == *key {
                return Some(&self.pairs[i].value);
            }
            candidates &= candidates - 1;
        }
        None
    }

    pub fn remove(&mut self, key: &K, meta_hash: u8, probe: bool) -> bool {
        let bitmap = self.get_bitmap();
        let member = self.get_member();
        let mask = if probe { bitmap & member } else { bitmap & !member };
        let mut candidates = sse_cmp8(&self.finger_array, meta_hash) & mask & 0x3FFF;
        while candidates != 0 {
            let i = candidates.trailing_zeros() as usize;
            if self.pairs[i].key == *key {
                let mut new_bitmap = self.bitmap & (!(1 << (i + 18))) & (!(1 << (i + 4)));
                new_bitmap -= 1;
                self.bitmap = new_bitmap;
                return true;
            }
            candidates &= candidates - 1;
        }
        false
    }

    pub fn is_full(&self) -> bool {
        self.count() >= K_NUM_PAIR_PER_BUCKET
    }

    pub fn find_org_displacement(&self) -> Option<usize> {
        let mask = self.get_inverse_member();
        if mask == 0 { return None; }
        Some(mask.trailing_zeros() as usize)
    }

    pub fn find_probe_displacement(&self) -> Option<usize> {
        let mask = self.get_member();
        if mask == 0 { return None; }
        Some(mask.trailing_zeros() as usize)
    }

    pub fn get_pair(&self, idx: usize) -> Option<(K, V, u8)> {
        if (self.get_bitmap() & (1 << idx)) != 0 {
            Some((self.pairs[idx].key, self.pairs[idx].value, self.finger_array[idx]))
        } else {
            None
        }
    }

    pub fn insert_displace(&mut self, key: K, value: V, meta_hash: u8, slot: usize, probe: bool) {
        self.pairs[slot] = Pair { key, value };
        self.finger_array[slot] = meta_hash;
        self.bitmap |= 1 << (slot + 18);
        if probe {
            self.bitmap |= 1 << (slot + 4);
        } else {
            self.bitmap &= !(1 << (slot + 4));
        }
        // BUGFIX (validity): C++ set_hash increments count. The original Rust port
        // omitted this, so a displaced bucket reported 14 allocated but count()==13;
        // a full bucket then looked non-full, a later insert silently failed on the
        // full allocation bitmap, and the source slot was overwritten -> key lost.
        self.bitmap += 1;
    }
}

impl<K, V> Clone for Bucket<K, V>
where
    K: Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    fn clone(&self) -> Self {
        Self {
            version_lock: self.version_lock,
            bitmap: self.bitmap,
            finger_array: self.finger_array,
            overflow_bitmap: self.overflow_bitmap,
            overflow_index: self.overflow_index,
            overflow_member: self.overflow_member,
            overflow_count: self.overflow_count,
            unused: self.unused,
            pairs: self.pairs,
            _phantom: PhantomData,
        }
    }
}

impl<K, V> Default for Bucket<K, V>
where
    K: Eq + Clone + Copy + Default,
    V: Clone + Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_insert_get() {
        let mut bucket = Bucket::new();
        assert!(bucket.insert(1u64, 10u64, 42, false).is_ok());
        assert_eq!(bucket.get(&1u64, 42, false), Some(&10u64));
    }

    #[test]
    fn test_bucket_remove() {
        let mut bucket = Bucket::new();
        bucket.insert(1u64, 10u64, 42, false).unwrap();
        assert!(bucket.remove(&1u64, 42, false));
        assert_eq!(bucket.get(&1u64, 42, false), None);
    }
}
