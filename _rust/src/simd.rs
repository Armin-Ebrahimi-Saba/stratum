//! SIMD-accelerated row-level kernels for normalize.rs.
//!
//! Each public function dispatches to the best path available at runtime:
//!
//!   aarch64  NEON is mandatory on AArch64; no runtime check needed.
//!            4-accumulator 16-element unrolled loops saturate the FMA/FADD
//!            pipelines by hiding the 4-cycle instruction latency.
//!
//!   x86_64   AVX2 checked at runtime via is_x86_feature_detected!.
//!            Falls back to scalar (LLVM still emits SSE2) if absent.
//!
//!   other    Scalar path; LLVM auto-vectorizes at opt-level 3.

// ─── Public API ──────────────────────────────────────────────────────────────

/// Sum of squares of `row` (L2 norm squared).
#[inline]
pub fn norm_sq_row(row: &[f32]) -> f32 {
    dispatch_norm_sq(row)
}

/// Sum of absolute values of `row` (L1 norm).
#[inline]
pub fn sum_abs_row(row: &[f32]) -> f32 {
    dispatch_sum_abs(row)
}

/// Maximum absolute value of `row` (L∞ norm).
#[inline]
pub fn max_abs_row(row: &[f32]) -> f32 {
    dispatch_max_abs(row)
}

/// Multiply every element of `row` by `scale` in-place.
#[inline]
pub fn scale_row(row: &mut [f32], scale: f32) {
    dispatch_scale(row, scale);
}

// ─── aarch64 dispatch ────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_norm_sq(row: &[f32]) -> f32 { unsafe { neon::norm_sq(row) } }

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_sum_abs(row: &[f32]) -> f32 { unsafe { neon::sum_abs(row) } }

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_max_abs(row: &[f32]) -> f32 { unsafe { neon::max_abs(row) } }

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_scale(row: &mut [f32], scale: f32) { unsafe { neon::scale(row, scale) } }

// ─── x86_64 dispatch ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_norm_sq(row: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") { unsafe { avx2::norm_sq(row) } }
    else { scalar::norm_sq(row) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_sum_abs(row: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") { unsafe { avx2::sum_abs(row) } }
    else { scalar::sum_abs(row) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_max_abs(row: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") { unsafe { avx2::max_abs(row) } }
    else { scalar::max_abs(row) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_scale(row: &mut [f32], scale: f32) {
    if is_x86_feature_detected!("avx2") { unsafe { avx2::scale(row, scale) } }
    else { scalar::scale(row, scale) }
}

// ─── other architectures ─────────────────────────────────────────────────────

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline] fn dispatch_norm_sq(row: &[f32]) -> f32     { scalar::norm_sq(row) }
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline] fn dispatch_sum_abs(row: &[f32]) -> f32     { scalar::sum_abs(row) }
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline] fn dispatch_max_abs(row: &[f32]) -> f32     { scalar::max_abs(row) }
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline] fn dispatch_scale(row: &mut [f32], s: f32)  { scalar::scale(row, s) }

// ─── Scalar fallback ─────────────────────────────────────────────────────────

#[allow(dead_code)] // used on non-NEON / non-AVX2 paths
mod scalar {
    #[inline] pub fn norm_sq(row: &[f32]) -> f32 { row.iter().map(|&x| x * x).sum() }
    #[inline] pub fn sum_abs(row: &[f32]) -> f32 { row.iter().map(|&x| x.abs()).sum() }
    #[inline] pub fn max_abs(row: &[f32]) -> f32 { row.iter().map(|&x| x.abs()).fold(0.0f32, f32::max) }
    #[inline] pub fn scale(row: &mut [f32], s: f32) { for x in row { *x *= s; } }
}

// ─── NEON (aarch64) ──────────────────────────────────────────────────────────
//
// NEON is mandatory on AArch64; `#[target_feature]` unlocks the intrinsic
// names but no runtime guard is needed.  Each loop processes 16 f32 per
// iteration with 4 independent accumulators so that the 4-cycle FMA latency
// never stalls the pipeline.

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn norm_sq(row: &[f32]) -> f32 {
        let n   = row.len();
        let ptr = row.as_ptr();

        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);

        let n16 = n & !15;
        let mut i = 0usize;
        while i < n16 {
            let v0 = vld1q_f32(ptr.add(i));
            let v1 = vld1q_f32(ptr.add(i + 4));
            let v2 = vld1q_f32(ptr.add(i + 8));
            let v3 = vld1q_f32(ptr.add(i + 12));
            a0 = vfmaq_f32(a0, v0, v0); // a0 += v0 * v0
            a1 = vfmaq_f32(a1, v1, v1);
            a2 = vfmaq_f32(a2, v2, v2);
            a3 = vfmaq_f32(a3, v3, v3);
            i += 16;
        }

        a0 = vaddq_f32(a0, vaddq_f32(a1, vaddq_f32(a2, a3)));

        let n4 = n & !3;
        while i < n4 {
            let v = vld1q_f32(ptr.add(i));
            a0 = vfmaq_f32(a0, v, v);
            i += 4;
        }

        let mut sum = vaddvq_f32(a0); // horizontal add
        while i < n {
            let x = *ptr.add(i);
            sum += x * x;
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn sum_abs(row: &[f32]) -> f32 {
        let n   = row.len();
        let ptr = row.as_ptr();

        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);

        let n16 = n & !15;
        let mut i = 0usize;
        while i < n16 {
            a0 = vaddq_f32(a0, vabsq_f32(vld1q_f32(ptr.add(i))));
            a1 = vaddq_f32(a1, vabsq_f32(vld1q_f32(ptr.add(i + 4))));
            a2 = vaddq_f32(a2, vabsq_f32(vld1q_f32(ptr.add(i + 8))));
            a3 = vaddq_f32(a3, vabsq_f32(vld1q_f32(ptr.add(i + 12))));
            i += 16;
        }

        a0 = vaddq_f32(a0, vaddq_f32(a1, vaddq_f32(a2, a3)));

        let n4 = n & !3;
        while i < n4 {
            a0 = vaddq_f32(a0, vabsq_f32(vld1q_f32(ptr.add(i))));
            i += 4;
        }

        let mut sum = vaddvq_f32(a0);
        while i < n {
            sum += (*ptr.add(i)).abs();
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn max_abs(row: &[f32]) -> f32 {
        let n   = row.len();
        let ptr = row.as_ptr();

        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);

        let n16 = n & !15;
        let mut i = 0usize;
        while i < n16 {
            a0 = vmaxq_f32(a0, vabsq_f32(vld1q_f32(ptr.add(i))));
            a1 = vmaxq_f32(a1, vabsq_f32(vld1q_f32(ptr.add(i + 4))));
            a2 = vmaxq_f32(a2, vabsq_f32(vld1q_f32(ptr.add(i + 8))));
            a3 = vmaxq_f32(a3, vabsq_f32(vld1q_f32(ptr.add(i + 12))));
            i += 16;
        }

        a0 = vmaxq_f32(a0, vmaxq_f32(a1, vmaxq_f32(a2, a3)));

        let n4 = n & !3;
        while i < n4 {
            a0 = vmaxq_f32(a0, vabsq_f32(vld1q_f32(ptr.add(i))));
            i += 4;
        }

        let mut max = vmaxvq_f32(a0); // horizontal max
        while i < n {
            let x = (*ptr.add(i)).abs();
            if x > max { max = x; }
            i += 1;
        }
        max
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn scale(row: &mut [f32], s: f32) {
        let n   = row.len();
        let ptr = row.as_mut_ptr();
        let sv  = vdupq_n_f32(s);

        let n16 = n & !15;
        let mut i = 0usize;
        while i < n16 {
            vst1q_f32(ptr.add(i),      vmulq_f32(vld1q_f32(ptr.add(i)),      sv));
            vst1q_f32(ptr.add(i + 4),  vmulq_f32(vld1q_f32(ptr.add(i + 4)),  sv));
            vst1q_f32(ptr.add(i + 8),  vmulq_f32(vld1q_f32(ptr.add(i + 8)),  sv));
            vst1q_f32(ptr.add(i + 12), vmulq_f32(vld1q_f32(ptr.add(i + 12)), sv));
            i += 16;
        }
        while i < n {
            *ptr.add(i) *= s;
            i += 1;
        }
    }
}

// ─── AVX2 (x86_64) ───────────────────────────────────────────────────────────
//
// Only called after is_x86_feature_detected!("avx2") returns true, so the
// #[target_feature] annotation is safe.  Uses unaligned loads (_loadu_ps) so
// no alignment requirement on the input slice.

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    // Reduce __m256 to a single f32 sum via two 128-bit halves.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        let lo   = _mm256_castps256_ps128(v);    // [0,1,2,3]
        let hi   = _mm256_extractf128_ps(v, 1);  // [4,5,6,7]
        let s128 = _mm_add_ps(lo, hi);            // pairwise sum of halves
        let shuf = _mm_movehdup_ps(s128);
        let s    = _mm_add_ps(s128, shuf);
        let sh2  = _mm_movehl_ps(shuf, s);
        _mm_cvtss_f32(_mm_add_ss(s, sh2))
    }

    // Reduce __m256 to a single f32 max.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hmax256(v: __m256) -> f32 {
        let lo   = _mm256_castps256_ps128(v);
        let hi   = _mm256_extractf128_ps(v, 1);
        let m128 = _mm_max_ps(lo, hi);
        let shuf = _mm_movehdup_ps(m128);
        let m    = _mm_max_ps(m128, shuf);
        let sh2  = _mm_movehl_ps(shuf, m);
        _mm_cvtss_f32(_mm_max_ss(m, sh2))
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn norm_sq(row: &[f32]) -> f32 {
        let n   = row.len();
        let ptr = row.as_ptr();

        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();

        let n32 = n & !31;
        let mut i = 0usize;
        while i < n32 {
            let v0 = _mm256_loadu_ps(ptr.add(i));
            let v1 = _mm256_loadu_ps(ptr.add(i + 8));
            let v2 = _mm256_loadu_ps(ptr.add(i + 16));
            let v3 = _mm256_loadu_ps(ptr.add(i + 24));
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(v0, v0));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(v1, v1));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(v2, v2));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(v3, v3));
            i += 32;
        }

        a0 = _mm256_add_ps(a0, _mm256_add_ps(a1, _mm256_add_ps(a2, a3)));

        let n8 = n & !7;
        while i < n8 {
            let v = _mm256_loadu_ps(ptr.add(i));
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(v, v));
            i += 8;
        }

        let mut sum = hsum256(a0);
        while i < n { let x = *ptr.add(i); sum += x * x; i += 1; }
        sum
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn sum_abs(row: &[f32]) -> f32 {
        let n         = row.len();
        let ptr       = row.as_ptr();
        let sign_mask = _mm256_set1_ps(-0.0f32); // 0x80000000 in every lane

        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();

        let n32 = n & !31;
        let mut i = 0usize;
        while i < n32 {
            a0 = _mm256_add_ps(a0, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(ptr.add(i))));
            a1 = _mm256_add_ps(a1, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(ptr.add(i + 8))));
            a2 = _mm256_add_ps(a2, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(ptr.add(i + 16))));
            a3 = _mm256_add_ps(a3, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(ptr.add(i + 24))));
            i += 32;
        }

        a0 = _mm256_add_ps(a0, _mm256_add_ps(a1, _mm256_add_ps(a2, a3)));

        let n8 = n & !7;
        while i < n8 {
            a0 = _mm256_add_ps(a0, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(ptr.add(i))));
            i += 8;
        }

        let mut sum = hsum256(a0);
        while i < n { sum += (*ptr.add(i)).abs(); i += 1; }
        sum
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn max_abs(row: &[f32]) -> f32 {
        let n         = row.len();
        let ptr       = row.as_ptr();
        let sign_mask = _mm256_set1_ps(-0.0f32);

        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();

        let n32 = n & !31;
        let mut i = 0usize;
        while i < n32 {
            a0 = _mm256_max_ps(a0, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(ptr.add(i))));
            a1 = _mm256_max_ps(a1, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(ptr.add(i + 8))));
            a2 = _mm256_max_ps(a2, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(ptr.add(i + 16))));
            a3 = _mm256_max_ps(a3, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(ptr.add(i + 24))));
            i += 32;
        }

        a0 = _mm256_max_ps(a0, _mm256_max_ps(a1, _mm256_max_ps(a2, a3)));

        let n8 = n & !7;
        while i < n8 {
            a0 = _mm256_max_ps(a0, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(ptr.add(i))));
            i += 8;
        }

        let mut max = hmax256(a0);
        while i < n { let x = (*ptr.add(i)).abs(); if x > max { max = x; } i += 1; }
        max
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn scale(row: &mut [f32], s: f32) {
        let n   = row.len();
        let ptr = row.as_mut_ptr();
        let sv  = _mm256_set1_ps(s);

        let n32 = n & !31;
        let mut i = 0usize;
        while i < n32 {
            _mm256_storeu_ps(ptr.add(i),      _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)),      sv));
            _mm256_storeu_ps(ptr.add(i + 8),  _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)),  sv));
            _mm256_storeu_ps(ptr.add(i + 16), _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 16)), sv));
            _mm256_storeu_ps(ptr.add(i + 24), _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 24)), sv));
            i += 32;
        }
        while i < n { *ptr.add(i) *= s; i += 1; }
    }
}
