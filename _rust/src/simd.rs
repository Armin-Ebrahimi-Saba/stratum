//! SIMD kernels for ElasticNet inner loops.
//!
//! Three operations, each used in the hot path of coordinate descent:
//!   dot      — Σ a[i]·b[i]          (residual update, predict)
//!   axpy     — acc[i] += scale·x[i]  (X^T·r fold accumulation, reduce merge)
//!   sq_acc   — acc[i] += x[i]²       (column norm precomputation)
//!
//! Dispatch: NEON always on AArch64 (mandatory ISA extension).
//!           AVX2+FMA with runtime detection on x86_64, scalar fallback otherwise.
//!           4-register unrolling hides the 4-cycle FMA latency on both architectures.

/// Dot product: Σᵢ a[i] · b[i]
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 { dispatch_dot(a, b) }

/// AXPY accumulation: acc[i] += scale · x[i]
#[inline]
pub fn axpy(acc: &mut [f32], scale: f32, x: &[f32]) { dispatch_axpy(acc, scale, x) }

/// Squared accumulation: acc[i] += x[i]²
#[inline]
pub fn sq_acc(acc: &mut [f32], x: &[f32]) { dispatch_sq_acc(acc, x) }

// ── per-arch dispatch — separate cfg blocks avoid dead-code warnings ──────────

#[cfg(target_arch = "aarch64")]
#[inline] fn dispatch_dot(a: &[f32], b: &[f32]) -> f32 { unsafe { neon::dot(a, b) } }
#[cfg(target_arch = "aarch64")]
#[inline] fn dispatch_axpy(acc: &mut [f32], s: f32, x: &[f32]) { unsafe { neon::axpy(acc, s, x) } }
#[cfg(target_arch = "aarch64")]
#[inline] fn dispatch_sq_acc(acc: &mut [f32], x: &[f32]) { unsafe { neon::sq_acc(acc, x) } }

#[cfg(target_arch = "x86_64")]
#[inline] fn dispatch_dot(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { avx2::dot(a, b) }
    } else { scalar::dot(a, b) }
}
#[cfg(target_arch = "x86_64")]
#[inline] fn dispatch_axpy(acc: &mut [f32], s: f32, x: &[f32]) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { avx2::axpy(acc, s, x) }
    } else { scalar::axpy(acc, s, x) }
}
#[cfg(target_arch = "x86_64")]
#[inline] fn dispatch_sq_acc(acc: &mut [f32], x: &[f32]) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { avx2::sq_acc(acc, x) }
    } else { scalar::sq_acc(acc, x) }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline] fn dispatch_dot(a: &[f32], b: &[f32]) -> f32 { scalar::dot(a, b) }
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline] fn dispatch_axpy(acc: &mut [f32], s: f32, x: &[f32]) { scalar::axpy(acc, s, x) }
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline] fn dispatch_sq_acc(acc: &mut [f32], x: &[f32]) { scalar::sq_acc(acc, x) }

// ── scalar fallback ──────────────────────────────────────────────────────────
#[allow(dead_code)]
mod scalar {
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
    }
    pub fn axpy(acc: &mut [f32], s: f32, x: &[f32]) {
        for (a, &xi) in acc.iter_mut().zip(x.iter()) { *a += s * xi; }
    }
    pub fn sq_acc(acc: &mut [f32], x: &[f32]) {
        for (a, &xi) in acc.iter_mut().zip(x.iter()) { *a += xi * xi; }
    }
}

// ── NEON (AArch64) ───────────────────────────────────────────────────────────
// NEON is mandatory on AArch64 — no runtime detection needed.
// vfmaq_f32(a, b, c) = a + b*c
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Dot product, 4-accumulator unrolling over 16 f32 per outer step.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= n {
            s0 = vfmaq_f32(s0, vld1q_f32(ap.add(i)),    vld1q_f32(bp.add(i)));
            s1 = vfmaq_f32(s1, vld1q_f32(ap.add(i+4)),  vld1q_f32(bp.add(i+4)));
            s2 = vfmaq_f32(s2, vld1q_f32(ap.add(i+8)),  vld1q_f32(bp.add(i+8)));
            s3 = vfmaq_f32(s3, vld1q_f32(ap.add(i+12)), vld1q_f32(bp.add(i+12)));
            i += 16;
        }
        while i + 4 <= n {
            s0 = vfmaq_f32(s0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
            i += 4;
        }
        s0 = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        let mut sum = vaddvq_f32(s0);
        while i < n { sum += *ap.add(i) * *bp.add(i); i += 1; }
        sum
    }

    /// AXPY: acc += scale·x, 4-register unrolled.
    #[target_feature(enable = "neon")]
    pub unsafe fn axpy(acc: &mut [f32], scale: f32, x: &[f32]) {
        let n = acc.len();
        let sv = vdupq_n_f32(scale);
        let ap = acc.as_mut_ptr();
        let xp = x.as_ptr();
        let mut i = 0;
        while i + 16 <= n {
            vst1q_f32(ap.add(i),    vfmaq_f32(vld1q_f32(ap.add(i)),    sv, vld1q_f32(xp.add(i))));
            vst1q_f32(ap.add(i+4),  vfmaq_f32(vld1q_f32(ap.add(i+4)),  sv, vld1q_f32(xp.add(i+4))));
            vst1q_f32(ap.add(i+8),  vfmaq_f32(vld1q_f32(ap.add(i+8)),  sv, vld1q_f32(xp.add(i+8))));
            vst1q_f32(ap.add(i+12), vfmaq_f32(vld1q_f32(ap.add(i+12)), sv, vld1q_f32(xp.add(i+12))));
            i += 16;
        }
        while i + 4 <= n {
            vst1q_f32(ap.add(i), vfmaq_f32(vld1q_f32(ap.add(i)), sv, vld1q_f32(xp.add(i))));
            i += 4;
        }
        while i < n { *ap.add(i) += scale * *xp.add(i); i += 1; }
    }

    /// Squared accumulation: acc += x², 4-register unrolled.
    #[target_feature(enable = "neon")]
    pub unsafe fn sq_acc(acc: &mut [f32], x: &[f32]) {
        let n = acc.len();
        let ap = acc.as_mut_ptr();
        let xp = x.as_ptr();
        let mut i = 0;
        while i + 16 <= n {
            let x0 = vld1q_f32(xp.add(i));
            let x1 = vld1q_f32(xp.add(i+4));
            let x2 = vld1q_f32(xp.add(i+8));
            let x3 = vld1q_f32(xp.add(i+12));
            vst1q_f32(ap.add(i),    vfmaq_f32(vld1q_f32(ap.add(i)),    x0, x0));
            vst1q_f32(ap.add(i+4),  vfmaq_f32(vld1q_f32(ap.add(i+4)),  x1, x1));
            vst1q_f32(ap.add(i+8),  vfmaq_f32(vld1q_f32(ap.add(i+8)),  x2, x2));
            vst1q_f32(ap.add(i+12), vfmaq_f32(vld1q_f32(ap.add(i+12)), x3, x3));
            i += 16;
        }
        while i + 4 <= n {
            let xv = vld1q_f32(xp.add(i));
            vst1q_f32(ap.add(i), vfmaq_f32(vld1q_f32(ap.add(i)), xv, xv));
            i += 4;
        }
        while i < n { *ap.add(i) += *xp.add(i) * *xp.add(i); i += 1; }
    }
}

// ── AVX2 + FMA (x86_64) ──────────────────────────────────────────────────────
// _mm256_fmadd_ps(a, b, c) = a*b + c
// Horizontal sum via cast+extract to 128-bit, then two hadd-style steps.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Horizontal sum of an 8-wide f32 vector.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum256(v: __m256) -> f32 {
        let lo   = _mm256_castps256_ps128(v);
        let hi   = _mm256_extractf128_ps(v, 1);
        let s4   = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s4);
        let s2   = _mm_add_ps(s4, shuf);
        let s1   = _mm_add_ss(s2, _mm_movehl_ps(shuf, s2));
        _mm_cvtss_f32(s1)
    }

    /// Dot product, 4-accumulator unrolling over 32 f32 per outer step.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)),    _mm256_loadu_ps(bp.add(i)),    s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i+8)),  _mm256_loadu_ps(bp.add(i+8)),  s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i+16)), _mm256_loadu_ps(bp.add(i+16)), s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i+24)), _mm256_loadu_ps(bp.add(i+24)), s3);
            i += 32;
        }
        while i + 8 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), s0);
            i += 8;
        }
        let mut sum = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
        while i < n { sum += *ap.add(i) * *bp.add(i); i += 1; }
        sum
    }

    /// AXPY: acc += scale·x, 4-register unrolled.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy(acc: &mut [f32], scale: f32, x: &[f32]) {
        let n = acc.len();
        let sv = _mm256_set1_ps(scale);
        let ap = acc.as_mut_ptr();
        let xp = x.as_ptr();
        let mut i = 0;
        while i + 32 <= n {
            _mm256_storeu_ps(ap.add(i),    _mm256_fmadd_ps(sv, _mm256_loadu_ps(xp.add(i)),    _mm256_loadu_ps(ap.add(i))));
            _mm256_storeu_ps(ap.add(i+8),  _mm256_fmadd_ps(sv, _mm256_loadu_ps(xp.add(i+8)),  _mm256_loadu_ps(ap.add(i+8))));
            _mm256_storeu_ps(ap.add(i+16), _mm256_fmadd_ps(sv, _mm256_loadu_ps(xp.add(i+16)), _mm256_loadu_ps(ap.add(i+16))));
            _mm256_storeu_ps(ap.add(i+24), _mm256_fmadd_ps(sv, _mm256_loadu_ps(xp.add(i+24)), _mm256_loadu_ps(ap.add(i+24))));
            i += 32;
        }
        while i + 8 <= n {
            _mm256_storeu_ps(ap.add(i), _mm256_fmadd_ps(sv, _mm256_loadu_ps(xp.add(i)), _mm256_loadu_ps(ap.add(i))));
            i += 8;
        }
        while i < n { *ap.add(i) += scale * *xp.add(i); i += 1; }
    }

    /// Squared accumulation: acc += x², 4-register unrolled.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn sq_acc(acc: &mut [f32], x: &[f32]) {
        let n = acc.len();
        let ap = acc.as_mut_ptr();
        let xp = x.as_ptr();
        let mut i = 0;
        while i + 32 <= n {
            let x0 = _mm256_loadu_ps(xp.add(i));
            let x1 = _mm256_loadu_ps(xp.add(i+8));
            let x2 = _mm256_loadu_ps(xp.add(i+16));
            let x3 = _mm256_loadu_ps(xp.add(i+24));
            _mm256_storeu_ps(ap.add(i),    _mm256_fmadd_ps(x0, x0, _mm256_loadu_ps(ap.add(i))));
            _mm256_storeu_ps(ap.add(i+8),  _mm256_fmadd_ps(x1, x1, _mm256_loadu_ps(ap.add(i+8))));
            _mm256_storeu_ps(ap.add(i+16), _mm256_fmadd_ps(x2, x2, _mm256_loadu_ps(ap.add(i+16))));
            _mm256_storeu_ps(ap.add(i+24), _mm256_fmadd_ps(x3, x3, _mm256_loadu_ps(ap.add(i+24))));
            i += 32;
        }
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(xp.add(i));
            _mm256_storeu_ps(ap.add(i), _mm256_fmadd_ps(xv, xv, _mm256_loadu_ps(ap.add(i))));
            i += 8;
        }
        while i < n { *ap.add(i) += *xp.add(i) * *xp.add(i); i += 1; }
    }
}
