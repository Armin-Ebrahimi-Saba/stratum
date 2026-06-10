//! ElasticNet regression via cyclic-Jacobi coordinate descent.
//!
//! Minimises: (1/2n)||y − Xw − b||² + α·ρ·||w||₁ + (α/2)·(1−ρ)·||w||²
//!   where α = alpha, ρ = l1_ratio.
//!
//! Algorithm: Jacobi parallel coordinate descent
//! -----------------------------------------------
//! Unlike Gauss-Seidel CD (where each coordinate update immediately modifies
//! the residual seen by subsequent coordinates in the same pass), we compute
//! all coordinate updates simultaneously from the *same* residual snapshot,
//! then apply all residual corrections in one batch.
//!
//!   for each outer iteration:
//!     1. dot_j   = (1/n) · X[:,j]ᵀ · r   for all j  (one parallel fold/reduce)
//!     2. rho_j   = dot_j + w[j] · ||X[:,j]||²/n      (scalar, O(n_cols))
//!     3. w_new_j = soft_threshold(rho_j, α·ρ) / (||X[:,j]||²/n + α·(1−ρ))
//!     4. r[i]   -= Σ_j X[i,j] · (w_new_j − w_j)      (one parallel row pass)
//!
//! Parallelism + SIMD strategy
//! ----------------------------
//! Both Steps 1 and 4 are row-wise sweeps over the C-contiguous X buffer.
//! Each Rayon thread owns a contiguous row-chunk; SIMD kernels (simd::dot,
//! simd::axpy, simd::sq_acc) vectorize the per-row inner loops.
//!
//!   Step 1  par_chunks + fold/reduce with simd::axpy — each thread
//!           accumulates acc[j] += r[i]·X[i,j] using FMA.  ONE Rayon sync.
//!
//!   Step 4  par_chunks + par_iter_mut with simd::dot — each thread
//!           computes r[i] -= dot(X[i,:], Δw) using FMA.  ONE Rayon sync.
//!
//! With 2 Rayon barriers per outer iteration and SIMD inner loops,
//! synchronisation overhead is amortised over O(n_rows·n_cols) FMA work.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use crate::simd;

// ─── Public types ─────────────────────────────────────────────────────────────

pub struct ElasticNetModel {
    pub n_cols:    usize,
    pub coef:      Vec<f32>,
    pub intercept: f32,
    pub n_iter:    usize, // actual passes until convergence
}

// ─── Core algorithm ───────────────────────────────────────────────────────────

#[inline]
fn soft_threshold(x: f32, t: f32) -> f32 {
    if x > t { x - t } else if x < -t { x + t } else { 0.0 }
}

/// Fit an ElasticNet model using Jacobi parallel coordinate descent.
///
/// # Parameters
/// - `x`             — (n_rows × n_cols) design matrix, C-contiguous f32
/// - `y`             — (n_rows,) target vector, f32
/// - `alpha`         — total regularisation strength (≥ 0)
/// - `l1_ratio`      — mixing: 0 = Ridge, 1 = Lasso
/// - `max_iter`      — maximum full passes over all coordinates
/// - `tol`           — stop when max relative |Δw_j| < tol
/// - `fit_intercept` — whether to fit an unpenalised bias term
pub fn elastic_net_fit(
    x:             &Array2<f32>,
    y:             &Array1<f32>,
    alpha:         f32,
    l1_ratio:      f32,
    max_iter:      usize,
    tol:           f32,
    fit_intercept: bool,
) -> ElasticNetModel {
    let (n_rows, n_cols) = x.dim();
    assert_eq!(y.len(), n_rows, "y length must match number of rows in X");
    let n      = n_rows as f32;
    let l1_pen = alpha * l1_ratio;
    let l2_pen = alpha * (1.0 - l1_ratio);

    // All operations work on the flat C-contiguous row buffer directly.
    // Chunks of size n_cols correspond to individual rows of X.
    let x_raw = x.as_slice().expect("X must be C-contiguous");

    // ── Precompute (1/n)·||X[:,j]||² for all j in one row-wise parallel pass ──
    let col_norm_sq: Vec<f32> = x_raw
        .par_chunks(n_cols)
        .fold(
            || vec![0.0f32; n_cols],
            |mut acc, row| { simd::sq_acc(&mut acc, row); acc },
        )
        .reduce(
            || vec![0.0f32; n_cols],
            |mut a, b| { simd::axpy(&mut a, 1.0, &b); a },
        )
        .into_iter()
        .map(|s| s / n)
        .collect();

    // ── Initialise ────────────────────────────────────────────────────────────
    let mut coef = vec![0.0f32; n_cols];
    let mut intercept = if fit_intercept {
        y.iter().sum::<f32>() / n
    } else {
        0.0
    };
    // r = y − X·w − b; with w = 0: r = y − intercept
    let mut r: Vec<f32> = y.iter().map(|&yi| yi - intercept).collect();

    let mut n_iter = max_iter;

    for iter in 0..max_iter {
        // ── Step 1: X^T · r for all columns — one parallel fold/reduce ────────
        // Each thread accumulates partial sums over its row-chunk.
        // Result: dots[j] = sum_i r[i] · x[i,j]
        let dots: Vec<f32> = x_raw
            .par_chunks(n_cols)
            .zip(r.par_iter())
            .fold(
                || vec![0.0f32; n_cols],
                |mut acc, (row, &ri)| { simd::axpy(&mut acc, ri, row); acc },
            )
            .reduce(
                || vec![0.0f32; n_cols],
                |mut a, b| { simd::axpy(&mut a, 1.0, &b); a },
            );

        // ── Step 2: update all coordinates (sequential, O(n_cols)) ───────────
        let mut max_delta   = 0.0f32;
        let mut coef_deltas = vec![0.0f32; n_cols];
        for j in 0..n_cols {
            if col_norm_sq[j] == 0.0 { continue; }
            let rho_j = dots[j] / n + coef[j] * col_norm_sq[j];
            let new_w = soft_threshold(rho_j, l1_pen) / (col_norm_sq[j] + l2_pen);
            let delta = new_w - coef[j];
            coef_deltas[j] = delta;
            coef[j]        = new_w;
            let rel = delta.abs() / new_w.abs().max(1.0);
            if rel > max_delta { max_delta = rel; }
        }

        // ── Step 3: update residuals — one parallel row pass ─────────────────
        // r[i] -= X[i,:] · delta_w   (row dot-product with coef change vector)
        x_raw
            .par_chunks(n_cols)
            .zip(r.par_iter_mut())
            .for_each(|(row, ri)| {
                *ri -= simd::dot(row, &coef_deltas);
            });

        // ── Step 4: intercept update (sequential, cheap) ──────────────────────
        if fit_intercept {
            let b_delta = r.iter().sum::<f32>() / n;
            if b_delta.abs() > f32::EPSILON {
                for ri in r.iter_mut() { *ri -= b_delta; }
                intercept += b_delta;
            }
        }

        // ── Convergence ──────────────────────────────────────────────────────
        if max_delta < tol {
            n_iter = iter + 1;
            break;
        }
    }

    ElasticNetModel { n_cols, coef, intercept, n_iter }
}

/// Predict with a fitted model: ŷ = X·w + b.
pub fn elastic_net_predict(
    x:     &Array2<f32>,
    model: &ElasticNetModel,
) -> Result<Vec<f32>, String> {
    if x.ncols() != model.n_cols {
        return Err(format!(
            "n_cols mismatch: got {} but model expects {}",
            x.ncols(), model.n_cols
        ));
    }
    let coef      = &model.coef;
    let intercept = model.intercept;
    let preds = x.as_slice()
        .ok_or("elastic_net_predict: C-contiguous input required")?
        .par_chunks(model.n_cols)
        .map(|row| simd::dot(row, coef) + intercept)
        .collect();
    Ok(preds)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPS: f32 = 1e-2;

    fn arr2(rows: Vec<Vec<f32>>) -> Array2<f32> {
        let n_rows = rows.len();
        let n_cols = rows[0].len();
        Array2::from_shape_vec(
            (n_rows, n_cols),
            rows.into_iter().flatten().collect(),
        ).unwrap()
    }

    #[test]
    fn fits_linear_signal() {
        let x = arr2(vec![
            vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 4.0],
            vec![4.0, 3.0], vec![5.0, 5.0],
        ]);
        let y: Array1<f32> = array![8.0, 7.0, 18.0, 17.0, 25.0];
        let model = elastic_net_fit(&x, &y, 1e-4, 0.5, 5000, 1e-6, false);
        let preds = elastic_net_predict(&x, &model).unwrap();
        for (p, &t) in preds.iter().zip(y.iter()) {
            assert!((p - t).abs() < 0.1, "pred={p:.3} target={t:.3}");
        }
    }

    #[test]
    fn lasso_produces_sparsity() {
        let x = arr2(vec![
            vec![1.0, 0.0, 5.0], vec![2.0, 0.0, 6.0],
            vec![3.0, 0.0, 7.0], vec![4.0, 0.0, 8.0],
        ]);
        let y: Array1<f32> = array![5.0, 10.0, 15.0, 20.0];
        let model = elastic_net_fit(&x, &y, 1.0, 1.0, 5000, 1e-6, false);
        assert!(model.coef[1].abs() < EPS, "zero column should stay zero");
    }

    #[test]
    fn ridge_does_not_zero_coefficients() {
        let x = arr2(vec![
            vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0], vec![4.0, 5.0],
        ]);
        let y: Array1<f32> = array![3.0, 5.0, 7.0, 9.0];
        let model = elastic_net_fit(&x, &y, 0.1, 0.0, 5000, 1e-6, false);
        assert!(model.coef[0].abs() > 1e-4 || model.coef[1].abs() > 1e-4);
        let preds = elastic_net_predict(&x, &model).unwrap();
        let mse: f32 = preds.iter().zip(y.iter()).map(|(p, &t)| (p - t).powi(2)).sum::<f32>() / 4.0;
        assert!(mse < 1.0, "MSE={mse:.4}");
    }

    #[test]
    fn fit_intercept_shifts_prediction() {
        let x = arr2(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let y: Array1<f32> = array![11.0, 12.0, 13.0]; // y = x + 10
        let model = elastic_net_fit(&x, &y, 1e-4, 0.5, 5000, 1e-6, true);
        assert!((model.intercept - 10.0).abs() < 0.5, "intercept={:.3}", model.intercept);
    }

    #[test]
    fn predict_ncols_mismatch_errors() {
        let x = arr2(vec![vec![1.0, 2.0]]);
        let y: Array1<f32> = array![3.0];
        let model = elastic_net_fit(&x, &y, 0.1, 0.5, 100, 1e-4, false);
        let bad = arr2(vec![vec![1.0]]);
        assert!(elastic_net_predict(&bad, &model).is_err());
    }

    #[test]
    fn zero_input_zero_predictions() {
        let x = Array2::<f32>::zeros((4, 3));
        let y: Array1<f32> = array![1.0, 2.0, 3.0, 4.0];
        let model = elastic_net_fit(&x, &y, 0.1, 0.5, 100, 1e-4, true);
        assert!(model.coef.iter().all(|&c| c.abs() < EPS));
    }
}
