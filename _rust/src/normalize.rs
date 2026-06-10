use ndarray::{Array2, Axis};
use rayon::prelude::*;

use crate::simd;

// ---- Stateless row-wise normalizations ----
//
// All three functions bypass ndarray's axis iterator and operate on the raw
// C-contiguous slice directly.  par_chunks_mut partitions the flat buffer into
// per-row slices, and simd::* dispatch to NEON (aarch64) or AVX2 (x86_64).

pub fn normalize_l2(data: &mut Array2<f32>) {
    let n_cols = data.ncols();
    data.as_slice_mut()
        .expect("normalize_l2: C-contiguous input required")
        .par_chunks_mut(n_cols)
        .for_each(|row| {
            let norm_sq = simd::norm_sq_row(row);
            if norm_sq > 0.0 {
                simd::scale_row(row, 1.0 / norm_sq.sqrt());
            }
        });
}

pub fn normalize_l1(data: &mut Array2<f32>) {
    let n_cols = data.ncols();
    data.as_slice_mut()
        .expect("normalize_l1: C-contiguous input required")
        .par_chunks_mut(n_cols)
        .for_each(|row| {
            let norm = simd::sum_abs_row(row);
            if norm > 0.0 {
                simd::scale_row(row, 1.0 / norm);
            }
        });
}

pub fn normalize_max(data: &mut Array2<f32>) {
    let n_cols = data.ncols();
    data.as_slice_mut()
        .expect("normalize_max: C-contiguous input required")
        .par_chunks_mut(n_cols)
        .for_each(|row| {
            let max_abs = simd::max_abs_row(row);
            if max_abs > 0.0 {
                simd::scale_row(row, 1.0 / max_abs);
            }
        });
}

// ---- MinMaxScaler (per-column, fits to [0, 1]) ----

pub struct MinMaxModel {
    pub n_cols: usize,
    pub min: Vec<f32>,
    pub scale: Vec<f32>, // 1 / (max - min); 0 for constant columns
}

// ---- MinMax stat helpers (two strategies for fitting column stats) ----

// Single row-wise pass using par_chunks on the raw C-contiguous slice.
// Each rayon thread accumulates its own (min, max) vecs; a final reduce merges them.
// No extra allocation proportional to input size — accumulators are O(n_cols) each.
fn min_max_stats_rowwise(data: &Array2<f32>) -> (Vec<f32>, Vec<f32>) {
    let n_cols = data.ncols();
    let raw = data.as_slice().expect("C-contiguous");
    raw.par_chunks(n_cols)
        .fold(
            || (vec![f32::INFINITY; n_cols], vec![f32::NEG_INFINITY; n_cols]),
            |(mut mn, mut mx), row| {
                for (j, &v) in row.iter().enumerate() {
                    if v < mn[j] { mn[j] = v; }
                    if v > mx[j] { mx[j] = v; }
                }
                (mn, mx)
            },
        )
        .reduce(
            || (vec![f32::INFINITY; n_cols], vec![f32::NEG_INFINITY; n_cols]),
            |(mut mn1, mut mx1), (mn2, mx2)| {
                for j in 0..n_cols {
                    mn1[j] = mn1[j].min(mn2[j]);
                    mx1[j] = mx1[j].max(mx2[j]);
                }
                (mn1, mx1)
            },
        )
}

fn min_max_apply(data: &Array2<f32>, min: Vec<f32>, scale: Vec<f32>) -> (MinMaxModel, Array2<f32>) {
    let n_rows = data.nrows();
    let n_cols = data.ncols();
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(data.axis_iter(Axis(0)))
        .for_each(|(mut out_row, in_row)| {
            for j in 0..n_cols {
                out_row[j] = (in_row[j] - min[j]) * scale[j];
            }
        });
    (MinMaxModel { n_cols, min, scale }, out)
}

fn stats_to_scale(min: &[f32], max: &[f32]) -> Vec<f32> {
    min.iter().zip(max.iter()).map(|(&mn, &mx)| {
        let range = mx - mn;
        if range > 0.0 { 1.0 / range } else { 0.0 }
    }).collect()
}

pub fn min_max_fit(data: &Array2<f32>) -> (MinMaxModel, Array2<f32>) {
    let (min, max) = min_max_stats_rowwise(data);
    let scale = stats_to_scale(&min, &max);
    min_max_apply(data, min, scale)
}


pub fn min_max_transform(data: &Array2<f32>, model: &MinMaxModel) -> Result<Array2<f32>, String> {
    if data.ncols() != model.n_cols {
        return Err(format!(
            "n_cols mismatch: input has {} columns but model expects {}",
            data.ncols(),
            model.n_cols
        ));
    }
    let n_rows = data.nrows();
    let n_cols = model.n_cols;
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(data.axis_iter(Axis(0)))
        .for_each(|(mut out_row, in_row)| {
            for j in 0..n_cols {
                out_row[j] = (in_row[j] - model.min[j]) * model.scale[j];
            }
        });
    Ok(out)
}

// ---- StandardScaler (per-column z-score: subtract mean, divide by std) ----

pub struct StandardScalerModel {
    pub n_cols: usize,
    pub mean: Vec<f32>,
    pub inv_std: Vec<f32>, // 1 / std; 0 for zero-variance columns
}

// ---- StandardScaler stat helpers ----

// Single-pass parallel Welford (Chan's algorithm).
//
// Each fold accumulator holds (count, mean[], M2[]) where M2[j] is the sum of
// squared deviations from the running mean for column j.  The reduce step
// merges two accumulators using Chan's numerically-stable combination formula:
//
//   delta      = mean_b - mean_a
//   mean_ab    = mean_a + delta * n_b / (n_a + n_b)
//   M2_ab      = M2_a + M2_b + delta² * n_a * n_b / (n_a + n_b)
//
// At the end: population variance = M2 / n, inv_std = 1 / sqrt(variance).
fn std_scaler_stats_rowwise(data: &Array2<f32>) -> (Vec<f32>, Vec<f32>) {
    let n_cols = data.ncols();
    let raw = data.as_slice().expect("C-contiguous");

    type State = (usize, Vec<f32>, Vec<f32>); // (count, mean, M2)

    let (n_total, mean, m2): State = raw
        .par_chunks(n_cols)
        .fold(
            || (0usize, vec![0.0f32; n_cols], vec![0.0f32; n_cols]),
            |(mut count, mut mean, mut m2), row| {
                count += 1;
                for j in 0..n_cols {
                    let delta  = row[j] - mean[j];
                    mean[j]   += delta / count as f32;
                    m2[j]     += delta * (row[j] - mean[j]); // delta * delta2
                }
                (count, mean, m2)
            },
        )
        .reduce(
            || (0usize, vec![0.0f32; n_cols], vec![0.0f32; n_cols]),
            |(n1, mean1, m2_1), (n2, mean2, m2_2)| {
                let n = n1 + n2;
                if n == 0 {
                    return (0, vec![0.0f32; n_cols], vec![0.0f32; n_cols]);
                }
                let (n1f, n2f, nf) = (n1 as f32, n2 as f32, n as f32);
                let mut out_mean = vec![0.0f32; n_cols];
                let mut out_m2   = vec![0.0f32; n_cols];
                for j in 0..n_cols {
                    let delta    = mean2[j] - mean1[j];
                    out_mean[j]  = mean1[j] + delta * (n2f / nf);
                    out_m2[j]    = m2_1[j] + m2_2[j] + delta * delta * (n1f * n2f / nf);
                }
                (n, out_mean, out_m2)
            },
        );

    let inv_std: Vec<f32> = m2.iter().map(|&s| {
        let var = s / n_total as f32;
        if var > 0.0 { 1.0 / var.sqrt() } else { 0.0 }
    }).collect();

    (mean, inv_std)
}

fn std_scaler_apply(
    data: &Array2<f32>,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
) -> (StandardScalerModel, Array2<f32>) {
    let n_rows = data.nrows();
    let n_cols = data.ncols();
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(data.axis_iter(Axis(0)))
        .for_each(|(mut out_row, in_row)| {
            for j in 0..n_cols {
                out_row[j] = (in_row[j] - mean[j]) * inv_std[j];
            }
        });
    (StandardScalerModel { n_cols, mean, inv_std }, out)
}

pub fn standard_scaler_fit(data: &Array2<f32>) -> (StandardScalerModel, Array2<f32>) {
    let (mean, inv_std) = std_scaler_stats_rowwise(data);
    std_scaler_apply(data, mean, inv_std)
}


pub fn standard_scaler_transform(
    data: &Array2<f32>,
    model: &StandardScalerModel,
) -> Result<Array2<f32>, String> {
    if data.ncols() != model.n_cols {
        return Err(format!(
            "n_cols mismatch: input has {} columns but model expects {}",
            data.ncols(),
            model.n_cols
        ));
    }
    let n_rows = data.nrows();
    let n_cols = model.n_cols;
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(data.axis_iter(Axis(0)))
        .for_each(|(mut out_row, in_row)| {
            for j in 0..n_cols {
                out_row[j] = (in_row[j] - model.mean[j]) * model.inv_std[j];
            }
        });
    Ok(out)
}

// ---- Unit tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPS: f32 = 1e-5;

    #[test]
    fn l2_unit_norm() {
        let mut data = array![[3.0f32, 4.0], [1.0, 0.0], [0.0, 0.0]];
        normalize_l2(&mut data);
        // Row 0: norm = 5, so [0.6, 0.8]
        assert!((data[[0, 0]] - 0.6).abs() < EPS);
        assert!((data[[0, 1]] - 0.8).abs() < EPS);
        // Row 1: norm = 1, unchanged
        assert!((data[[1, 0]] - 1.0).abs() < EPS);
        assert!((data[[1, 1]] - 0.0).abs() < EPS);
        // Row 2: all zeros, stays zero
        assert!((data[[2, 0]] - 0.0).abs() < EPS);
    }

    #[test]
    fn l1_unit_norm() {
        let mut data = array![[3.0f32, 1.0], [-2.0, 2.0], [0.0, 0.0]];
        normalize_l1(&mut data);
        // Row 0: L1 = 4, so [0.75, 0.25]
        assert!((data[[0, 0]] - 0.75).abs() < EPS);
        assert!((data[[0, 1]] - 0.25).abs() < EPS);
        // Row 1: L1 = 4, so [-0.5, 0.5]
        assert!((data[[1, 0]] + 0.5).abs() < EPS);
        assert!((data[[1, 1]] - 0.5).abs() < EPS);
        // All-zero row unchanged
        assert!((data[[2, 0]]).abs() < EPS);
    }

    #[test]
    fn max_unit_norm() {
        let mut data = array![[2.0f32, -6.0, 3.0], [0.0, 0.0, 0.0]];
        normalize_max(&mut data);
        // Row 0: max_abs = 6, so [1/3, -1.0, 0.5]
        assert!((data[[0, 0]] - (2.0 / 6.0)).abs() < EPS);
        assert!((data[[0, 1]] + 1.0).abs() < EPS);
        assert!((data[[0, 2]] - 0.5).abs() < EPS);
        // All-zero row unchanged
        assert!(data[[1, 0]].abs() < EPS);
    }

    #[test]
    fn min_max_fit_basic() {
        let data = array![[0.0f32, 10.0], [5.0, 20.0], [10.0, 30.0]];
        let (model, out) = min_max_fit(&data);
        assert_eq!(model.n_cols, 2);
        // Col 0: min=0, max=10, scale=0.1
        assert!((model.min[0] - 0.0).abs() < EPS);
        assert!((model.scale[0] - 0.1).abs() < EPS);
        // Scaled values for col 0: [0.0, 0.5, 1.0]
        assert!((out[[0, 0]] - 0.0).abs() < EPS);
        assert!((out[[1, 0]] - 0.5).abs() < EPS);
        assert!((out[[2, 0]] - 1.0).abs() < EPS);
        // All output values in [0, 1]
        assert!(out.iter().all(|&v| v >= -EPS && v <= 1.0 + EPS));
    }

    #[test]
    fn min_max_constant_column() {
        let data = array![[5.0f32, 1.0], [5.0, 2.0], [5.0, 3.0]];
        let (model, out) = min_max_fit(&data);
        // Constant column 0: scale = 0, output = 0
        assert!((model.scale[0] - 0.0).abs() < EPS);
        assert!(out.column(0).iter().all(|&v| v.abs() < EPS));
    }

    #[test]
    fn min_max_transform_matches_fit() {
        let train = array![[0.0f32, 0.0], [10.0, 100.0]];
        let (model, fit_out) = min_max_fit(&train);
        let transform_out = min_max_transform(&train, &model).unwrap();
        for (a, b) in fit_out.iter().zip(transform_out.iter()) {
            assert!((a - b).abs() < EPS);
        }
    }

    #[test]
    fn min_max_transform_ncols_mismatch() {
        let train = array![[0.0f32, 0.0], [1.0, 1.0]];
        let (model, _) = min_max_fit(&train);
        let bad = array![[0.0f32, 0.0, 0.0]];
        assert!(min_max_transform(&bad, &model).is_err());
    }

    #[test]
    fn standard_scaler_zero_mean_unit_std() {
        let data = array![[1.0f32, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let (model, out) = standard_scaler_fit(&data);
        // Col 0: mean=2, std=sqrt(2/3)
        assert!((model.mean[0] - 2.0).abs() < EPS);
        // Output col 0 should have zero mean
        let col0_mean: f32 = out.column(0).iter().sum::<f32>() / 3.0;
        assert!(col0_mean.abs() < EPS);
        // Output col 0 should have unit population std
        let col0_var: f32 = out.column(0).iter().map(|&x| x * x).sum::<f32>() / 3.0;
        assert!((col0_var - 1.0).abs() < EPS);
    }

    #[test]
    fn standard_scaler_constant_column() {
        let data = array![[7.0f32, 1.0], [7.0, 2.0], [7.0, 3.0]];
        let (model, out) = standard_scaler_fit(&data);
        // Constant column: inv_std = 0, output = 0
        assert!((model.inv_std[0] - 0.0).abs() < EPS);
        assert!(out.column(0).iter().all(|&v| v.abs() < EPS));
    }

    #[test]
    fn standard_scaler_transform_matches_fit() {
        let train = array![[1.0f32, 10.0], [3.0, 30.0]];
        let (model, fit_out) = standard_scaler_fit(&train);
        let transform_out = standard_scaler_transform(&train, &model).unwrap();
        for (a, b) in fit_out.iter().zip(transform_out.iter()) {
            assert!((a - b).abs() < EPS);
        }
    }

    #[test]
    fn standard_scaler_transform_ncols_mismatch() {
        let train = array![[0.0f32, 0.0], [1.0, 1.0]];
        let (model, _) = standard_scaler_fit(&train);
        let bad = array![[0.0f32]];
        assert!(standard_scaler_transform(&bad, &model).is_err());
    }
}
