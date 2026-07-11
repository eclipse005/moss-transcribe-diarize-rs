//! VQAdaptor + 4× time-merge, mirror of the Python `VQAdaptor` / `time_merge`.
//!
//! `time_merge`: (B, T, D) → (B, T//M, D*M)  with M = audio_merge_size.
//! `VQAdaptor`: Linear(in→hidden) → SiLU → Linear(hidden→hidden) → LayerNorm(hidden).

use anyhow::Result;
use gemm::{gemm, Parallelism};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::raw_tensor::RawTensor;

/// (B, T, D) → (B, T_trim//M, D*M). Drops the trailing T%M frames like the Python version.
pub fn time_merge(data: &[f32], b: usize, t: usize, d: usize, merge: usize) -> (Vec<f32>, usize, usize) {
    let t_trim = (t / merge) * merge;
    let out_t = t_trim / merge;
    let out_d = d * merge;
    let mut out = vec![0.0f32; b * out_t * out_d];
    // For each output token, gather M consecutive frames of D features.
    out.par_chunks_mut(out_d).enumerate().for_each(|(idx, row)| {
        let ib = idx / out_t;
        let ot = idx % out_t;
        for m in 0..merge {
            for j in 0..d {
                let src_t = ot * merge + m;
                row[m * d + j] = data[(ib * t + src_t) * d + j];
            }
        }
    });
    (out, out_t, out_d)
}

pub struct CpuVqAdaptor {
    l0_w: Vec<f32>, l0_b: Vec<f32>,   // Linear(input_dim → hidden)
    l2_w: Vec<f32>, l2_b: Vec<f32>,   // Linear(hidden → hidden)
    ln_w: Vec<f32>, ln_b: Vec<f32>,   // LayerNorm(hidden)
    hidden: usize,
    eps: f32,
}

impl CpuVqAdaptor {
    pub fn load(weights: &HashMap<String, RawTensor>, prefix: &str, hidden: usize, eps: f32) -> Result<Self> {
        let g = |name: &str| -> Result<Vec<f32>> {
            Ok(weights.get(name).ok_or_else(|| anyhow::anyhow!("missing {}", name))?.to_f32_vec()?)
        };
        let p = |s: &str| format!("{}.{}", prefix, s);
        Ok(Self {
            l0_w: g(&p("layers.0.weight"))?,
            l0_b: g(&p("layers.0.bias"))?,
            l2_w: g(&p("layers.2.weight"))?,
            l2_b: g(&p("layers.2.bias"))?,
            ln_w: g(&p("layers.3.weight"))?,
            ln_b: g(&p("layers.3.bias"))?,
            hidden,
            eps,
        })
    }

    /// x: [B, T, input_dim]. Returns [B, T, hidden].
    pub fn forward(&self, x: &[f32], b: usize, t: usize, input_dim: usize) -> Vec<f32> {
        let m = b * t;
        // Linear 0: [m, input_dim] → [m, hidden]
        let mut h = vec![0.0f32; m * self.hidden];
        unsafe {
            gemm(
                m, self.hidden, input_dim,
                h.as_mut_ptr(), 1, self.hidden as isize, false,
                x.as_ptr(), 1, input_dim as isize,
                self.l0_w.as_ptr(), input_dim as isize, 1,
                0.0, 1.0, false, false, false,
                Parallelism::Rayon(0),
            );
        }
        h.par_chunks_mut(self.hidden).for_each(|row| {
            for j in 0..self.hidden { row[j] += self.l0_b[j]; }
            // SiLU
            for j in 0..self.hidden {
                let g = row[j];
                row[j] = g / (1.0 + (-g).exp());
            }
        });

        // Linear 2: [m, hidden] → [m, hidden]
        let mut h2 = vec![0.0f32; m * self.hidden];
        unsafe {
            gemm(
                m, self.hidden, self.hidden,
                h2.as_mut_ptr(), 1, self.hidden as isize, false,
                h.as_ptr(), 1, self.hidden as isize,
                self.l2_w.as_ptr(), self.hidden as isize, 1,
                0.0, 1.0, false, false, false,
                Parallelism::Rayon(0),
            );
        }
        h2.par_chunks_mut(self.hidden).for_each(|row| {
            for j in 0..self.hidden { row[j] += self.l2_b[j]; }
        });

        // LayerNorm(hidden)
        let mut out = vec![0.0f32; m * self.hidden];
        out.par_chunks_mut(self.hidden).zip(h2.par_chunks(self.hidden)).for_each(|(o, row)| {
            let mut mean = 0.0f64;
            for &v in row { mean += v as f64; }
            mean /= self.hidden as f64;
            let mut var = 0.0f64;
            for &v in row { let dd = v as f64 - mean; var += dd * dd; }
            var /= self.hidden as f64;
            let inv_std = 1.0 / (var + self.eps as f64).sqrt();
            for j in 0..self.hidden {
                o[j] = ((row[j] as f64 - mean) * inv_std * self.ln_w[j] as f64 + self.ln_b[j] as f64) as f32;
            }
        });
        out
    }
}
