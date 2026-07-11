//! CPU-resident text decoder for Qwen3-ASR.
//!
//! Mirrors the structure of `cudarc_engine.rs` but runs everything on the host CPU:
//! - `gemm` crate handles every matmul, **with `Parallelism::Rayon(0)` forced even on
//!   the m=1 GEMVs** that burn-flex leaves single-threaded (its threshold is m*n*k >= 7M
//!   and decode never reaches that).  This is the single biggest win — without it a
//!   20-core machine spends decode on one core.
//! - `rayon` parallelises every hand-written elementwise/reduction op (rms_norm,
//!   silu_mul, attention) across heads or rows.
//! - Activations are f32 (Vec<f32>). **Weights are stored as f16** and read directly
//!   by the m=1 GEMV (halved memory bandwidth → faster on memory-bound decode).
//!   For prefill (m>1), weights are converted to f32 before `gemm` crate calls.
//! - KV cache is pre-allocated per layer (Vec<f32> sized [b, nkvh, max_seq, d]).
//! - The embed table is reused as the lm_head weight (Qwen3 ties them).
//!
//! This module is `#[cfg(feature = "cpu")]` only — the CUDA path is untouched.

use anyhow::Result;
use crate::raw_tensor::RawTensor;
use gemm::{gemm, Parallelism};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::config::TextDecoderConfig;

// ═══════════════════════════════════════════════════════════════════════
//  Tensors (host-side, f32)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl CpuTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "CpuTensor len mismatch (shape {:?})", shape);
        Self { data, shape }
    }
    pub fn numel(&self) -> usize { self.data.len() }
    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        assert_eq!(self.numel(), shape.iter().product::<usize>());
        self.shape = shape; self
    }
}

pub(crate) struct CpuWeight {
    pub data: Vec<f32>,
    pub rows: usize,   // = out_features (N)
    pub cols: usize,   // = in_features  (K)
}

/// Weight matrix stored as f16 — halved memory vs f32.
/// Read directly by m=1 GEMV (halved bandwidth for memory-bound decode).
/// Converted to f32 on-the-fly for m>1 GEMM (prefill).
pub(crate) struct CpuWeightF16 {
    pub data: Vec<half::f16>,
    pub rows: usize,
    pub cols: usize,
}

impl CpuWeightF16 {
    /// Sequential f16→f32. Better for many small calls (audio encoder: ~100/forward).
    #[allow(dead_code)]
    pub(crate) fn to_f32(&self) -> CpuWeight {
        let data: Vec<f32> = self.data.iter().map(|v| v.to_f32()).collect();
        CpuWeight { data, rows: self.rows, cols: self.cols }
    }
}

/// Weight matrix stored as **per-channel symmetric INT8**: `scale[i] = max(|w_row_i|)/127`,
/// `data[i,j] ∈ [-127,127]`.  Halved memory vs f16, and (on AVX2) a 4× wider SIMD dot
/// product — the decode-path win.  `cols` is padded up to a multiple of 32 (zero-filled)
/// so the GEMV kernel is branch-free.
pub(crate) struct CpuWeightI8 {
    pub data: Vec<i8>,     // [rows, cols] row-major, cols padded to multiple of 32
    pub scale: Vec<f32>,   // [rows] per-output-row symmetric scale
    pub rows: usize,       // = N (out features)
    pub cols: usize,       // padded K (in features)
}

impl CpuWeightI8 {
    /// Per-channel symmetric quantisation from an f16 weight. Each output row gets its own
    /// scale = max(|row|)/127; values clipped to [-127,127] (using /127, not /128, so two
    /// quantised values multiply to ≤16129 — stays in i16 without saturation).  `cols` is
    /// padded to a multiple of 32 with zeros.
    pub(crate) fn from_f16(w: &CpuWeightF16) -> Self {
        let rows = w.rows;
        let k = w.cols;
        let kpad = (k + 31) / 32 * 32;
        let mut data = vec![0i8; rows * kpad];
        let mut scale = vec![0.0f32; rows];
        for i in 0..rows {
            let src = &w.data[i * k..(i + 1) * k];
            // Pass 1: per-row max abs.
            let mut amax = 0.0f32;
            for v in src { let a = v.to_f32().abs(); if a > amax { amax = a; } }
            let s = amax / 127.0;
            scale[i] = s;
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            // Pass 2: quantise + clip into the padded row (tail [k..kpad] stays zero).
            let dst = &mut data[i * kpad..(i + 1) * kpad];
            for j in 0..k {
                dst[j] = (src[j].to_f32() * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
        CpuWeightI8 { data, scale, rows, cols: kpad }
    }

    /// Dequantise to a real-k f32 weight for the prefill GEMM path (drops the 32-padding so
    /// the returned `CpuWeight` matches the f16 path's shape).  Rayon over rows.  One-time
    /// per transcribe call (prefill runs once).
    pub(crate) fn dequant_to_f32(&self, real_k: usize) -> CpuWeight {
        debug_assert!(real_k <= self.cols);
        let kpad = self.cols;
        let data: Vec<f32> = (0..self.rows).into_par_iter().flat_map_iter(|i| {
            let s = self.scale[i];
            let row = &self.data[i * kpad..i * kpad + real_k];
            row.iter().map(move |&q| q as f32 * s)
        }).collect();
        CpuWeight { data, rows: self.rows, cols: real_k }
    }
}

/// Decoder linear over an INT8 weight.  Decode (m=1) reads the INT8 weight directly (AVX2
/// GEMV); prefill (m>1) dequantises to f32 for the gemm crate.  The decoder body is permanently
/// INT8 — `lm_head` / `embed_table` / the audio encoder stay f16 via `linear_gemv_f16`.
fn linear_i8(x: &CpuTensor, w: &CpuWeightI8) -> CpuTensor {
    let nd = x.shape.len();
    let m: usize = x.shape[..nd - 1].iter().product();
    let k = x.shape[nd - 1];
    let n = w.rows;
    assert!(w.cols >= k && w.cols % 32 == 0,
            "linear I8 K mismatch: x last={} vs W padded cols={}", k, w.cols);
    let mut out_shape = x.shape.clone();
    out_shape[nd - 1] = n;
    if m == 1 {
        let out = linear_gemv_i8(&x.data, w);
        return CpuTensor::new(out, out_shape);
    }
    let w_f32 = w.dequant_to_f32(k);
    let mut out = vec![0.0f32; m * n];
    gemm_row_major(&mut out, &x.data, &w_f32, m, 0.0);
    CpuTensor::new(out, out_shape)
}

/// `linear_i8` with a residual add (`out += result`).  Decode (m=1) computes a fresh GEMV then
/// a scalar `+=`; prefill folds the residual via `gemm` beta=1.0.
fn linear_accum_i8(out: &mut CpuTensor, x: &CpuTensor, w: &CpuWeightI8) {
    let nd = x.shape.len();
    let m: usize = x.shape[..nd - 1].iter().product();
    let k = x.shape[nd - 1];
    let n = w.rows;
    assert!(w.cols >= k && w.cols % 32 == 0);
    assert_eq!(out.numel(), m * n);
    if m == 1 {
        let add = linear_gemv_i8(&x.data, w);
        for (o, a) in out.data.iter_mut().zip(add.iter()) { *o += *a; }
        return;
    }
    let w_f32 = w.dequant_to_f32(k);
    gemm_row_major(&mut out.data, &x.data, &w_f32, m, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════
//  Matmul: y = x @ W^T  with forced rayon parallelism
// ═══════════════════════════════════════════════════════════════════════
//
// Layout: x [m, k] row-major, W [n, k] row-major (PyTorch nn.Linear convention),
// y [m, n] row-major.  Computing y = x @ W^T means each y[i, j] = sum_k x[i, k] * W[j, k].
//
// gemm's stride API: dst_cs = stride for j+1, dst_rs = stride for i+1.  For row-major C[m,n]
// element (i, j) is at i*n + j, so cs=1, rs=n.  Same for x.  W^T conceptually is [k, n] with
// element (i, j) = W[j, i]; since W is row-major [n, k] (W[j, i] at j*k + i), B's cs = k, rs = 1.

#[allow(dead_code)]
pub(crate) fn linear(x: &CpuTensor, w: &CpuWeight) -> CpuTensor {
    let nd = x.shape.len();
    let m: usize = x.shape[..nd - 1].iter().product();
    let k = x.shape[nd - 1];
    let n = w.rows;
    assert_eq!(k, w.cols, "linear K mismatch: x last={} vs W cols={}", k, w.cols);
    let mut out_shape = x.shape.clone();
    out_shape[nd - 1] = n;
    // m=1 fast path: hand-written rayon GEMV avoids gemm crate's m=1 perf cliff.
    if m == 1 {
        let out = linear_gemv(&x.data, w);
        return CpuTensor::new(out, out_shape);
    }
    let mut out = vec![0.0f32; m * n];
    gemm_row_major(&mut out, &x.data, w, m, 0.0);
    CpuTensor::new(out, out_shape)
}

// y[i, j] = sum_k x[i, k] * W[j, k] + beta * y[i, j].  Row-major throughout.
// Forces gemm to use `Parallelism::Rayon(0)` (all cores) even for m=1 GEMVs, which is
// the whole point of this engine — burn-flex's internal threshold leaves decode
// single-threaded otherwise.
fn gemm_row_major(out: &mut [f32], x: &[f32], w: &CpuWeight, m: usize, beta: f32) {
    let n = w.rows;
    let k = w.cols;
    unsafe {
        gemm(
            m, n, k,
            out.as_mut_ptr(),
            1,                  // dst_cs (column stride for j+1, since C is row-major)
            n as isize,         // dst_rs (row stride for i+1)
            beta != 0.0,
            x.as_ptr(),
            1,                  // lhs_cs
            k as isize,         // lhs_rs
            w.data.as_ptr(),
            k as isize,         // rhs_cs (B is W^T; j+1 advances by k in W's row-major layout)
            1,                  // rhs_rs
            beta,
            1.0,
            false, false, false,
            Parallelism::Rayon(0),
        );
    }
}

/// Hand-written m=1 GEMV optimized for the lm_head case (n = vocab, k = hidden).
/// gemm's m=1 path was observed to be ~7x slower than the memory-bandwidth limit on the
/// Qwen3-ASR lm_head (304MB weight, 28ms vs 3.8ms theoretical), likely because its
/// parallelism partition doesn't split the n axis cleanly enough for cache-friendly streaming.
/// This routine partitions n into rayon chunks and runs an inner dot-product loop that the
/// auto-vectoriser turns into AVX2 FMA instructions.
fn linear_gemv(x: &[f32], w: &CpuWeight) -> Vec<f32> {
    let n = w.rows;
    let k = w.cols;
    debug_assert_eq!(x.len(), k);
    let mut out = vec![0.0f32; n];
    // Aim for ~128 rows per chunk so each thread streams ~512KB of weight (fits L2 nicely
    // and gives plenty of work to amortize rayon spawn overhead).
    let chunk = (n / (rayon::current_num_threads() * 4)).max(64).min(2048);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, slab)| {
        let row0 = ci * chunk;
        for (offset, o) in slab.iter_mut().enumerate() {
            let row = row0 + offset;
            let w_row = &w.data[row * k..(row + 1) * k];
            // Auto-vectorised dot product.  Using f64 accumulator would be slow on f32-only SIMD;
            // f32 sum is fine for inference (vocab logits don't need extra precision).
            let mut acc = 0.0f32;
            for j in 0..k { acc += x[j] * w_row[j]; }
            *o = acc;
        }
    });
    out
}

/// Same as `linear_gemv` but reads f16 weights directly — halved memory bandwidth.
/// For m=1 GEMV the bottleneck is streaming the weight matrix from DRAM; reading
/// 2 bytes/element instead of 4 can be ~2x faster on bandwidth-limited systems.
/// The f16→f32 conversion happens in-register alongside the FMA.
fn linear_gemv_f16(x: &[f32], w: &CpuWeightF16) -> Vec<f32> {
    let n = w.rows;
    let k = w.cols;
    debug_assert_eq!(x.len(), k);
    let mut out = vec![0.0f32; n];
    let chunk = (n / (rayon::current_num_threads() * 4)).max(64).min(2048);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, slab)| {
        let row0 = ci * chunk;
        for (offset, o) in slab.iter_mut().enumerate() {
            let row = row0 + offset;
            let w_row = &w.data[row * k..(row + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k {
                // f16→f32 conversion + multiply in one expression; compiler
                // can batch the conversions with SSE/AVX auto-vectorization.
                acc += x[j] * w_row[j].to_f32();
            }
            *o = acc;
        }
    });
    out
}

// ─── INT8-weight GEMV (weight-only quant; activation stays f32, quantised per call) ──

/// Single-scale symmetric activation quantisation: `xs = max|x|/127`, `xq ∈ [-127,127]`.
/// Pads to `kpad` (a multiple of 32) to match the weight's padded cols.  Done once per
/// GEMV call — m=1 reuses the same x across all output rows, so the O(k) cost amortises
/// over O(rows·k) to ≤0.1%.
fn quantize_activation(x: &[f32], kpad: usize) -> (Vec<i8>, f32) {
    debug_assert!(x.len() <= kpad);
    let k = x.len();
    let mut xq = vec![0i8; kpad];
    let mut amax = 0.0f32;
    for &v in x { let a = v.abs(); if a > amax { amax = a; } }
    let s = amax / 127.0;
    let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
    for j in 0..k {
        xq[j] = (x[j] * inv).round().clamp(-127.0, 127.0) as i8;
    }
    (xq, s)
}

/// m=1 GEMV over an INT8 weight — scalar fallback, mirrors `linear_gemv_f16`'s rayon
/// structure.  `out[i] = scale[i] * xs * Σ_j (xq[j] * wq[i,j])`.  The +0 padding tail
/// contributes nothing.  Correctness reference for the AVX2 kernel (M2).
fn linear_gemv_i8_scalar(x: &[f32], w: &CpuWeightI8) -> Vec<f32> {
    let n = w.rows;
    let kpad = w.cols;
    debug_assert!(x.len() <= kpad);
    let (xq, xs) = quantize_activation(x, kpad);
    let mut out = vec![0.0f32; n];
    let chunk = (n / (rayon::current_num_threads() * 4)).max(64).min(2048);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, slab)| {
        let row0 = ci * chunk;
        for (offset, o) in slab.iter_mut().enumerate() {
            let row = row0 + offset;
            let wq = &w.data[row * kpad..(row + 1) * kpad];
            let ws = w.scale[row];
            let mut acc = 0.0f32;
            for j in 0..kpad {
                acc += xq[j] as f32 * wq[j] as f32;
            }
            *o = acc * ws * xs;
        }
    });
    out
}

/// AVX2 INT8 m=1 GEMV.  Per 32-element block: load 32 i8 (weight) × 32 i8 (activation),
/// widen each half to i16, then `_mm256_madd_epi16` does the signed i16×i16→i32 pairwise
/// product-sum directly (one instruction — products ≤16129, pair sums ≤32258, no saturation).
/// Eight i32 lanes accumulate across blocks, reduced once per row.  `quantize_activation`
/// runs once per call (m=1 shares x across all rows).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn linear_gemv_i8_avx2(x: &[f32], w: &CpuWeightI8) -> Vec<f32> {
    use core::arch::x86_64::*;
    let n = w.rows;
    let kpad = w.cols;
    debug_assert!(x.len() <= kpad && kpad % 32 == 0);
    let nblocks = kpad / 32;
    let (xq, xs) = quantize_activation(x, kpad);
    let mut out = vec![0.0f32; n];
    let chunk = (n / (rayon::current_num_threads() * 4)).max(64).min(2048);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, slab)| {
        let row0 = ci * chunk;
        // Raw pointers are derived inside the closure (not captured) so it stays `Send`.
        unsafe {
            let xq_ptr = xq.as_ptr();
            let wdata = w.data.as_ptr();
            for (offset, o) in slab.iter_mut().enumerate() {
                let row = row0 + offset;
                let wq_ptr = wdata.add(row * kpad);
                let ws = w.scale[row];
                let mut acc = _mm256_setzero_si256();
                for b in 0..nblocks {
                    let va = _mm256_loadu_si256(xq_ptr.add(b * 32) as *const __m256i);
                    let vb = _mm256_loadu_si256(wq_ptr.add(b * 32) as *const __m256i);
                    // Widen the two 128-bit halves of each vector i8×16 → i16×16.
                    let va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
                    let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(va));
                    let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
                    let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(vb));
                    // i16×i16 → pairwise i32 sum (8 i32 per half). No saturation in range.
                    let lo = _mm256_madd_epi16(va_lo, vb_lo);
                    let hi = _mm256_madd_epi16(va_hi, vb_hi);
                    acc = _mm256_add_epi32(acc, _mm256_add_epi32(lo, hi));
                }
                let mut lanes = [0i32; 8];
                _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
                let total: i32 = lanes.iter().sum();
                *o = total as f32 * ws * xs;
            }
        }
    });
    out
}

/// INT8 m=1 GEMV dispatcher.  AVX2 when the CPU supports it, else the scalar kernel.
/// `is_x86_feature_detected!` is mandatory — calling the AVX2 kernel on a non-AVX2 CPU
/// would be an illegal instruction (candle #2140).
fn linear_gemv_i8(x: &[f32], w: &CpuWeightI8) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { linear_gemv_i8_avx2(x, w) };
    }
    linear_gemv_i8_scalar(x, w)
}

// ═══════════════════════════════════════════════════════════════════════
//  Elementwise / reduction ops (rayon-parallel by row)
// ═══════════════════════════════════════════════════════════════════════

/// out[i, j] = x[i, j] * w[j] / sqrt(mean(x[i, :]^2) + eps)
pub(crate) fn rms_norm(x: &CpuTensor, w: &[f32], eps: f32) -> CpuTensor {
    let nd = x.shape.len();
    let last = x.shape[nd - 1];
    let outer: usize = x.shape[..nd - 1].iter().product();
    assert_eq!(w.len(), last);
    let mut out = vec![0.0f32; outer * last];
    out.par_chunks_mut(last)
        .zip(x.data.par_chunks(last))
        .for_each(|(o, xrow)| {
            let mut ss = 0.0f64;
            for &v in xrow { ss += (v as f64) * (v as f64); }
            let inv_rms = 1.0 / ((ss / last as f64 + eps as f64).sqrt() as f32);
            for j in 0..last {
                o[j] = xrow[j] * inv_rms * w[j];
            }
        });
    CpuTensor::new(out, x.shape.clone())
}

/// out = silu(gate) * up where gu = [gate | up] along last dim.  gu: [outer, 2*inter] → out: [outer, inter].
pub(crate) fn silu_mul_split(gu: &CpuTensor) -> CpuTensor {
    let nd = gu.shape.len();
    let two_inter = gu.shape[nd - 1];
    let inter = two_inter / 2;
    let outer: usize = gu.shape[..nd - 1].iter().product();
    let mut out = vec![0.0f32; outer * inter];
    out.par_chunks_mut(inter)
        .zip(gu.data.par_chunks(two_inter))
        .for_each(|(o, row)| {
            let (gate, up) = row.split_at(inter);
            for j in 0..inter {
                let g = gate[j];
                let sig = 1.0 / (1.0 + (-g).exp());
                o[j] = g * sig * up[j];
            }
        });
    let mut shape = gu.shape.clone();
    shape[nd - 1] = inter;
    CpuTensor::new(out, shape)
}

/// Embedding lookup from f16 table — converts each row to f32 on the fly.
pub(crate) fn embed_lookup_f16(table: &CpuWeightF16, ids: &[i64]) -> CpuTensor {
    let n = ids.len();
    let d = table.cols;
    let mut out = vec![0.0f32; n * d];
    out.par_chunks_mut(d)
        .zip(ids.par_iter())
        .for_each(|(o, &id)| {
            let src = &table.data[(id as usize) * d..(id as usize + 1) * d];
            for j in 0..d { o[j] = src[j].to_f32(); }
        });
    CpuTensor::new(out, vec![n, d])
}

/// Argmax of a flat slice.  Chunked rayon reduce (par_iter().max_by() has surprising
/// overhead per element on small types).
pub fn argmax(x: &[f32]) -> i32 {
    const CHUNK: usize = 4096;
    let n = x.len();
    let (idx, _) = (0..n).step_by(CHUNK).collect::<Vec<_>>()
        .par_iter()
        .map(|&start| {
            let end = (start + CHUNK).min(n);
            let mut best_idx = start;
            let mut best_val = x[start];
            for i in (start + 1)..end {
                if x[i] > best_val { best_val = x[i]; best_idx = i; }
            }
            (best_idx, best_val)
        })
        .reduce(|| (0usize, f32::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
    idx as i32
}

// ═══════════════════════════════════════════════════════════════════════
//  Rotary embedding (in-place on Q or K head row)
// ═══════════════════════════════════════════════════════════════════════
//
// For a head row of length d, with `cos[i]` and `sin[i]` (i in [0, d)), the standard
// "half-rotation" Qwen formulation is:
//   for i < d/2:        out[i] = x[i]*cos[i] - x[i+d/2]*sin[i]
//   for i in [d/2, d):  out[i] = x[i]*cos[i] + x[i-d/2]*sin[i]

#[inline]
fn apply_rotary_row(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let d = x.len();
    let half = d / 2;
    // Cache halves before overwrite.
    let mut tmp = [0.0f32; 256]; // head_dim <= 256 covers Qwen3-ASR (hd=128)
    debug_assert!(d <= tmp.len());
    tmp[..d].copy_from_slice(x);
    for i in 0..half {
        x[i]        = tmp[i]        * cos[i]        - tmp[i + half] * sin[i];
        x[i + half] = tmp[i + half] * cos[i + half] + tmp[i]        * sin[i + half];
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Fused QKV extract + (Q,K) norm + rotary + KV cache write
// ═══════════════════════════════════════════════════════════════════════
//
// qkv: [b, s, total_cols] where total_cols = q_dim + 2*kv_dim
//   q_dim = nqh * hd,  kv_dim = nkvh * hd
// Outputs:
//   q_out: [b, nqh, s, hd]   (norm+rotary applied)
//   k_cache, v_cache: [b, nkvh, max_seq, hd]  at rows [start..start+s)
//     - K: norm+rotary; V: raw copy
pub(crate) fn qkv_extract_qkv_norm_rotary_cache(
    qkv: &CpuTensor,
    qn_w: &[f32], kn_w: &[f32],
    cos_table: &[f32], sin_table: &[f32],
    k_cache: &mut [f32], v_cache: &mut [f32],
    b: usize, nqh: usize, nkvh: usize, hd: usize,
    q_dim: usize, kv_dim: usize,
    max_seq: usize, start: usize, pos_offset: usize, eps: f32,
) -> CpuTensor {
    let s = qkv.shape[1];
    let total_cols = qkv.shape[2];
    let mut q_out = vec![0.0f32; b * nqh * s * hd];

    // Parallelise over (b, s) — each token is independent.  Within a token we sequentially
    // handle all heads (nqh + nkvh ≤ 24 for Qwen3-ASR, not worth nested rayon).
    //
    // We need disjoint &mut access to the per-(b,s) slices of q_out, k_cache, v_cache.  Use
    // par_chunks_mut on q_out and index k_cache/v_cache by computed offset (the writes for
    // different (b, s) never overlap because of the (start + is) offset into max_seq).
    let q_per_token = nqh * hd;
    // Pre-split for parallel iteration.
    q_out.par_chunks_mut(q_per_token).enumerate().for_each(|(token_idx, q_dst)| {
        let ib = token_idx / s;
        let is = token_idx % s;
        let cs_row = pos_offset + is;
        let cos_row = &cos_table[cs_row * hd..(cs_row + 1) * hd];
        let sin_row = &sin_table[cs_row * hd..(cs_row + 1) * hd];

        let row_base = (ib * s + is) * total_cols;
        let qkv_row = &qkv.data[row_base..row_base + total_cols];
        let q_src = &qkv_row[..q_dim];
        let k_src = &qkv_row[q_dim..q_dim + kv_dim];
        let v_src = &qkv_row[q_dim + kv_dim..];

        // Q: per head, RMSNorm(qn_w) → rotary → write to q_dst
        for h in 0..nqh {
            let head_in = &q_src[h * hd..(h + 1) * hd];
            let head_out = &mut q_dst[h * hd..(h + 1) * hd];
            let mut ss = 0.0f64;
            for &v in head_in { ss += (v as f64) * (v as f64); }
            let inv_rms = 1.0 / ((ss / hd as f64 + eps as f64).sqrt() as f32);
            for j in 0..hd { head_out[j] = head_in[j] * inv_rms * qn_w[j]; }
            apply_rotary_row(head_out, cos_row, sin_row);
        }

        // K: per kv-head, RMSNorm(kn_w) → rotary → write to k_cache[ib, h, start+is, :]
        // V: raw copy to v_cache.
        // SAFETY: k_cache/v_cache are passed in as &mut [f32].  Different (ib, is) write
        // disjoint slots in max_seq, and different (ib, h) write disjoint heads.  Per-token
        // parallel iteration above is the only outer parallelism.
        let k_cache_ptr = k_cache.as_ptr() as *mut f32;
        let v_cache_ptr = v_cache.as_ptr() as *mut f32;
        for h in 0..nkvh {
            let k_in = &k_src[h * hd..(h + 1) * hd];
            let v_in = &v_src[h * hd..(h + 1) * hd];
            let cache_idx = ((ib * nkvh + h) * max_seq + (start + is)) * hd;
            unsafe {
                let k_dst = std::slice::from_raw_parts_mut(k_cache_ptr.add(cache_idx), hd);
                let v_dst = std::slice::from_raw_parts_mut(v_cache_ptr.add(cache_idx), hd);
                let mut ss = 0.0f64;
                for &v in k_in { ss += (v as f64) * (v as f64); }
                let inv_rms = 1.0 / ((ss / hd as f64 + eps as f64).sqrt() as f32);
                for j in 0..hd { k_dst[j] = k_in[j] * inv_rms * kn_w[j]; }
                apply_rotary_row(k_dst, cos_row, sin_row);
                v_dst.copy_from_slice(v_in);
            }
        }
    });

    CpuTensor::new(q_out, vec![b, nqh, s, hd])
}

// ═══════════════════════════════════════════════════════════════════════
//  Fused GQA attention (decode-only, s_q = 1)
// ═══════════════════════════════════════════════════════════════════════
//
// q: [b, nqh, 1, hd]
// k_cache, v_cache: [b, nkvh, max_seq, hd]  (valid rows [0..cur_len))
// out: [b, nqh, 1, hd]
//
// One (b, qh) per rayon job: compute scores = q · K^T, softmax, out = attn · V.
// Done in f32 throughout; cheap vs prefill because cur_len is small (decode-only).
pub(crate) fn fused_gqa_decode(
    q: &CpuTensor,
    k_cache: &[f32], v_cache: &[f32],
    b: usize, nqh: usize, nkvh: usize, max_seq: usize, hd: usize, cur_len: usize,
    scale: f32,
) -> CpuTensor {
    let n_rep = nqh / nkvh;
    let mut out = vec![0.0f32; b * nqh * hd];
    out.par_chunks_mut(hd).enumerate().for_each(|(idx, out_head)| {
        let ib = idx / nqh;
        let qh = idx % nqh;
        let kh = qh / n_rep;
        let q_row = &q.data[(ib * nqh + qh) * hd..(ib * nqh + qh + 1) * hd];
        let k_base = (ib * nkvh + kh) * max_seq * hd;
        let v_base = (ib * nkvh + kh) * max_seq * hd;

        // Compute scores[0..cur_len] in a scratch buffer.
        let mut scores = vec![0.0f32; cur_len];
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..cur_len {
            let k_row = &k_cache[k_base + t * hd..k_base + (t + 1) * hd];
            let mut dot = 0.0f32;
            for j in 0..hd { dot += q_row[j] * k_row[j]; }
            let s = dot * scale;
            scores[t] = s;
            if s > max_score { max_score = s; }
        }
        // softmax
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;
        // out_head[j] = sum_t scores[t] * V[t, j] * inv_sum
        for j in 0..hd { out_head[j] = 0.0; }
        for t in 0..cur_len {
            let v_row = &v_cache[v_base + t * hd..v_base + (t + 1) * hd];
            let w = scores[t] * inv_sum;
            for j in 0..hd { out_head[j] += w * v_row[j]; }
        }
    });
    CpuTensor::new(out, vec![b, nqh, 1, hd])
}

// ═══════════════════════════════════════════════════════════════════════
//  Decoder layer
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuDecoderLayer {
    pub iln_w: Vec<f32>,
    pub pln_w: Vec<f32>,
    pub qn_w: Vec<f32>,
    pub kn_w: Vec<f32>,
    pub qkv_w: CpuWeightI8,
    pub o_w: CpuWeightI8,
    pub gu_w: CpuWeightI8,   // [2*inter, hidden]
    pub dp_w: CpuWeightI8,   // [hidden, inter]
    pub nqh: usize, pub nkvh: usize, pub hd: usize, pub eps: f32,
}

impl CpuDecoderLayer {
    pub fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        cfg: &TextDecoderConfig,
    ) -> Result<Self> {
        // The four decoder-body linears are permanently INT8 (per-channel symmetric quantisation,
        // AVX2 GEMV at decode time). Loaded as f16 from safetensors, then quantised once here.
        // Validated vs the f16 path on all 7 fixtures: normalised CER ~1%, RTFx 1.2-1.6×.
        let q = |w: Result<CpuWeightF16>| -> Result<CpuWeightI8> {
            Ok(CpuWeightI8::from_f16(&w?))
        };
        Ok(Self {
            iln_w: load_vec_f32(weights, &format!("{}.input_layernorm.weight", prefix))?,
            pln_w: load_vec_f32(weights, &format!("{}.post_attention_layernorm.weight", prefix))?,
            qn_w: load_vec_f32(weights, &format!("{}.self_attn.q_norm.weight", prefix))?,
            kn_w: load_vec_f32(weights, &format!("{}.self_attn.k_norm.weight", prefix))?,
            qkv_w: q(load_fused_qkv_f16(weights, &format!("{}.self_attn", prefix)))?,
            o_w: q(load_weight_f16(weights, &format!("{}.self_attn.o_proj.weight", prefix)))?,
            gu_w: q(load_fused_gate_up_f16(weights, &format!("{}.mlp", prefix)))?,
            dp_w: q(load_weight_f16(weights, &format!("{}.mlp.down_proj.weight", prefix)))?,
            nqh: cfg.num_attention_heads,
            nkvh: cfg.num_key_value_heads,
            hd: cfg.head_dim,
            eps: cfg.rms_norm_eps as f32,
        })
    }

    /// x: [b, s, hidden] consumed; returns h (post all residuals).
    pub fn forward(
        &self,
        x: CpuTensor,
        cos_table: &[f32], sin_table: &[f32],
        kv: &mut CpuKvCache, layer_idx: usize,
        kv_start: usize, use_causal: bool,
    ) -> CpuTensor {
        let b = x.shape[0]; let s = x.shape[1];

        // 1. Input RMSNorm
        let normed = rms_norm(&x, &self.iln_w, self.eps);

        // 2. Fused QKV linear
        let qkv = linear_i8(&normed, &self.qkv_w);
        drop(normed);
        let q_dim = self.nqh * self.hd;
        let kv_dim = self.nkvh * self.hd;

        // 3. Extract Q+K+V, norm+rotary Q&K, write K/V to cache.
        let q = qkv_extract_qkv_norm_rotary_cache(
            &qkv, &self.qn_w, &self.kn_w, cos_table, sin_table,
            &mut kv.k[layer_idx], &mut kv.v[layer_idx],
            b, self.nqh, self.nkvh, self.hd, q_dim, kv_dim,
            kv.max_seq, kv_start, kv_start, self.eps,
        );
        drop(qkv);
        let cur_len = kv_start + s;

        // 4. Attention.  Decode-only path (s == 1) uses fused_gqa_decode; prefill (s > 1)
        //    materialises full attention scores via two matmuls.
        let attn_out = if s == 1 {
            let scale = 1.0f32 / (self.hd as f32).sqrt();
            fused_gqa_decode(&q, &kv.k[layer_idx], &kv.v[layer_idx],
                b, self.nqh, self.nkvh, kv.max_seq, self.hd, cur_len, scale)
        } else {
            prefill_attention(&q, &kv.k[layer_idx], &kv.v[layer_idx],
                b, self.nqh, self.nkvh, kv.max_seq, self.hd, cur_len, use_causal)
        };

        // 5. O projection with residual add.  attn_out is laid out as [b, s, nqh, hd]
        //    regardless of whether decode (fused_gqa_decode) or prefill (prefill_attention)
        //    produced it, so we can reshape directly to [b, s, nqh*hd] — no swap needed.
        let attn_flat = attn_out.reshape(vec![b, s, self.nqh * self.hd]);
        let mut h = x;
        linear_accum_i8(&mut h, &attn_flat, &self.o_w);
        drop(attn_flat);

        // 6. Post-attn RMSNorm
        let normed2 = rms_norm(&h, &self.pln_w, self.eps);

        // 7. Gate-up linear → SiLU·up
        let gu = linear_i8(&normed2, &self.gu_w);
        drop(normed2);
        let activated = silu_mul_split(&gu);
        drop(gu);

        // 8. Down projection with residual add
        linear_accum_i8(&mut h, &activated, &self.dp_w);
        h
    }
}

// Prefill-path attention: scores = q @ K^T, softmax, out = scores @ V.  Both matmuls go through
// the `gemm` crate (AVX2-FMA + cache tiling) rather than hand-rolled scalar loops — those
// compiled to SSE2-only on the default baseline and dominated prefill (~60% at 180s).  Layouts:
//   q bytes:   [b, s, nqh, hd]   — q[ib, i, qh, :] at ((ib*s+i)*nqh + qh)*hd, stride nqh*hd per token
//   k/v cache: [b, nkvh, max_seq, hd] → per (ib, kh) a contiguous [cur_len, hd] view
//   out bytes: [b, s, nqh, hd]   — same strided layout as q
// GQA: n_rep = nqh/nkvh query heads share each kv-head's K/V.  One rayon job per (b, qh); the
// gemm calls use Parallelism::None (single-threaded per job) so the b*nqh jobs fill the cores
// without nested-parallelism oversubscription.  The gemm stride API reads q / writes out in
// place — no gather/scatter temporaries.
fn prefill_attention(
    q: &CpuTensor,
    k_cache: &[f32], v_cache: &[f32],
    b: usize, nqh: usize, nkvh: usize, max_seq: usize, hd: usize, cur_len: usize,
    causal: bool,
) -> CpuTensor {
    let s = q.shape[2];
    let n_rep = nqh / nkvh;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let tok_stride = (nqh * hd) as isize; // f32 elements between consecutive tokens in q / out
    let out = vec![0.0f32; b * s * nqh * hd];

    (0..b * nqh).into_par_iter().for_each(|idx| {
        let ib = idx / nqh;
        let qh = idx % nqh;
        let kh = qh / n_rep;
        let kv_base = (ib * nkvh + kh) * max_seq * hd;
        let k_head = &k_cache[kv_base..kv_base + cur_len * hd]; // [cur_len, hd]
        let v_head = &v_cache[kv_base..kv_base + cur_len * hd];
        let head_off = ((ib * s) * nqh + qh) * hd; // start of q[ib,0,qh,:] and out[ib,0,qh,:]

        // scores[s, cur_len] = q[ib,:,qh,:] @ K_head^T  (q read with per-token stride; no gather).
        let mut scores = vec![0.0f32; s * cur_len];
        unsafe {
            let q_ptr = q.data.as_ptr().add(head_off);
            gemm(
                s, cur_len, hd,
                scores.as_mut_ptr(), 1, cur_len as isize, false,
                q_ptr, 1, tok_stride,          // q strided: lhs_cs=1, lhs_rs=tok_stride
                k_head.as_ptr(), hd as isize, 1, // K as K^T: rhs_cs=hd, rhs_rs=1
                0.0, 1.0, false, false, false,
                Parallelism::None,
            );
        }
        // scale + causal mask (masked → -inf so softmax zeroes them).
        let offset = cur_len - s;
        for i in 0..s {
            let limit = if causal { i + offset + 1 } else { cur_len };
            let row = &mut scores[i * cur_len..(i + 1) * cur_len];
            for (t, cell) in row.iter_mut().enumerate() {
                if t >= limit { *cell = f32::NEG_INFINITY; } else { *cell *= scale; }
            }
        }
        // softmax per row.
        for i in 0..s {
            let row = &mut scores[i * cur_len..(i + 1) * cur_len];
            let mut mx = f32::NEG_INFINITY;
            for &v in row.iter() { if v > mx { mx = v; } }
            let mut sum = 0.0f32;
            for v in row.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            let inv = 1.0 / sum;
            for v in row.iter_mut() { *v *= inv; }
        }
        // out[ib,:,qh,:] = scores @ V_head  (written with per-token stride; no scatter).
        // SAFETY: each (ib, qh) job writes disjoint strided rows of `out`; derived from a shared
        // &out inside the closure so it stays Send (raw ptrs themselves are !Send).
        unsafe {
            let out_ptr = out.as_ptr() as *mut f32;
            let dst = out_ptr.add(head_off);
            gemm(
                s, hd, cur_len,
                dst, 1, tok_stride, false,        // out strided: dst_cs=1, dst_rs=tok_stride
                scores.as_ptr(), 1, cur_len as isize,
                v_head.as_ptr(), 1, hd as isize,  // V [cur_len,hd] standard: rhs_cs=1, rhs_rs=hd
                0.0, 1.0, false, false, false,
                Parallelism::None,
            );
        }
    });

    // Logical shape [b, nqh, s, hd] but actual bytes are [b, s, nqh, hd] — the caller reshapes
    // to [b, s, nqh*hd] without swap_dims_12.
    CpuTensor::new(out, vec![b, nqh, s, hd])
}

// ═══════════════════════════════════════════════════════════════════════
//  KV cache
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub(crate) struct CpuKvCache {
    pub k: Vec<Vec<f32>>,  // per layer: [b, nkvh, max_seq, hd]
    pub v: Vec<Vec<f32>>,
    pub cur_len: usize,
    pub max_seq: usize,
    pub b: usize,
    pub nkvh: usize,
    pub hd: usize,
}

impl CpuKvCache {
    pub fn new(num_layers: usize, b: usize, nkvh: usize, max_seq: usize, hd: usize) -> Self {
        let cap = b * nkvh * max_seq * hd;
        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k.push(vec![0.0; cap]);
            v.push(vec![0.0; cap]);
        }
        Self { k, v, cur_len: 0, max_seq, b, nkvh, hd }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Text decoder
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub(crate) struct CpuTextDecoder {
    pub embed_table: CpuWeightF16,   // [vocab, hidden] (reused as lm_head)
    pub layers: Vec<CpuDecoderLayer>,
    pub norm_w: Vec<f32>,            // [hidden]
    pub eps: f32,
    pub config: TextDecoderConfig,
}

impl CpuTextDecoder {
    pub fn load(weights: &HashMap<String, RawTensor>, prefix: &str, config: &TextDecoderConfig) -> Result<Self> {
        let embed_table = load_weight_f16(weights, &format!("{}.embed_tokens.weight", prefix))?;
        let norm_w = load_vec_f32(weights, &format!("{}.norm.weight", prefix))?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(CpuDecoderLayer::load(weights, &format!("{}.layers.{}", prefix, i), config)?);
        }
        Ok(Self { embed_table, layers, norm_w, eps: config.rms_norm_eps as f32, config: config.clone() })
    }

    /// Embed ids into [n, hidden] tensor.
    pub fn embed_ids(&self, ids: &[i64]) -> CpuTensor {
        embed_lookup_f16(&self.embed_table, ids)
    }

    /// Forward pass.
    /// hs: [1, sl, hidden].  cos/sin_table: full [total_positions, hd] tables.
    /// kv_start: how many positions already in cache.
    /// Returns logits as [1, out_sl, vocab] (out_sl = 1 if llo).
    pub fn forward(
        &self,
        hs: CpuTensor,
        cos_table: &[f32], sin_table: &[f32],
        kv: &mut CpuKvCache, kv_start: usize, use_causal: bool, llo: bool,
    ) -> CpuTensor {
        let sl = hs.shape[1];
        let mut h = hs;
        let t_layers = std::time::Instant::now();
        // Per-layer timing — emit only on sample step 100 of decode (s=1).
        let mut layer_ms = [0f64; 28];
        for (i, layer) in self.layers.iter().enumerate() {
            let t = std::time::Instant::now();
            h = layer.forward(h, cos_table, sin_table, kv, i, kv_start, use_causal);
            layer_ms[i] = t.elapsed().as_secs_f64() * 1000.0;
        }
        let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
        kv.cur_len = kv_start + sl;

        // Final RMSNorm
        let t_post = std::time::Instant::now();
        let h = rms_norm(&h, &self.norm_w, self.eps);

        // Low-latency: keep only last token for prefill
        let h = if llo && sl > 1 {
            let hidden = h.shape[2];
            let last_off = (sl - 1) * hidden;
            let last_data = h.data[last_off..last_off + hidden].to_vec();
            CpuTensor::new(last_data, vec![1, 1, hidden])
        } else {
            h
        };

        // lm_head (shared with embed_table): hand-written m=1 GEMV reading f16 directly.
        let logits_vec = linear_gemv_f16(&h.data, &self.embed_table);
        let vocab = self.embed_table.rows;
        let logits = CpuTensor::new(logits_vec, vec![1, 1, vocab]);
        let post_ms = t_post.elapsed().as_secs_f64() * 1000.0;
        if sl == 1 {
            if kv_start % 100 == 99 {
                let mut s = format!("step kv_start={}: layers={:.2}ms final+lm={:.2}ms | per layer: ",
                                    kv_start, layers_ms, post_ms);
                for (i, &ms) in layer_ms.iter().enumerate() {
                    if i > 0 { s.push_str(", "); }
                    s.push_str(&format!("{}:{:.2}", i, ms));
                }
                log::info!("{}", s);
            }
        }
        logits
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Standard RoPE cos/sin precompute (f32; cpu_engine path)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) use crate::mrope::compute_rope_cos_sin;

// ═══════════════════════════════════════════════════════════════════════
//  Weight loading helpers
// ═══════════════════════════════════════════════════════════════════════

fn load_f32_vec(weights: &HashMap<String, RawTensor>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let td = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?;
    let (data, shape) = td.as_f32()?;
    Ok((data, shape))
}

fn load_vec_f32(weights: &HashMap<String, RawTensor>, name: &str) -> Result<Vec<f32>> {
    let (data, _) = load_f32_vec(weights, name)?;
    Ok(data)
}

fn load_f16_vec(weights: &HashMap<String, RawTensor>, name: &str) -> Result<(Vec<half::f16>, Vec<usize>)> {
    let td = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?;
    let (data, shape) = td.as_f16()?;
    Ok((data, shape))
}

fn load_weight_f16(weights: &HashMap<String, RawTensor>, name: &str) -> Result<CpuWeightF16> {
    let (data, shape) = load_f16_vec(weights, name)?;
    assert_eq!(shape.len(), 2, "weight {} should be 2D", name);
    Ok(CpuWeightF16 { data, rows: shape[0], cols: shape[1] })
}

/// Load Q+K+V projections and concatenate into a single [q_dim + 2*kv_dim, hidden] f16 matrix.
fn load_fused_qkv_f16(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<CpuWeightF16> {
    let (qw, qs) = load_f16_vec(weights, &format!("{}.q_proj.weight", prefix))?;
    let (kw, ks) = load_f16_vec(weights, &format!("{}.k_proj.weight", prefix))?;
    let (vw, vs) = load_f16_vec(weights, &format!("{}.v_proj.weight", prefix))?;
    let q_dim = qs[0]; let kv_dim = ks[0]; let hidden = qs[1];
    assert_eq!(ks[1], hidden); assert_eq!(vs[1], hidden);
    let mut fused = Vec::with_capacity((q_dim + 2 * kv_dim) * hidden);
    fused.extend_from_slice(&qw);
    fused.extend_from_slice(&kw);
    fused.extend_from_slice(&vw);
    Ok(CpuWeightF16 { data: fused, rows: q_dim + 2 * kv_dim, cols: hidden })
}

/// Load gate_proj and up_proj, concatenate into [2*inter, hidden] f16 matrix.
fn load_fused_gate_up_f16(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<CpuWeightF16> {
    let (gw, gs) = load_f16_vec(weights, &format!("{}.gate_proj.weight", prefix))?;
    let (uw, us) = load_f16_vec(weights, &format!("{}.up_proj.weight", prefix))?;
    let inter = gs[0]; let hidden = gs[1];
    assert_eq!(us[0], inter); assert_eq!(us[1], hidden);
    let mut fused = Vec::with_capacity(2 * inter * hidden);
    fused.extend_from_slice(&gw);
    fused.extend_from_slice(&uw);
    Ok(CpuWeightF16 { data: fused, rows: 2 * inter, cols: hidden })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16w(rows: usize, cols: usize, vals: &[f32]) -> CpuWeightF16 {
        assert_eq!(vals.len(), rows * cols);
        CpuWeightF16 {
            data: vals.iter().map(|&v| half::f16::from_f32(v)).collect(),
            rows,
            cols,
        }
    }

    /// INT8 scalar GEMV must stay close to the f16 GEMV reference (per-channel weight quant
    /// + single-scale activation quant).  This is the M1 correctness gate; the AVX2 kernel
    /// (M2) is checked against this scalar kernel separately.
    #[test]
    fn int8_scalar_gemv_matches_f16() {
        // 4×8 weight (cols zero-padded to 32 inside `from_f16`).
        let w = f16w(4, 8, &[
            0.5, -0.25, 0.125, 0.9, -0.6, 0.3, -0.1, 0.45,
            -0.8, 0.2, 0.33, -0.5, 0.7, -0.15, 0.6, -0.4,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            -0.95, 0.55, -0.35, 0.75, -0.65, 0.85, -0.45, 0.25,
        ]);
        let x: Vec<f32> = vec![0.7, -0.4, 0.2, 0.9, -0.3, 0.6, -0.8, 0.15];
        assert_eq!(x.len(), w.cols);

        let wi = CpuWeightI8::from_f16(&w);
        assert_eq!(wi.cols, 32, "cols must be padded to a multiple of 32");
        assert_eq!(wi.rows, 4);

        let ref_out = linear_gemv_f16(&x, &w);
        let i8_out = linear_gemv_i8_scalar(&x, &wi);

        assert_eq!(ref_out.len(), i8_out.len());
        for (r, i) in ref_out.iter().zip(i8_out.iter()) {
            let tol = 0.05 * r.abs() + 0.1;
            assert!((r - i).abs() < tol, "int8 vs f16: ref={} got={} diff={}", r, i, r - i);
        }
    }

    /// A zero activation must yield (exactly) zero output — exercises the `amax == 0` guard.
    #[test]
    fn int8_zero_activation() {
        let w = f16w(2, 8, &[0.5; 16]);
        let wi = CpuWeightI8::from_f16(&w);
        let x = vec![0.0f32; 8];
        let out = linear_gemv_i8_scalar(&x, &wi);
        for v in &out {
            assert!(v.abs() < 1e-5, "zero activation should give ~0, got {}", v);
        }
    }

    /// `dequant_to_f32` (prefill path) must approximate the original f16 weight per row.
    #[test]
    fn int8_dequant_approximates_f16() {
        let w = f16w(3, 8, &[
            0.5, -0.25, 0.125, 0.9, -0.6, 0.3, -0.1, 0.45,
            -0.8, 0.2, 0.33, -0.5, 0.7, -0.15, 0.6, -0.4,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        ]);
        let wi = CpuWeightI8::from_f16(&w);
        let dq = wi.dequant_to_f32(w.cols); // real k = 8
        assert_eq!((dq.rows, dq.cols), (3, 8));
        for i in 0..w.rows {
            for j in 0..w.cols {
                let orig = w.data[i * w.cols + j].to_f32();
                let tol = 0.06 * orig.abs() + wi.scale[i];
                assert!((dq.data[i * w.cols + j] - orig).abs() < tol,
                        "dequant row {} col {}: orig={} got={}", i, j, orig, dq.data[i * w.cols + j]);
            }
        }
    }

    /// AVX2 kernel must match the scalar kernel element-wise — the M2 SIMD correctness gate.
    /// They should agree near-bit-exactly: both accumulate exact integer products (i8×i8 fits
    /// i32; 32 terms ≤ ~516k stay exact in both i32 and f32) before the single `* ws * xs`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn int8_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping: host has no AVX2");
            return;
        }
        // 8×32 weight, diverse values in [-0.9, 0.9] (a sin pattern exercises mixed signs
        // and varied per-row magnitudes → varied per-channel scales).
        let vals: Vec<f32> = (0..(8 * 32)).map(|k| ((k as f32) * 0.037).sin() * 0.9).collect();
        let w = f16w(8, 32, &vals);
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.07 - 1.1).collect();
        assert_eq!(x.len(), w.cols);

        let wi = CpuWeightI8::from_f16(&w);
        let scalar = linear_gemv_i8_scalar(&x, &wi);
        let avx2 = unsafe { linear_gemv_i8_avx2(&x, &wi) };

        assert_eq!(scalar.len(), avx2.len());
        for (s, a) in scalar.iter().zip(avx2.iter()) {
            assert!((s - a).abs() < 1e-3 * s.abs() + 1e-4,
                    "avx2 vs scalar: scalar={} avx2={}", s, a);
        }
    }

    /// `prefill_attention` (gemm-crate path) must match a brute-force causal GQA reference.
    /// Guards the strided q/out access, the K^T vs V rhs strides, causal mask, and softmax.
    #[test]
    fn prefill_attention_matches_reference() {
        let b = 1; let nqh = 4; let nkvh = 2; let hd = 8;
        let s = 3; let cur_len = 3; let max_seq = 4;
        let n_rep = nqh / nkvh;
        let scale = 1.0f32 / (hd as f32).sqrt();
        // q bytes [b, s, nqh, hd]; k/v cache [b, nkvh, max_seq, hd].
        let q = CpuTensor::new(
            (0..b * s * nqh * hd).map(|i| ((i as f32) * 0.13).sin()).collect(),
            vec![b, nqh, s, hd],
        );
        let kv_len = b * nkvh * max_seq * hd;
        let k_cache: Vec<f32> = (0..kv_len).map(|i| ((i as f32) * 0.17).cos()).collect();
        let v_cache: Vec<f32> = (0..kv_len).map(|i| ((i as f32) * 0.11).sin()).collect();
        let causal = true;
        let out = prefill_attention(&q, &k_cache, &v_cache, b, nqh, nkvh, max_seq, hd, cur_len, causal);

        let offset = cur_len - s;
        for ib in 0..b {
            for qh in 0..nqh {
                let kh = qh / n_rep;
                for i in 0..s {
                    let limit = if causal { i + offset + 1 } else { cur_len };
                    // scores[t] = (q · K[t]) * scale, masked → 0 weight.
                    let mut scores = vec![0.0f32; cur_len];
                    for t in 0..limit {
                        let mut dot = 0.0f32;
                        for j in 0..hd {
                            dot += q.data[((ib * s + i) * nqh + qh) * hd + j]
                                * k_cache[((ib * nkvh + kh) * max_seq + t) * hd + j];
                        }
                        scores[t] = dot * scale;
                    }
                    let mx = (0..limit).map(|t| scores[t]).fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for t in 0..limit { scores[t] = (scores[t] - mx).exp(); sum += scores[t]; }
                    let inv = 1.0 / sum;
                    for t in 0..limit { scores[t] *= inv; }
                    for j in 0..hd {
                        let mut acc = 0.0f32;
                        for t in 0..limit {
                            acc += scores[t] * v_cache[((ib * nkvh + kh) * max_seq + t) * hd + j];
                        }
                        let got = out.data[((ib * s + i) * nqh + qh) * hd + j];
                        assert!((got - acc).abs() < 1e-4 * acc.abs() + 1e-5,
                                "prefill_attn mismatch ib={} qh={} i={} j={}: got={} exp={}",
                                ib, qh, i, j, got, acc);
                    }
                }
            }
        }
    }
}
