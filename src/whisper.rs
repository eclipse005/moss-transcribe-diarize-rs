//! CPU Whisper-Medium audio encoder (mirror of HF WhisperEncoder).
//!
//! Pipeline (per the Python `get_audio_features`):
//!   mel [num_mel, T] → conv1(k3,s2)+GELU → conv2(k3,s2)+GELU → permute(0,2,1)
//!     → + learned embed_positions[0..T_out]
//!     → 24 encoder layers (pre-LN attn + pre-LN GELU FFN, full bidirectional)
//!     → final layer_norm
//! Returns last_hidden_state [1, T_out, d_model].
//!
//! Attention is **full** (bidirectional, no mask, no causal) — Whisper at inference
//! feeds the whole log-mel frame range at once.

use anyhow::Result;
use gemm::{gemm, Parallelism};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::config::WhisperAudioConfig;
use crate::raw_tensor::RawTensor;

// ─── CPU tensors (local to the encoder) ───────────────────────────────

pub(crate) struct CpuEncTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl CpuEncTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected);
        Self { data, shape }
    }
    #[allow(dead_code)]
    pub(crate) fn numel(&self) -> usize { self.data.len() }
}

// ─── Whisper conv1d (kernel=3, stride=2, padding=1) ───────────────────
//
// PyTorch nn.Conv1d stores weight as [out_channels, in_channels, kernel_size].
// For kernel=3, stride=2, pad=1: out_t = (in_t + 2*pad - kernel)/stride + 1.

/// Conv1d kernel=3, padding=1, configurable stride (Whisper: conv1 stride=1, conv2 stride=2).
/// Weight layout: [out_ch, in_ch, 3] (PyTorch). x: [b, in_ch, in_t].
fn conv1d_k3(x: &CpuEncTensor, weight: &[f32], bias: &[f32],
             in_ch: usize, out_ch: usize, stride: usize) -> CpuEncTensor {
    let b = x.shape[0];
    let in_t = x.shape[2];
    let out_t = (in_t + 2 - 3) / stride + 1;
    let mut out = vec![0.0f32; b * out_ch * out_t];
    out.par_chunks_mut(out_ch * out_t).enumerate().for_each(|(ib, slab)| {
        for oc in 0..out_ch {
            let w_row = &weight[oc * in_ch * 3..(oc + 1) * in_ch * 3];
            let bias_oc = bias[oc];
            for ot in 0..out_t {
                let mut acc = bias_oc;
                for ic in 0..in_ch {
                    let x_base = ib * in_ch * in_t + ic * in_t;
                    let w_base = ic * 3;
                    // ot maps to input center ot*stride; kernel taps ot*stride-1, ot*stride, ot*stride+1 (pad=1)
                    for k in 0..3usize {
                        let it = (ot * stride) as isize + k as isize - 1;
                        if it >= 0 && (it as usize) < in_t {
                            acc += x.data[x_base + it as usize] * w_row[w_base + k];
                        }
                    }
                }
                slab[oc * out_t + ot] = acc;
            }
        }
    });
    CpuEncTensor::new(out, vec![b, out_ch, out_t])
}

fn gelu_inplace(x: &mut [f32]) {
    // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    x.par_iter_mut().for_each(|v| {
        *v = 0.5 * *v * (1.0 + erf(*v * 0.7071067811865475));
    });
}

// erf approximation (Abramowitz-Stegun 7.1.26); HF uses torch.erf which is exact.
// For f32 inference alignment we use a high-accuracy rational approx.
fn erf(x: f32) -> f32 {
    // Use the same erff that the GPU kernel relies on conceptually; here a
    // numerically tight approximation (max error ~1.5e-7) so f32 alignment holds.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    // constants
    let a1 = 0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 = 1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 = 1.061405429_f32;
    let p = 0.3275911_f32;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ─── LayerNorm (with bias) ────────────────────────────────────────────

fn layer_norm(x: &CpuEncTensor, w: &[f32], bias: &[f32], eps: f32) -> CpuEncTensor {
    let last = x.shape[x.shape.len() - 1];
    let outer: usize = x.shape[..x.shape.len() - 1].iter().product();
    let mut out = vec![0.0f32; outer * last];
    out.par_chunks_mut(last).zip(x.data.par_chunks(last)).for_each(|(o, row)| {
        let mut mean = 0.0f64;
        for &v in row { mean += v as f64; }
        mean /= last as f64;
        let mut var = 0.0f64;
        for &v in row { let d = v as f64 - mean; var += d * d; }
        var /= last as f64;
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        for j in 0..last {
            o[j] = ((row[j] as f64 - mean) * inv_std * w[j] as f64 + bias[j] as f64) as f32;
        }
    });
    CpuEncTensor::new(out, x.shape.clone())
}

// ─── Linear (matmul y = x @ W^T) for encoder ──────────────────────────

fn linear(x: &CpuEncTensor, w: &[f32], bias: Option<&[f32]>, n_out: usize) -> CpuEncTensor {
    let nd = x.shape.len();
    let m: usize = x.shape[..nd - 1].iter().product();
    let k = x.shape[nd - 1];
    let mut out = vec![0.0f32; m * n_out];
    unsafe {
        gemm(
            m, n_out, k,
            out.as_mut_ptr(), 1, n_out as isize, false,
            x.data.as_ptr(), 1, k as isize,
            w.as_ptr(), k as isize, 1,
            0.0, 1.0, false, false, false,
            Parallelism::Rayon(0),
        );
    }
    if let Some(b) = bias {
        out.par_chunks_mut(n_out).for_each(|row| {
            for j in 0..n_out { row[j] += b[j]; }
        });
    }
    let mut shape = x.shape.clone();
    shape[nd - 1] = n_out;
    CpuEncTensor::new(out, shape)
}

// ─── Full multi-head attention (bidirectional, no mask) ───────────────

fn attention(x: &CpuEncTensor, q_w: &[f32], q_b: &[f32],
             k_w: &[f32], k_b: &[f32],
             v_w: &[f32], v_b: &[f32],
             o_w: &[f32], o_b: &[f32],
             d_model: usize, n_heads: usize) -> CpuEncTensor {
    let b = x.shape[0];
    let s = x.shape[1];
    let head_dim = d_model / n_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q = linear(x, q_w, Some(q_b), d_model);
    let k = linear(x, k_w, Some(k_b), d_model);
    let v = linear(x, v_w, Some(v_b), d_model);
    // q,k,v: [b, s, d_model] — layout is [b, s, n_heads, head_dim] interleaved

    // scores per (b, head): [s, s] = q_head @ k_head^T * scale → softmax → @ v_head
    let out = vec![0.0f32; b * s * d_model];
    // SAFETY: each (b, head) writes disjoint head_dim slices of `out`. The raw
    // pointer is derived inside the closure (not captured) so the closure stays Send.
    //
    // QK^T and AV use the gemm crate (tiled AVX2-FMA microkernels) instead of
    // hand-written scalar triple loops — the same proven pattern as the decoder's
    // prefill_attention. The q/k/v head slices are contiguous (head_dim f32 each)
    // and the per-token stride within q/k/v is d_model, so gemm gets efficient
    // row-major access on K^T and V.
    (0..b * n_heads).into_par_iter().for_each(|idx| {
        let ib = idx / n_heads;
        let h = idx % n_heads;
        let off = h * head_dim;

        // --- QK^T: scores[s, s] = q_head[s,hd] @ k_head^T[hd,s] ---
        // q/k/v are laid out as [b, s, d_model] with heads interleaved, so token i's
        // head-h slice is at [i*d_model + h*head_dim .. .. + head_dim].
        // For gemm: C[i,j] = sum_k A[i,k] * B[k,j], we need B[k,j] = k[j,k].
        // A = q: lhs_cs=1 (contiguous head_dim), lhs_rs=d_model (token stride)
        // B^T = K^T: rhs_cs=d_model (advance j→j+1), rhs_rs=1 (advance k→k+1)
        let mut scores = vec![0.0f32; s * s];
        unsafe {
            let q_ptr = q.data.as_ptr().add((ib * s) * d_model + off);
            let k_ptr = k.data.as_ptr().add((ib * s) * d_model + off);
            gemm(
                s, s, head_dim,
                scores.as_mut_ptr(), 1, s as isize, false,
                q_ptr, 1, d_model as isize,
                k_ptr, d_model as isize, 1,
                0.0, scale, false, false, false,
                Parallelism::None,
            );
        }

        // softmax per row (bidirectional — no mask)
        for i in 0..s {
            let row = &mut scores[i * s..(i + 1) * s];
            let mut mx = f32::NEG_INFINITY;
            for &v in row.iter() { if v > mx { mx = v; } }
            let mut sum = 0.0f32;
            for v in row.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            let inv = 1.0 / sum;
            for v in row.iter_mut() { *v *= inv; }
        }

        // --- AV: out_head[s, hd] = scores[s,s] @ v_head[s,hd] ---
        unsafe {
            let out_ptr = out.as_ptr() as *mut f32;
            let dst = out_ptr.add((ib * s) * d_model + off);
            let v_ptr = v.data.as_ptr().add((ib * s) * d_model + off);
            gemm(
                s, head_dim, s,
                dst, 1, d_model as isize, false,
                scores.as_ptr(), 1, s as isize,
                v_ptr, 1, d_model as isize,      // V strided: rhs_cs=1, rhs_rs=d_model
                0.0, 1.0, false, false, false,
                Parallelism::None,
            );
        }
    });

    let attn = CpuEncTensor::new(out, x.shape.clone());
    // output projection
    linear(&attn, o_w, Some(o_b), d_model)
}

// ─── Encoder layer ────────────────────────────────────────────────────

struct CpuWhisperLayer {
    sln_w: Vec<f32>, sln_b: Vec<f32>,   // self_attn_layer_norm
    fln_w: Vec<f32>, fln_b: Vec<f32>,   // final_layer_norm
    // attention
    q_w: Vec<f32>, q_b: Vec<f32>,
    k_w: Vec<f32>, k_b: Vec<f32>,
    v_w: Vec<f32>, v_b: Vec<f32>,
    o_w: Vec<f32>, o_b: Vec<f32>,
    // ffn
    fc1_w: Vec<f32>, fc1_b: Vec<f32>,
    fc2_w: Vec<f32>, fc2_b: Vec<f32>,
}

impl CpuWhisperLayer {
    fn load(weights: &HashMap<String, RawTensor>, prefix: &str, cfg: &WhisperAudioConfig) -> Result<Self> {
        let g = |name: &str| -> Result<Vec<f32>> {
            Ok(weights.get(name).ok_or_else(|| anyhow::anyhow!("missing {}", name))?.to_f32_vec()?)
        };
        // Whisper attention: k_proj has bias=False (per HF WhisperAttention); q/v/out have bias.
        let g_opt = |name: &str| -> Vec<f32> {
            weights.get(name).map(|t| t.to_f32_vec().unwrap_or_default()).unwrap_or_default()
        };
        let p = |s: &str| format!("{}.{}", prefix, s);
        let d_model = cfg.d_model;
        Ok(Self {
            sln_w: g(&p("self_attn_layer_norm.weight"))?,
            sln_b: g(&p("self_attn_layer_norm.bias"))?,
            fln_w: g(&p("final_layer_norm.weight"))?,
            fln_b: g(&p("final_layer_norm.bias"))?,
            q_w: g(&p("self_attn.q_proj.weight"))?,
            q_b: g(&p("self_attn.q_proj.bias"))?,
            k_w: g(&p("self_attn.k_proj.weight"))?,
            k_b: { let b = g_opt(&p("self_attn.k_proj.bias")); if b.is_empty() { vec![0.0; d_model] } else { b } },
            v_w: g(&p("self_attn.v_proj.weight"))?,
            v_b: g(&p("self_attn.v_proj.bias"))?,
            o_w: g(&p("self_attn.out_proj.weight"))?,
            o_b: g(&p("self_attn.out_proj.bias"))?,
            fc1_w: g(&p("fc1.weight"))?,
            fc1_b: g(&p("fc1.bias"))?,
            fc2_w: g(&p("fc2.weight"))?,
            fc2_b: g(&p("fc2.bias"))?,
        })
    }

    fn forward(&self, x: &CpuEncTensor, d_model: usize, n_heads: usize, ffn: usize, eps: f32) -> CpuEncTensor {
        // residual = x; h = self_attn_layer_norm(x); attn = self_attn(h); x = residual + attn
        let normed = layer_norm(x, &self.sln_w, &self.sln_b, eps);
        let attn = attention(&normed, &self.q_w, &self.q_b, &self.k_w, &self.k_b,
                             &self.v_w, &self.v_b, &self.o_w, &self.o_b, d_model, n_heads);
        let mut h = CpuEncTensor::new(
            x.data.iter().zip(attn.data.iter()).map(|(a, b)| a + b).collect(),
            x.shape.clone(),
        );
        // residual = h; h2 = final_layer_norm(h); ffn = fc2(gelu(fc1(h2))); h = residual + ffn
        let normed2 = layer_norm(&h, &self.fln_w, &self.fln_b, eps);
        let mut hidden = linear(&normed2, &self.fc1_w, Some(&self.fc1_b), ffn);
        gelu_inplace(&mut hidden.data);
        let ffn_out = linear(&hidden, &self.fc2_w, Some(&self.fc2_b), d_model);
        for (v, a) in h.data.iter_mut().zip(ffn_out.data.iter()) { *v += a; }
        h
    }
}

// Suppress unused-field warnings for config-derived fields stored for clarity.
impl CpuWhisperLayer {
    // dims are passed through `forward` from the encoder; no stored fields needed.
}

// ─── Full encoder ─────────────────────────────────────────────────────

pub(crate) struct CpuWhisperEncoder {
    conv1_w: Vec<f32>, conv1_b: Vec<f32>,   // [d_model, num_mel, 1]
    conv2_w: Vec<f32>, conv2_b: Vec<f32>,   // [d_model, d_model, 1]
    embed_positions: Vec<f32>,              // [max_source_positions, d_model]
    layers: Vec<CpuWhisperLayer>,
    ln_w: Vec<f32>, ln_b: Vec<f32>,         // final layer_norm
    d_model: usize,
    n_heads: usize,
    ffn: usize,
    eps: f32,
}

impl CpuWhisperEncoder {
    pub(crate) fn load(weights: &HashMap<String, RawTensor>, prefix: &str, cfg: &WhisperAudioConfig) -> Result<Self> {
        let g = |name: &str| -> Result<Vec<f32>> {
            Ok(weights.get(name).ok_or_else(|| anyhow::anyhow!("missing {}", name))?.to_f32_vec()?)
        };
        let p = |s: &str| format!("{}.{}", prefix, s);
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            layers.push(CpuWhisperLayer::load(weights, &format!("{}.layers.{}", prefix, i), cfg)?);
        }
        Ok(Self {
            conv1_w: g(&p("conv1.weight"))?,
            conv1_b: g(&p("conv1.bias"))?,
            conv2_w: g(&p("conv2.weight"))?,
            conv2_b: g(&p("conv2.bias"))?,
            embed_positions: g(&p("embed_positions.weight"))?,
            layers,
            ln_w: g(&p("layer_norm.weight"))?,
            ln_b: g(&p("layer_norm.bias"))?,
            d_model: cfg.d_model,
            n_heads: cfg.encoder_attention_heads,
            ffn: cfg.encoder_ffn_dim,
            eps: 1e-5, // Whisper uses default LayerNorm eps 1e-5
        })
    }

    /// mel: [1, num_mel, nb_max_frames] (f32). Returns [1, T_out, d_model].
    pub(crate) fn forward(&self, mel: &CpuEncTensor) -> CpuEncTensor {
        // conv1 (k=3, s=1, p=1) + GELU, conv2 (k=3, s=2, p=1) + GELU.
        let num_mel = mel.shape[1];
        let mut x = conv1d_k3(mel, &self.conv1_w, &self.conv1_b, num_mel, self.d_model, 1);
        gelu_inplace(&mut x.data);
        let mut x = conv1d_k3(&x, &self.conv2_w, &self.conv2_b, self.d_model, self.d_model, 2);
        gelu_inplace(&mut x.data);

        // permute(0, 2, 1): [b, d_model, T_out] → [b, T_out, d_model]
        let b = x.shape[0];
        let t_out = x.shape[2];
        let d = self.d_model;
        let mut perm = vec![0.0f32; b * t_out * d];
        for ib in 0..b {
            for t in 0..t_out {
                for c in 0..d {
                    perm[(ib * t_out + t) * d + c] = x.data[(ib * d + c) * t_out + t];
                }
            }
        }
        let mut h = CpuEncTensor::new(perm, vec![b, t_out, d]);

        // + learned positional embeddings (rows 0..t_out)
        for t in 0..t_out {
            let pe_row = &self.embed_positions[t * d..(t + 1) * d];
            for ib in 0..b {
                let dst = &mut h.data[(ib * t_out + t) * d..(ib * t_out + t + 1) * d];
                for j in 0..d { dst[j] += pe_row[j]; }
            }
        }

        // encoder layers
        for layer in &self.layers {
            h = layer.forward(&h, d, self.n_heads, self.ffn, self.eps);
        }

        // final layer_norm
        layer_norm(&h, &self.ln_w, &self.ln_b, self.eps)
    }
}
