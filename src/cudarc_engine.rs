//! GPU-resident text decoder for Qwen3-ASR.
//!
//! All operational tensors live in GPU memory as `GpuTensor` (CudaSlice<bf16>).
//! cuBLAS handles matmul; hand-written CUDA kernels handle every element-wise
//! op (RMSNorm, SiLU·up, softmax, rotary, repeat-KV, embed, argmax, etc.).
//! There are no CPU↔GPU round-trips in the steady-state decode loop —
//! weights, KV cache, MRoPE tables, and the embedding table are all uploaded
//! once at load time and stay on the device.

use anyhow::Result;
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use half::bf16;
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::TextDecoderConfig;
use crate::raw_tensor::RawTensor;

const KERNEL_SRC: &str = include_str!("kernels/kernels.cu");

// ═══════════════════════════════════════════════════════════════════════
//  GpuTensor — owned bf16 tensor on the GPU
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct GpuTensor {
    pub(crate) data: CudaSlice<bf16>,
    shape: Vec<usize>,
}

impl GpuTensor {
    pub fn new(data: CudaSlice<bf16>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "GpuTensor data len mismatch");
        Self { data, shape }
    }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn numel(&self) -> usize { self.data.len() }
    /// Reshape without moving data.
    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        assert_eq!(self.data.len(), shape.iter().product::<usize>());
        Self { data: self.data.clone(), shape }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  CpuTensor — used only for weight loading + initial input staging
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub(crate) struct CpuTensor {
    pub data: Vec<bf16>,
    pub shape: Vec<usize>,
}

impl CpuTensor {
    pub fn new(data: Vec<bf16>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "CpuTensor len mismatch");
        Self { data, shape }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  CudaState — context, stream, cuBLAS handle, kernel registry
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub(crate) struct CudaKernels {
    pub rms_norm: CudaFunction,
    pub add_residual_rms_norm: CudaFunction,
    pub add: CudaFunction,
    pub add_inplace: CudaFunction,
    pub silu_mul: CudaFunction,
    pub silu_mul_split: CudaFunction,
    pub softmax_causal: CudaFunction,
    pub rotary_emb: CudaFunction,
    pub rms_norm_rotary: CudaFunction,
    pub repeat_kv_from_cache: CudaFunction,
    pub embed_lookup: CudaFunction,
    pub embed_lookup_single_i32: CudaFunction,
    pub argmax: CudaFunction,
    pub argmax_into_slot: CudaFunction,
    pub swap_dims_12: CudaFunction,
    pub qkv_split: CudaFunction,
    pub qkv_extract_q_norm_rotary: CudaFunction,
    pub qkv_extract_kv_norm_rotary_cache: CudaFunction,
    pub qkv_extract_qkv_norm_rotary_cache: CudaFunction,
    pub kv_cache_write: CudaFunction,
    pub kv_cache_write_pair: CudaFunction,
    pub gelu: CudaFunction,
    pub gelu_inplace: CudaFunction,
    pub layer_norm: CudaFunction,
    pub add_bias_inplace: CudaFunction,
    pub slice_dim2: CudaFunction,
    pub concat_dim2_write: CudaFunction,
    pub im2col_3x3: CudaFunction,
    pub conv_postprocess: CudaFunction,
    pub fused_gqa_decode: CudaFunction,
    pub fused_gqa_decode_split_p1: CudaFunction,
    pub fused_gqa_decode_split_p2: CudaFunction,
    pub permute_bcft_to_btcf: CudaFunction,
    pub add_pe: CudaFunction,
    pub conv1d_k3_gelu: CudaFunction,
    pub permute_bct_to_btc: CudaFunction,
}

pub(crate) struct CudaState {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    pub k: CudaKernels,
}

unsafe impl Send for CudaState {}
unsafe impl Sync for CudaState {}

impl CudaState {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        Self::new_with_ctx(&ctx)
    }

    pub fn new_with_ctx(ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;

        // Enable Tensor Core math mode — no-op on Pascal, ensures TC usage on Ampere+
        unsafe {
            sys::cublasSetMathMode(*blas.handle(), sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH);
        }

        // NVRTC: target native arch for better codegen
        let cuda_include = std::env::var("CUDA_PATH")
            .map(|p| format!("{}/include", p))
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        // Leak the arch string — this runs once at init, ~20 bytes is negligible.
        let arch: Option<&'static str> = ctx.compute_capability().ok().map(|(major, minor)| {
            &*Box::leak(format!("sm_{}{}", major, minor).into_boxed_str())
        });
        let opts = CompileOptions {
            arch,
            include_paths: vec![cuda_include],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)
            .map_err(|e| anyhow::anyhow!("kernel compile failed: {:?}", e))?;
        let module = ctx.load_module(ptx)?;

        let k = CudaKernels {
            rms_norm: module.load_function("rms_norm_f16")?,
            add_residual_rms_norm: module.load_function("add_residual_rms_norm_f16")?,
            add: module.load_function("add_f16")?,
            add_inplace: module.load_function("add_inplace_f16")?,
            silu_mul: module.load_function("silu_mul_f16")?,
            silu_mul_split: module.load_function("silu_mul_split_f16")?,
            softmax_causal: module.load_function("softmax_scaled_causal_f16")?,
            rotary_emb: module.load_function("rotary_emb_f16")?,
            rms_norm_rotary: module.load_function("rms_norm_rotary_f16")?,
            repeat_kv_from_cache: module.load_function("repeat_kv_from_cache_f16")?,
            embed_lookup: module.load_function("embed_lookup_f16")?,
            embed_lookup_single_i32: module.load_function("embed_lookup_single_i32_f16")?,
            argmax: module.load_function("argmax_f16")?,
            argmax_into_slot: module.load_function("argmax_into_slot_f16")?,
            swap_dims_12: module.load_function("swap_dims_12_f16")?,
            qkv_split: module.load_function("qkv_split_f16")?,
            qkv_extract_q_norm_rotary: module.load_function("qkv_extract_q_norm_rotary_f16")?,
            qkv_extract_kv_norm_rotary_cache: module.load_function("qkv_extract_kv_norm_rotary_cache_f16")?,
            qkv_extract_qkv_norm_rotary_cache: module.load_function("qkv_extract_qkv_norm_rotary_cache_f16")?,
            kv_cache_write: module.load_function("kv_cache_write_f16")?,
            kv_cache_write_pair: module.load_function("kv_cache_write_pair_f16")?,
            gelu: module.load_function("gelu_f16")?,
            gelu_inplace: module.load_function("gelu_inplace_f16")?,
            layer_norm: module.load_function("layer_norm_f16")?,
            add_bias_inplace: module.load_function("add_bias_inplace_f16")?,
            slice_dim2: module.load_function("slice_dim2_f16")?,
            concat_dim2_write: module.load_function("concat_dim2_write_f16")?,
            im2col_3x3: module.load_function("im2col_3x3_s2p1_f16")?,
            conv_postprocess: module.load_function("conv_postprocess_f16")?,
            fused_gqa_decode: module.load_function("fused_gqa_decode_f16")?,
            fused_gqa_decode_split_p1: module.load_function("fused_gqa_decode_split_p1_f16")?,
            fused_gqa_decode_split_p2: module.load_function("fused_gqa_decode_split_p2_f16")?,
            permute_bcft_to_btcf: module.load_function("permute_bcft_to_btcf_f16")?,
            add_pe: module.load_function("add_pe_f16")?,
            conv1d_k3_gelu: module.load_function("conv1d_k3_gelu_f16")?,
            permute_bct_to_btc: module.load_function("permute_bct_to_btc_f16")?,
        };

        Ok(Self { ctx: ctx.clone(), stream, blas, k })
    }

    pub fn upload_f16(&self, data: &[bf16]) -> Result<CudaSlice<bf16>> {
        Ok(self.stream.clone_htod(data)?)
    }
    pub fn upload_i64(&self, data: &[i64]) -> Result<CudaSlice<i64>> {
        Ok(self.stream.clone_htod(data)?)
    }
    pub fn alloc_zeros_f16(&self, n: usize) -> Result<CudaSlice<bf16>> {
        Ok(self.stream.alloc_zeros::<bf16>(n)?)
    }
    /// Allocate uninitialized bf16 — caller MUST ensure every byte is written before read.
    /// Saves one memset_d8_async vs `alloc_zeros_f16` for cuBLAS/kernel outputs that are
    /// fully overwritten (beta=0 GEMM, fused attention writing all of `out`, etc.).
    pub fn alloc_uninit_f16(&self, n: usize) -> Result<CudaSlice<bf16>> {
        Ok(unsafe { self.stream.alloc::<bf16>(n)? })
    }
    /// Allocate uninitialized i32 — same semantics as `alloc_uninit_f16`.
    pub fn alloc_uninit_i32(&self, n: usize) -> Result<CudaSlice<i32>> {
        Ok(unsafe { self.stream.alloc::<i32>(n)? })
    }
    pub fn download_f16(&self, slice: &CudaSlice<bf16>) -> Result<Vec<bf16>> {
        Ok(self.stream.clone_dtoh(slice)?)
    }
    pub fn download_i32(&self, slice: &CudaSlice<i32>) -> Result<Vec<i32>> {
        Ok(self.stream.clone_dtoh(slice)?)
    }
    /// D2H into pinned host memory. `dst.as_ptr()` synchronizes then returns the value.
    pub fn download_i32_into_pinned(&self, src: &CudaSlice<i32>, dst: &mut PinnedHostSlice<i32>) -> Result<()> {
        Ok(self.stream.memcpy_dtoh(src, dst)?)
    }

    pub fn upload_tensor(&self, t: &CpuTensor) -> Result<GpuTensor> {
        let d = self.upload_f16(&t.data)?;
        Ok(GpuTensor::new(d, t.shape.clone()))
    }
    pub fn download_tensor(&self, t: &GpuTensor) -> Result<CpuTensor> {
        let d = self.download_f16(&t.data)?;
        Ok(CpuTensor::new(d, t.shape.clone()))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  cuBLAS wrappers (GPU-resident)
// ═══════════════════════════════════════════════════════════════════════

impl CudaState {
    /// y = x @ W^T   (x: [..., K], W: [N, K], y: [..., N])
    pub fn linear_gpu(&self, x: &GpuTensor, w: &GpuWeight) -> Result<GpuTensor> {
        let nd = x.shape().len();
        let m: usize = x.shape()[..nd - 1].iter().product();
        let k = x.shape()[nd - 1];
        let n = w.rows;
        assert_eq!(k, w.cols, "linear K mismatch: x last={} vs W cols={}", k, w.cols);
        let mut y = self.alloc_uninit_f16(m * n)?;
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32, n: m as i32, k: k as i32,
                    alpha: bf16::from_f32(1.0),
                    lda: k as i32, ldb: k as i32,
                    beta: bf16::from_f32(0.0), ldc: n as i32,
                },
                &w.data, &x.data, &mut y,
            )?;
        }
        let mut out_shape = x.shape().to_vec();
        out_shape[nd - 1] = n;
        Ok(GpuTensor::new(y, out_shape))
    }

    /// y = y + x @ W^T  — cuBLAS GEMM with beta=1, in-place accumulation on `y`.
    /// Used to fuse a residual add into a linear projection (saves an add_inplace launch).
    pub fn linear_gpu_accum(&self, y: &mut GpuTensor, x: &GpuTensor, w: &GpuWeight) -> Result<()> {
        let nd = x.shape().len();
        let m: usize = x.shape()[..nd - 1].iter().product();
        let k = x.shape()[nd - 1];
        let n = w.rows;
        assert_eq!(k, w.cols, "linear_gpu_accum K mismatch: x last={} vs W cols={}", k, w.cols);
        assert_eq!(y.numel(), m * n, "linear_gpu_accum y size mismatch");
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32, n: m as i32, k: k as i32,
                    alpha: bf16::from_f32(1.0),
                    lda: k as i32, ldb: k as i32,
                    beta: bf16::from_f32(1.0), ldc: n as i32,
                },
                &w.data, &x.data, &mut y.data,
            )?;
        }
        Ok(())
    }

    /// scores = Q @ K^T  (Q: [b,h,m,d], K: [b,h,n,d] → [b,h,m,n])
    pub fn attention_qk(&self, q: &GpuTensor, k: &GpuTensor) -> Result<GpuTensor> {
        let b = q.shape()[0]; let h = q.shape()[1]; let m = q.shape()[2]; let d = q.shape()[3];
        let n = k.shape()[2];
        let mut s = self.alloc_uninit_f16(b * h * m * n)?;
        let batch = (b * h) as i32;
        unsafe {
            self.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32, n: m as i32, k: d as i32,
                        alpha: bf16::from_f32(1.0),
                        lda: d as i32, ldb: d as i32,
                        beta: bf16::from_f32(0.0), ldc: n as i32,
                    },
                    batch_size: batch,
                    stride_a: (n * d) as i64,
                    stride_b: (m * d) as i64,
                    stride_c: (m * n) as i64,
                },
                &k.data, &q.data, &mut s,
            )?;
        }
        Ok(GpuTensor::new(s, vec![b, h, m, n]))
    }

    /// out = attn @ V  (attn: [b,h,m,n], V: [b,h,n,d] → [b,h,m,d])
    pub fn attention_av(&self, attn: &GpuTensor, v: &GpuTensor) -> Result<GpuTensor> {
        let b = attn.shape()[0]; let h = attn.shape()[1];
        let m = attn.shape()[2]; let n = attn.shape()[3];
        let d = v.shape()[3];
        let mut o = self.alloc_uninit_f16(b * h * m * d)?;
        let batch = (b * h) as i32;
        unsafe {
            self.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_N,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: d as i32, n: m as i32, k: n as i32,
                        alpha: bf16::from_f32(1.0),
                        lda: d as i32, ldb: n as i32,
                        beta: bf16::from_f32(0.0), ldc: d as i32,
                    },
                    batch_size: batch,
                    stride_a: (n * d) as i64,
                    stride_b: (m * n) as i64,
                    stride_c: (m * d) as i64,
                },
                &v.data, &attn.data, &mut o,
            )?;
        }
        Ok(GpuTensor::new(o, vec![b, h, m, d]))
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  GPU element-wise kernel wrappers
// ═══════════════════════════════════════════════════════════════════════

fn block_for_reduction(last: usize) -> u32 {
    // Power-of-two block size, max 1024, at least 32. The reduction needs bs to be power of 2.
    let mut bs: u32 = 32;
    let target = last as u32;
    while bs < target && bs < 1024 { bs *= 2; }
    bs.min(1024).max(32)
}

#[allow(dead_code)]
impl CudaState {
    pub fn rms_norm(&self, x: &GpuTensor, w: &CudaSlice<bf16>, eps: f32) -> Result<GpuTensor> {
        let nd = x.shape().len();
        let last = x.shape()[nd - 1];
        let outer: usize = x.shape()[..nd - 1].iter().product();
        let mut out = self.alloc_uninit_f16(x.numel())?;
        let bs = block_for_reduction(last);
        let cfg = LaunchConfig {
            grid_dim: (outer as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let last_i = last as i32;
        let outer_i = outer as i32;
        let mut b = self.stream.launch_builder(&self.k.rms_norm);
        b.arg(&mut out); b.arg(&x.data); b.arg(w);
        b.arg(&last_i); b.arg(&outer_i); b.arg(&eps);
        unsafe { b.launch(cfg) }?;
        Ok(GpuTensor::new(out, x.shape().to_vec()))
    }

    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let n = a.numel();
        let mut out = self.alloc_uninit_f16(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.add);
        bb.arg(&mut out); bb.arg(&a.data); bb.arg(&b.data); bb.arg(&n_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, a.shape().to_vec()))
    }

    pub fn add_inplace(&self, a: &mut GpuTensor, b: &GpuTensor) -> Result<()> {
        let n = a.numel();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_inplace);
        bb.arg(&mut a.data); bb.arg(&b.data); bb.arg(&n_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// Fused: residual += add_in (in-place), then out = rms_norm(residual, w).
    /// Saves one kernel launch vs separate `add` + `rms_norm`.
    pub fn add_residual_rms_norm(&self, residual: &mut GpuTensor, add_in: &GpuTensor, w: &CudaSlice<bf16>, eps: f32) -> Result<GpuTensor> {
        let nd = residual.shape().len();
        let last = residual.shape()[nd - 1];
        let outer: usize = residual.shape()[..nd - 1].iter().product();
        let mut out = self.alloc_uninit_f16(residual.numel())?;
        let bs = block_for_reduction(last);
        let cfg = LaunchConfig {
            grid_dim: (outer as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let last_i = last as i32; let outer_i = outer as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_residual_rms_norm);
        bb.arg(&mut residual.data); bb.arg(&mut out); bb.arg(&add_in.data); bb.arg(w);
        bb.arg(&last_i); bb.arg(&outer_i); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, residual.shape().to_vec()))
    }

    pub fn silu_mul_split(&self, gu: &GpuTensor) -> Result<GpuTensor> {
        let nd = gu.shape().len();
        let two_inter = gu.shape()[nd - 1];
        let inter = two_inter / 2;
        let outer: usize = gu.shape()[..nd - 1].iter().product();
        let mut out = self.alloc_uninit_f16(outer * inter)?;
        let cfg = LaunchConfig::for_num_elems((outer * inter) as u32);
        let outer_i = outer as i32;
        let inter_i = inter as i32;
        let mut bb = self.stream.launch_builder(&self.k.silu_mul_split);
        bb.arg(&mut out); bb.arg(&gu.data);
        bb.arg(&outer_i); bb.arg(&inter_i);
        unsafe { bb.launch(cfg) }?;
        let mut out_shape = gu.shape().to_vec();
        out_shape[nd - 1] = inter;
        Ok(GpuTensor::new(out, out_shape))
    }

    /// scores [b,h,m,n] → softmax(scale * scores) with optional causal mask. Out of place.
    pub fn softmax_scaled_causal(&self, scores: &GpuTensor, scale: f32, causal: bool) -> Result<GpuTensor> {
        let s = scores.shape();
        assert_eq!(s.len(), 4, "softmax expects [b,h,m,n]");
        let bh = s[0] * s[1];
        let m = s[2]; let n = s[3];
        let rows = bh * m;
        let bs = block_for_reduction(n);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let mut out = self.alloc_uninit_f16(scores.numel())?;
        let m_i = m as i32;
        let n_i = n as i32;
        let causal_i: i32 = if causal { 1 } else { 0 };
        let mut bb = self.stream.launch_builder(&self.k.softmax_causal);
        bb.arg(&mut out); bb.arg(&scores.data);
        bb.arg(&m_i); bb.arg(&n_i); bb.arg(&scale); bb.arg(&causal_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, s.to_vec()))
    }

    /// Q/K rotary embedding. x [b, h, s, d], cos/sin [total_s, d] (full table).
    /// pos_offset is added to each `is` to index into cos/sin.
    pub fn rotary_emb(&self, x: &GpuTensor, cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>, pos_offset: usize) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 4);
        let n = x.numel();
        let mut out = self.alloc_uninit_f16(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let s0 = s[0] as i32; let s1 = s[1] as i32; let s2 = s[2] as i32; let s3 = s[3] as i32;
        let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.rotary_emb);
        bb.arg(&mut out); bb.arg(&x.data); bb.arg(cos); bb.arg(sin);
        bb.arg(&s0); bb.arg(&s1); bb.arg(&s2); bb.arg(&s3); bb.arg(&po);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, s.to_vec()))
    }

    /// Fused per-head RMSNorm + rotary on Q or K. x [b, h, s, d].
    /// cos/sin: [total_s, d] (full table). pos_offset is added to each `is`.
    pub fn rms_norm_rotary(&self, x: &GpuTensor, w: &CudaSlice<bf16>,
                           cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
                           pos_offset: usize, eps: f32) -> Result<GpuTensor>
    {
        let s = x.shape();
        assert_eq!(s.len(), 4);
        let (b, h, sl, d) = (s[0], s[1], s[2], s[3]);
        let mut out = self.alloc_uninit_f16(x.numel())?;
        let bs = block_for_reduction(d);
        let cfg = LaunchConfig {
            grid_dim: ((b * h * sl) as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let b_i = b as i32; let h_i = h as i32; let sl_i = sl as i32; let d_i = d as i32;
        let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.rms_norm_rotary);
        bb.arg(&mut out); bb.arg(&x.data); bb.arg(w); bb.arg(cos); bb.arg(sin);
        bb.arg(&b_i); bb.arg(&h_i); bb.arg(&sl_i); bb.arg(&d_i); bb.arg(&po);
        bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, h, sl, d]))
    }

    /// Repeat-KV from a sparse KV cache, producing a dense [b, nqh, cur_len, d] view.
    pub fn repeat_kv_from_cache(&self, cache: &CudaSlice<bf16>,
        b: usize, nkvh: usize, max_seq: usize, d: usize, n_rep: usize, cur_len: usize,
    ) -> Result<GpuTensor> {
        let nqh = nkvh * n_rep;
        let total = b * nqh * cur_len * d;
        let mut out = self.alloc_uninit_f16(total)?;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let b_i = b as i32; let nkvh_i = nkvh as i32; let max_i = max_seq as i32;
        let d_i = d as i32; let nrep_i = n_rep as i32; let cur_i = cur_len as i32;
        let mut bb = self.stream.launch_builder(&self.k.repeat_kv_from_cache);
        bb.arg(&mut out); bb.arg(cache);
        bb.arg(&b_i); bb.arg(&nkvh_i); bb.arg(&max_i); bb.arg(&d_i);
        bb.arg(&nrep_i); bb.arg(&cur_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, nqh, cur_len, d]))
    }

    pub fn embed_lookup(&self, table: &GpuWeight, ids_gpu: &CudaSlice<i64>) -> Result<GpuTensor> {
        let n = ids_gpu.len();
        let d = table.cols;
        let mut out = self.alloc_uninit_f16(n * d)?;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32; let d_i = d as i32;
        let mut bb = self.stream.launch_builder(&self.k.embed_lookup);
        bb.arg(&mut out); bb.arg(&table.data); bb.arg(ids_gpu);
        bb.arg(&n_i); bb.arg(&d_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![n, d]))
    }

    pub fn argmax(&self, x: &GpuTensor) -> Result<i32> {
        let n = x.numel();
        let mut out = self.alloc_uninit_i32(1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.argmax);
        bb.arg(&mut out); bb.arg(&x.data); bb.arg(&n_i);
        unsafe { bb.launch(cfg) }?;
        let v = self.download_i32(&out)?;
        Ok(v[0])
    }

    /// Argmax that writes its result into `token_buf[slot]` (preallocated) instead of
    /// allocating a fresh i32 each call.  Pair with `download_i32(token_buf)` to get the value.
    pub fn argmax_into(&self, x: &GpuTensor, token_buf: &mut CudaSlice<i32>, slot: usize) -> Result<()> {
        let n = x.numel();
        self.argmax_into_flat(&x.data, n, token_buf, slot)
    }

    /// Argmax on a flat buffer — no GpuTensor wrapper, no D2D clone.
    /// CUDA Graph–safe.
    pub fn argmax_into_flat(&self, x: &CudaSlice<bf16>, n: usize, token_buf: &mut CudaSlice<i32>, slot: usize) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32; let slot_i = slot as i32;
        let mut bb = self.stream.launch_builder(&self.k.argmax_into_slot);
        bb.arg(token_buf); bb.arg(x); bb.arg(&n_i); bb.arg(&slot_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    pub fn swap_dims_12(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 4);
        let (d0, d1, d2, d3) = (s[0], s[1], s[2], s[3]);
        let mut out = self.alloc_uninit_f16(x.numel())?;
        let cfg = LaunchConfig::for_num_elems(x.numel() as u32);
        let d0_i = d0 as i32; let d1_i = d1 as i32; let d2_i = d2 as i32; let d3_i = d3 as i32;
        let mut bb = self.stream.launch_builder(&self.k.swap_dims_12);
        bb.arg(&mut out); bb.arg(&x.data);
        bb.arg(&d0_i); bb.arg(&d1_i); bb.arg(&d2_i); bb.arg(&d3_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![d0, d2, d1, d3]))
    }

    /// Extract one head-group from a fused QKV tensor in [b, h, s, d] layout.
    pub fn qkv_split(&self, qkv: &GpuTensor, h: usize, d: usize, offset: usize) -> Result<GpuTensor> {
        let s = qkv.shape();
        assert_eq!(s.len(), 3);
        let (b, sl, total) = (s[0], s[1], s[2]);
        let mut out = self.alloc_uninit_f16(b * h * sl * d)?;
        let cfg = LaunchConfig::for_num_elems((b * h * sl * d) as u32);
        let b_i = b as i32; let sl_i = sl as i32; let h_i = h as i32; let d_i = d as i32;
        let tot_i = total as i32; let off_i = offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.qkv_split);
        bb.arg(&mut out); bb.arg(&qkv.data);
        bb.arg(&b_i); bb.arg(&sl_i); bb.arg(&h_i); bb.arg(&d_i);
        bb.arg(&tot_i); bb.arg(&off_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, h, sl, d]))
    }

    /// Fused: extract Q from fused QKV, apply RMSNorm + rotary in one launch.
    /// qkv: [b, s, q_dim + 2*kv_dim].  Returns Q: [b, nqh, s, d].
    pub fn qkv_extract_q_norm_rotary(&self, qkv: &GpuTensor, qn_w: &CudaSlice<bf16>,
        cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
        nqh: usize, d: usize, pos_offset: usize, eps: f32,
    ) -> Result<GpuTensor> {
        let s = qkv.shape();
        assert_eq!(s.len(), 3);
        let (b, sl, total_cols) = (s[0], s[1], s[2]);
        let mut q_out = self.alloc_uninit_f16(b * nqh * sl * d)?;
        let bs = block_for_reduction(d);
        let cfg = LaunchConfig {
            grid_dim: ((b * nqh * sl) as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let b_i = b as i32; let nqh_i = nqh as i32; let sl_i = sl as i32;
        let d_i = d as i32; let tot_i = total_cols as i32; let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.qkv_extract_q_norm_rotary);
        bb.arg(&mut q_out); bb.arg(&qkv.data); bb.arg(qn_w); bb.arg(cos); bb.arg(sin);
        bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&sl_i); bb.arg(&d_i);
        bb.arg(&tot_i); bb.arg(&po); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(q_out, vec![b, nqh, sl, d]))
    }

    /// Fused: extract K (with RMSNorm+rotary) and V (raw) from fused QKV, write both into KV cache.
    /// Replaces qkv_split×2 + rms_norm_rotary + kv_cache_write_pair (4 launches → 1).
    pub fn qkv_extract_kv_norm_rotary_cache(&self,
        k_cache: &mut CudaSlice<bf16>, v_cache: &mut CudaSlice<bf16>,
        qkv: &GpuTensor, kn_w: &CudaSlice<bf16>,
        cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
        nkvh: usize, d: usize, q_dim: usize, kv_dim: usize,
        max_seq: usize, start: usize, pos_offset: usize, eps: f32,
    ) -> Result<()> {
        let s = qkv.shape();
        assert_eq!(s.len(), 3);
        let (b, sl, total_cols) = (s[0], s[1], s[2]);
        let bs = block_for_reduction(d);
        let cfg = LaunchConfig {
            grid_dim: ((b * nkvh * sl) as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let b_i = b as i32; let nkvh_i = nkvh as i32; let sl_i = sl as i32;
        let d_i = d as i32; let tot_i = total_cols as i32;
        let q_i = q_dim as i32; let kv_i = kv_dim as i32;
        let max_i = max_seq as i32; let start_i = start as i32; let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.qkv_extract_kv_norm_rotary_cache);
        bb.arg(k_cache); bb.arg(v_cache); bb.arg(&qkv.data); bb.arg(kn_w);
        bb.arg(cos); bb.arg(sin);
        bb.arg(&b_i); bb.arg(&nkvh_i); bb.arg(&sl_i); bb.arg(&d_i); bb.arg(&tot_i);
        bb.arg(&q_i); bb.arg(&kv_i);
        bb.arg(&max_i); bb.arg(&start_i); bb.arg(&po); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// Single-kernel fused QKV extraction: builds Q, K (with norm+rotary) and V (raw),
    /// writes Q to a fresh output and K/V into their respective caches.
    /// Replaces qkv_extract_q_norm_rotary + qkv_extract_kv_norm_rotary_cache (2 launches → 1).
    pub fn qkv_extract_qkv_norm_rotary_cache(&self,
        k_cache: &mut CudaSlice<bf16>, v_cache: &mut CudaSlice<bf16>,
        qkv: &GpuTensor, qn_w: &CudaSlice<bf16>, kn_w: &CudaSlice<bf16>,
        cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
        nqh: usize, nkvh: usize, d: usize, q_dim: usize, kv_dim: usize,
        max_seq: usize, start: usize, pos_offset: usize, eps: f32,
    ) -> Result<GpuTensor> {
        let s = qkv.shape();
        assert_eq!(s.len(), 3);
        let (b, sl, total_cols) = (s[0], s[1], s[2]);
        let mut q_out = self.alloc_uninit_f16(b * nqh * sl * d)?;
        let bs = block_for_reduction(d);
        let cfg = LaunchConfig {
            grid_dim: ((b * sl) as u32, (nqh + nkvh) as u32, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let b_i = b as i32; let nqh_i = nqh as i32; let nkvh_i = nkvh as i32;
        let sl_i = sl as i32; let d_i = d as i32; let tot_i = total_cols as i32;
        let q_i = q_dim as i32; let kv_i = kv_dim as i32;
        let max_i = max_seq as i32; let start_i = start as i32; let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.qkv_extract_qkv_norm_rotary_cache);
        bb.arg(&mut q_out); bb.arg(k_cache); bb.arg(v_cache); bb.arg(&qkv.data);
        bb.arg(qn_w); bb.arg(kn_w); bb.arg(cos); bb.arg(sin);
        bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&nkvh_i); bb.arg(&sl_i); bb.arg(&d_i); bb.arg(&tot_i);
        bb.arg(&q_i); bb.arg(&kv_i);
        bb.arg(&max_i); bb.arg(&start_i); bb.arg(&po); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(q_out, vec![b, nqh, sl, d]))
    }

    /// Write a [b, nkvh, s_new, d] tensor into a [b, nkvh, max_seq, d] cache at offset `start`.
    pub fn kv_cache_write(&self, cache: &mut CudaSlice<bf16>, k_new: &GpuTensor,
        b: usize, nkvh: usize, max_seq: usize, d: usize, start: usize,
    ) -> Result<()> {
        let s_new = k_new.shape()[2];
        let total = b * nkvh * s_new * d;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let b_i = b as i32; let nkvh_i = nkvh as i32; let max_i = max_seq as i32;
        let d_i = d as i32; let start_i = start as i32; let snew_i = s_new as i32;
        let mut bb = self.stream.launch_builder(&self.k.kv_cache_write);
        bb.arg(cache); bb.arg(&k_new.data);
        bb.arg(&b_i); bb.arg(&nkvh_i); bb.arg(&max_i); bb.arg(&d_i);
        bb.arg(&start_i); bb.arg(&snew_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// Fused: write K and V into their caches in one kernel.
    pub fn kv_cache_write_pair(&self, k_cache: &mut CudaSlice<bf16>, v_cache: &mut CudaSlice<bf16>,
        k_new: &GpuTensor, v_new: &GpuTensor,
        b: usize, nkvh: usize, max_seq: usize, d: usize, start: usize,
    ) -> Result<()> {
        let s_new = k_new.shape()[2];
        let total = b * nkvh * s_new * d;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let b_i = b as i32; let nkvh_i = nkvh as i32; let max_i = max_seq as i32;
        let d_i = d as i32; let start_i = start as i32; let snew_i = s_new as i32;
        let mut bb = self.stream.launch_builder(&self.k.kv_cache_write_pair);
        bb.arg(k_cache); bb.arg(v_cache); bb.arg(&k_new.data); bb.arg(&v_new.data);
        bb.arg(&b_i); bb.arg(&nkvh_i); bb.arg(&max_i); bb.arg(&d_i);
        bb.arg(&start_i); bb.arg(&snew_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    pub fn gelu_inplace(&self, x: &mut GpuTensor) -> Result<()> {
        let n = x.numel();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.gelu_inplace);
        bb.arg(&mut x.data); bb.arg(&n_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    pub fn layer_norm(&self, x: &GpuTensor, w: &CudaSlice<bf16>, bias: &CudaSlice<bf16>, eps: f32) -> Result<GpuTensor> {
        let nd = x.shape().len();
        let last = x.shape()[nd - 1];
        let outer: usize = x.shape()[..nd - 1].iter().product();
        let mut out = self.alloc_uninit_f16(x.numel())?;
        let bs = block_for_reduction(last);
        let cfg = LaunchConfig {
            grid_dim: (outer as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4 * 2,  // sum + sum_sq
        };
        let last_i = last as i32; let outer_i = outer as i32;
        let mut bb = self.stream.launch_builder(&self.k.layer_norm);
        bb.arg(&mut out); bb.arg(&x.data); bb.arg(w); bb.arg(bias);
        bb.arg(&last_i); bb.arg(&outer_i); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, x.shape().to_vec()))
    }

    pub fn add_bias_inplace(&self, x: &mut GpuTensor, bias: &CudaSlice<bf16>) -> Result<()> {
        let nd = x.shape().len();
        let last = x.shape()[nd - 1];
        let outer: usize = x.shape()[..nd - 1].iter().product();
        let cfg = LaunchConfig::for_num_elems((outer * last) as u32);
        let outer_i = outer as i32; let last_i = last as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_bias_inplace);
        bb.arg(&mut x.data); bb.arg(bias);
        bb.arg(&outer_i); bb.arg(&last_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// Slice [b, h, s, d] → [b, h, len, d] taking rows `start..start+len`.
    pub fn slice_dim2(&self, x: &GpuTensor, start: usize, len: usize) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 4);
        let (b, h, sl, d) = (s[0], s[1], s[2], s[3]);
        assert!(start + len <= sl);
        let total = b * h * len * d;
        let mut out = self.alloc_uninit_f16(total)?;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let b_i = b as i32; let h_i = h as i32; let sl_i = sl as i32;
        let d_i = d as i32; let start_i = start as i32; let len_i = len as i32;
        let mut bb = self.stream.launch_builder(&self.k.slice_dim2);
        bb.arg(&mut out); bb.arg(&x.data);
        bb.arg(&b_i); bb.arg(&h_i); bb.arg(&sl_i); bb.arg(&d_i);
        bb.arg(&start_i); bb.arg(&len_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, h, len, d]))
    }

    /// Write a [b, h, len, d] chunk into a pre-allocated [b, h, s, d] buffer at `dst_offset`.
    pub fn concat_dim2_write(&self, dst: &mut CudaSlice<bf16>, src: &GpuTensor,
        b: usize, h: usize, s: usize, d: usize, dst_offset: usize,
    ) -> Result<()> {
        let len = src.shape()[2];
        let total = b * h * len * d;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let b_i = b as i32; let h_i = h as i32; let s_i = s as i32;
        let d_i = d as i32; let off_i = dst_offset as i32; let len_i = len as i32;
        let mut bb = self.stream.launch_builder(&self.k.concat_dim2_write);
        bb.arg(dst); bb.arg(&src.data);
        bb.arg(&b_i); bb.arg(&h_i); bb.arg(&s_i); bb.arg(&d_i);
        bb.arg(&off_i); bb.arg(&len_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// 3×3 conv2d with stride=2, padding=1 — via im2col + cuBLAS GEMM + fused bias/GELU.
    /// input: [b, c_in, h, w]; weight: [c_out, c_in, 3, 3]; bias: [c_out].
    /// Returns [b, c_out, h_out, w_out] with GELU applied.
    pub fn conv2d_3x3_s2p1_gelu(&self, x: &GpuTensor, weight: &CudaSlice<bf16>,
        c_out: usize, c_in: usize, bias: &CudaSlice<bf16>,
    ) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 4);
        let (b, c_in_chk, h, w) = (s[0], s[1], s[2], s[3]);
        assert_eq!(c_in_chk, c_in);
        let h_out = (h + 2 - 3) / 2 + 1;
        let w_out = (w + 2 - 3) / 2 + 1;

        // im2col: [b*h_out*w_out, c_in*9] (row-major)
        let m = b * h_out * w_out;
        let k = c_in * 9;
        let mut col = self.alloc_uninit_f16(m * k)?;
        let cfg = LaunchConfig::for_num_elems((m * k) as u32);
        let b_i = b as i32; let cin_i = c_in as i32;
        let h_i = h as i32; let w_i = w as i32;
        let ho_i = h_out as i32; let wo_i = w_out as i32;
        let mut bb = self.stream.launch_builder(&self.k.im2col_3x3);
        bb.arg(&mut col); bb.arg(&x.data);
        bb.arg(&b_i); bb.arg(&cin_i); bb.arg(&h_i); bb.arg(&w_i);
        bb.arg(&ho_i); bb.arg(&wo_i);
        unsafe { bb.launch(cfg) }?;

        // GEMM: gemm_out[m, c_out] = col[m, k] @ weight^T[k, c_out]
        // We need weight reinterpreted as [c_out, c_in*9] (already that layout since
        // weight is stored as [c_out, c_in, 3, 3] row-major = [c_out, c_in*9]).
        let n = c_out;
        let mut gemm_out = self.alloc_uninit_f16(m * n)?;
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,  // weight: K-major in row-major = T in col-major
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32, n: m as i32, k: k as i32,
                    alpha: bf16::from_f32(1.0),
                    lda: k as i32, ldb: k as i32,
                    beta: bf16::from_f32(0.0), ldc: n as i32,
                },
                weight, &col, &mut gemm_out,
            )?;
        }
        drop(col);

        // Post: [b*h_out*w_out, c_out] → [b, c_out, h_out, w_out] with bias+GELU
        let total = b * c_out * h_out * w_out;
        let mut out = self.alloc_uninit_f16(total)?;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let cout_i = c_out as i32;
        let mut bb = self.stream.launch_builder(&self.k.conv_postprocess);
        bb.arg(&mut out); bb.arg(&gemm_out); bb.arg(bias);
        bb.arg(&b_i); bb.arg(&cout_i); bb.arg(&ho_i); bb.arg(&wo_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, c_out, h_out, w_out]))
    }

    /// Conv1d (kernel=3, padding=1, configurable stride) + GELU for the Whisper stem.
    /// x: [b, in_ch, in_t], weight: [out_ch, in_ch, 3], bias: [out_ch].
    /// Returns [b, out_ch, out_t] with GELU applied.
    pub fn conv1d_k3_gelu(&self, x: &GpuTensor, weight: &CudaSlice<bf16>, bias: &CudaSlice<bf16>,
                          in_ch: usize, out_ch: usize, stride: usize) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 3);
        let (b, in_ch_chk, in_t) = (s[0], s[1], s[2]);
        assert_eq!(in_ch_chk, in_ch);
        let out_t = (in_t + 2 - 3) / stride + 1;
        let total = b * out_ch * out_t;
        let mut out = self.alloc_uninit_f16(total)?;
        // Explicit launch config: 256 threads/block, grid covers all elements.
        let cfg = LaunchConfig {
            grid_dim: (((total as u32) + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let b_i = b as i32; let in_ch_i = in_ch as i32; let in_t_i = in_t as i32;
        let out_ch_i = out_ch as i32; let out_t_i = out_t as i32; let stride_i = stride as i32;
        let mut bb = self.stream.launch_builder(&self.k.conv1d_k3_gelu);
        bb.arg(&mut out); bb.arg(&x.data); bb.arg(weight); bb.arg(bias);
        bb.arg(&b_i); bb.arg(&in_ch_i); bb.arg(&in_t_i); bb.arg(&out_ch_i);
        bb.arg(&out_t_i); bb.arg(&stride_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, out_ch, out_t]))
    }

    /// Permute [b, c, t] → [b, t, c] (Whisper stem → transformer layout).
    pub fn permute_bct_to_btc(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 3);
        let (b, c, t) = (s[0], s[1], s[2]);
        let total = b * c * t;
        let mut out = self.alloc_uninit_f16(total)?;
        let cfg = LaunchConfig {
            grid_dim: (((total as u32) + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let b_i = b as i32; let c_i = c as i32; let t_i = t as i32;
        let mut bb = self.stream.launch_builder(&self.k.permute_bct_to_btc);
        bb.arg(&mut out); bb.arg(&x.data);
        bb.arg(&b_i); bb.arg(&c_i); bb.arg(&t_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, t, c]))
    }

    /// Broadcast positional-embedding add: out[b,t,d] = x[b,t,d] + pe[t,d].
    pub fn add_pe(&self, x: &GpuTensor, pe: &CudaSlice<bf16>, t: usize) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 3);
        let (b, t_chk, d) = (s[0], s[1], s[2]);
        assert_eq!(t_chk, t);
        let bt = b * t;
        let mut out = self.alloc_uninit_f16(b * t * d)?;
        // add_pe_f16 is __launch_bounds__(512): block must be ≤ 512.
        let block = (d as u32).min(512);
        let cfg = LaunchConfig { grid_dim: (bt as u32, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
        let d_i = d as i32; let t_i = t as i32; let bt_i = bt as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_pe);
        bb.arg(&mut out); bb.arg(&x.data); bb.arg(pe);
        bb.arg(&d_i); bb.arg(&t_i); bb.arg(&bt_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, x.shape().to_vec()))
    }

    /// Fused GQA attention for decode (s_q = 1).  Replaces repeat_kv + attention_qk + softmax + attention_av
    /// (4 launches + 2 big allocs) with a single kernel that reads K/V directly from the cache.
    /// Q: [b, nqh, 1, d].  Returns out: [b, nqh, 1, d].
    pub fn fused_gqa_decode(&self, q: &GpuTensor,
        k_cache: &CudaSlice<bf16>, v_cache: &CudaSlice<bf16>,
        nkvh: usize, max_seq: usize, cur_len: usize, scale: f32,
    ) -> Result<GpuTensor> {
        let s = q.shape();
        assert_eq!(s.len(), 4);
        assert_eq!(s[2], 1, "fused_gqa_decode requires s_q = 1");
        let (b, nqh, _, d) = (s[0], s[1], s[2], s[3]);
        let mut out = self.alloc_uninit_f16(b * nqh * d)?;
        // Adaptive block size: scale parallelism with cur_len so each thread does roughly the
        // same amount of work whether we have 200 or 2000 tokens of context.  bs must be a
        // multiple of d (so the stage-4 t-chunks layout works) and a power of 2 (for reductions).
        let bs: u32 = if cur_len > 1024 { 1024 }
                      else if cur_len > 512 { 512 }
                      else { 256 };
        let t_chunks = (bs as usize / d).max(1);
        // shared: scores[cur_len] + partial_out[d * t_chunks], both f32.
        let smem_bytes = (cur_len + d * t_chunks) * 4;
        let cfg = LaunchConfig {
            grid_dim: ((b * nqh) as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: smem_bytes as u32,
        };
        let b_i = b as i32; let nqh_i = nqh as i32; let nkvh_i = nkvh as i32;
        let max_i = max_seq as i32; let d_i = d as i32; let cur_i = cur_len as i32;
        let mut bb = self.stream.launch_builder(&self.k.fused_gqa_decode);
        bb.arg(&mut out); bb.arg(&q.data); bb.arg(k_cache); bb.arg(v_cache);
        bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&nkvh_i); bb.arg(&max_i);
        bb.arg(&d_i); bb.arg(&cur_i); bb.arg(&scale);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, nqh, 1, d]))
    }

    /// Split-K variant of `fused_gqa_decode`: divides the cur_len axis into N chunks across
    /// independent blocks, then merges with an online-softmax correction kernel.  Use when
    /// cur_len is large enough that the single-block kernel underutilizes the SMs.
    pub fn fused_gqa_decode_split(&self, q: &GpuTensor,
        k_cache: &CudaSlice<bf16>, v_cache: &CudaSlice<bf16>,
        nkvh: usize, max_seq: usize, cur_len: usize, scale: f32,
        chunk_size: usize,
    ) -> Result<GpuTensor> {
        let s = q.shape();
        let (b, nqh, _, d) = (s[0], s[1], s[2], s[3]);
        let n_chunks = (cur_len + chunk_size - 1) / chunk_size;

        // Buffers: partial out [b, nqh, n_chunks, d] f32, max/sum [b, nqh, n_chunks] f32.
        let part_out = self.stream.alloc_zeros::<f32>(b * nqh * n_chunks * d)?;
        let part_max = self.stream.alloc_zeros::<f32>(b * nqh * n_chunks)?;
        let part_sum = self.stream.alloc_zeros::<f32>(b * nqh * n_chunks)?;

        // Phase 1: per-chunk partial computation.
        let bs: u32 = 256;
        let t_split = (bs as usize / d).max(1);
        let smem_bytes = (chunk_size + d * t_split) * 4;
        let mut part_out_buf = part_out;
        let mut part_max_buf = part_max;
        let mut part_sum_buf = part_sum;
        {
            let cfg = LaunchConfig {
                grid_dim: ((b * nqh) as u32, n_chunks as u32, 1),
                block_dim: (bs, 1, 1),
                shared_mem_bytes: smem_bytes as u32,
            };
            let b_i = b as i32; let nqh_i = nqh as i32; let nkvh_i = nkvh as i32;
            let max_i = max_seq as i32; let d_i = d as i32;
            let cur_i = cur_len as i32; let cs_i = chunk_size as i32; let nc_i = n_chunks as i32;
            let mut bb = self.stream.launch_builder(&self.k.fused_gqa_decode_split_p1);
            bb.arg(&mut part_out_buf); bb.arg(&mut part_max_buf); bb.arg(&mut part_sum_buf);
            bb.arg(&q.data); bb.arg(k_cache); bb.arg(v_cache);
            bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&nkvh_i); bb.arg(&max_i); bb.arg(&d_i);
            bb.arg(&cur_i); bb.arg(&cs_i); bb.arg(&nc_i); bb.arg(&scale);
            unsafe { bb.launch(cfg) }?;
        }

        // Phase 2: merge chunks → final output.
        let mut out = self.alloc_uninit_f16(b * nqh * d)?;
        {
            let cfg = LaunchConfig {
                grid_dim: ((b * nqh) as u32, 1, 1),
                block_dim: (d as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let b_i = b as i32; let nqh_i = nqh as i32; let nc_i = n_chunks as i32; let d_i = d as i32;
            let mut bb = self.stream.launch_builder(&self.k.fused_gqa_decode_split_p2);
            bb.arg(&mut out); bb.arg(&part_out_buf); bb.arg(&part_max_buf); bb.arg(&part_sum_buf);
            bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&nc_i); bb.arg(&d_i);
            unsafe { bb.launch(cfg) }?;
        }
        Ok(GpuTensor::new(out, vec![b, nqh, 1, d]))
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  GpuWeight — 2-D weight matrix on GPU
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct GpuWeight {
    pub data: CudaSlice<bf16>,
    pub rows: usize,
    pub cols: usize,
}

// ═══════════════════════════════════════════════════════════════════════
//  KV cache (GPU-resident, pre-allocated)
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub(crate) struct GpuKvCache {
    pub k: Vec<CudaSlice<bf16>>,   // per layer: [b, nkvh, max_seq, d]
    pub v: Vec<CudaSlice<bf16>>,
    pub cur_len: usize,
    pub max_seq: usize,
    pub b: usize,
    pub nkvh: usize,
    pub d: usize,
}

impl GpuKvCache {
    pub fn new(cuda: &CudaState, num_layers: usize, b: usize, nkvh: usize, max_seq: usize, d: usize) -> Result<Self> {
        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k.push(cuda.alloc_zeros_f16(b * nkvh * max_seq * d)?);
            v.push(cuda.alloc_zeros_f16(b * nkvh * max_seq * d)?);
        }
        Ok(Self { k, v, cur_len: 0, max_seq, b, nkvh, d })
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Weight loading helpers
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
//  Weight loading helpers (pub(crate) so sibling modules can call directly)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn load_gpu_weight(cuda: &CudaState, weights: &HashMap<String, RawTensor>, name: &str) -> Result<GpuWeight> {
    let (data_f16, shape) = get_weight_f16(weights, name)?;
    assert_eq!(shape.len(), 2, "weight {} should be 2D", name);
    let dev = cuda.upload_f16(&data_f16)?;
    Ok(GpuWeight { data: dev, rows: shape[0], cols: shape[1] })
}

pub(crate) fn load_gpu_vec(cuda: &CudaState, weights: &HashMap<String, RawTensor>, name: &str) -> Result<CudaSlice<bf16>> {
    let (data_f16, _shape) = get_weight_f16(weights, name)?;
    cuda.upload_f16(&data_f16)
}

fn get_weight_f16(weights: &HashMap<String, RawTensor>, name: &str) -> Result<(Vec<bf16>, Vec<usize>)> {
    let td = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?;
    let (data_bf16, shape) = td.as_bf16()?;
    Ok((data_bf16, shape))
}

fn load_fused_qkv_weight(
    weights: &HashMap<String, RawTensor>, prefix: &str, cuda: &CudaState,
) -> Result<(GpuWeight, usize, usize)> {
    let (qw, qs) = get_weight_f16(weights, &format!("{}.q_proj.weight", prefix))?;
    let (kw, ks) = get_weight_f16(weights, &format!("{}.k_proj.weight", prefix))?;
    let (vw, vs) = get_weight_f16(weights, &format!("{}.v_proj.weight", prefix))?;
    let q_dim = qs[0]; let kv_dim = ks[0]; let hidden = qs[1];
    assert_eq!(ks[1], hidden); assert_eq!(vs[1], hidden);
    let mut fused = Vec::with_capacity((q_dim + 2 * kv_dim) * hidden);
    fused.extend_from_slice(&qw); fused.extend_from_slice(&kw); fused.extend_from_slice(&vw);
    let total_rows = q_dim + 2 * kv_dim;
    let dev = cuda.upload_f16(&fused)?;
    Ok((GpuWeight { data: dev, rows: total_rows, cols: hidden }, q_dim, kv_dim))
}

fn load_fused_gate_up_weight(
    weights: &HashMap<String, RawTensor>, prefix: &str, cuda: &CudaState,
) -> Result<(GpuWeight, usize)> {
    let (gw, gs) = get_weight_f16(weights, &format!("{}.gate_proj.weight", prefix))?;
    let (uw, us) = get_weight_f16(weights, &format!("{}.up_proj.weight", prefix))?;
    let intermediate = gs[0]; let hidden = gs[1];
    assert_eq!(us[0], intermediate); assert_eq!(us[1], hidden);
    let mut fused = Vec::with_capacity(2 * intermediate * hidden);
    fused.extend_from_slice(&gw); fused.extend_from_slice(&uw);
    let dev = cuda.upload_f16(&fused)?;
    Ok((GpuWeight { data: dev, rows: 2 * intermediate, cols: hidden }, intermediate))
}

// ═══════════════════════════════════════════════════════════════════════
//  DecodeScratch — pre-allocated temp buffers for the decode hot loop
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub(crate) struct DecodeScratch {
    // Per-layer temporaries (reused across layers since they run sequentially)
    pub norm1: CudaSlice<bf16>,           // [hidden_size]
    pub qkv: CudaSlice<bf16>,            // [fused_qkv_dim] = nqh*hd + 2*nkvh*hd = 4096
    pub q_out: CudaSlice<bf16>,          // [nqh * hd] = 2048
    pub attn_out: CudaSlice<bf16>,       // [nqh * hd] = 2048
    pub norm2: CudaSlice<bf16>,          // [hidden_size]
    pub gate_up: CudaSlice<bf16>,        // [2 * intermediate_size] = 6144
    pub activated: CudaSlice<bf16>,      // [intermediate_size] = 3072

    // Split attention partials — pre-sized for max_chunks
    pub split_part_out: CudaSlice<f32>, // [nqh * max_chunks * hd]
    pub split_part_max: CudaSlice<f32>, // [nqh * max_chunks]
    pub split_part_sum: CudaSlice<f32>, // [nqh * max_chunks]

    // Decoder-level temporaries
    pub embed_out: CudaSlice<bf16>,      // [hidden_size]
    pub final_norm: CudaSlice<bf16>,     // [hidden_size]
    pub logits: CudaSlice<bf16>,         // [vocab_size]

    // Pinned host buffer for async D2H of the decoded token.
    pub pinned_token: PinnedHostSlice<i32>,

    // Static dimensions (cached to avoid re-computing)
    pub hs: usize,     // hidden_size
    pub nqh: usize,    // num_attention_heads
    pub nkvh: usize,   // num_key_value_heads
    pub hd: usize,     // head_dim
    pub inter: usize,  // intermediate_size
    pub vocab: usize,  // vocab_size
}

impl DecodeScratch {
    pub fn new(cuda: &CudaState, max_seq: usize, cfg: &TextDecoderConfig) -> Result<Self> {
        let hs = cfg.hidden_size;
        let nqh = cfg.num_attention_heads;
        let nkvh = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;

        let fused_qkv = (nqh + 2 * nkvh) * hd; // Q + K + V concatenated
        // Use chunk_size=256 for max_chunks upper bound (conservative)
        let max_chunks = (max_seq + 255) / 256;

        Ok(Self {
            norm1: cuda.alloc_uninit_f16(hs)?,
            qkv: cuda.alloc_uninit_f16(fused_qkv)?,
            q_out: cuda.alloc_uninit_f16(nqh * hd)?,
            attn_out: cuda.alloc_uninit_f16(nqh * hd)?,
            norm2: cuda.alloc_uninit_f16(hs)?,
            gate_up: cuda.alloc_uninit_f16(2 * inter)?,
            activated: cuda.alloc_uninit_f16(inter)?,
            split_part_out: cuda.stream.alloc_zeros::<f32>(nqh * max_chunks * hd)?,
            split_part_max: cuda.stream.alloc_zeros::<f32>(nqh * max_chunks)?,
            split_part_sum: cuda.stream.alloc_zeros::<f32>(nqh * max_chunks)?,
            embed_out: cuda.alloc_uninit_f16(hs)?,
            final_norm: cuda.alloc_uninit_f16(hs)?,
            logits: cuda.alloc_uninit_f16(vocab)?,
            pinned_token: unsafe { cuda.ctx.alloc_pinned::<i32>(1)? },
            hs, nqh, nkvh, hd, inter, vocab,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  _into variants — same as originals but write into pre-allocated buffers
// ═══════════════════════════════════════════════════════════════════════

impl CudaState {
    /// rms_norm on a flat buffer — no GpuTensor wrapper, no D2D clone.
    /// CUDA Graph–safe (no cuMemcpyDtD).
    pub fn rms_norm_into_flat(&self, x: &CudaSlice<bf16>, last: usize, outer: usize, w: &CudaSlice<bf16>, eps: f32, out: &mut CudaSlice<bf16>) -> Result<()> {
        let bs = block_for_reduction(last);
        let cfg = LaunchConfig { grid_dim: (outer as u32, 1, 1), block_dim: (bs, 1, 1), shared_mem_bytes: bs * 4 };
        let last_i = last as i32; let outer_i = outer as i32;
        let mut b = self.stream.launch_builder(&self.k.rms_norm);
        b.arg(out); b.arg(x); b.arg(w); b.arg(&last_i); b.arg(&outer_i); b.arg(&eps);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// linear_gpu on a flat buffer — no GpuTensor wrapper, no D2D clone.
    /// CUDA Graph–safe (no cuMemcpyDtD).
    pub fn linear_gpu_into_flat(&self, x: &CudaSlice<bf16>, m: usize, k: usize, w: &GpuWeight, out: &mut CudaSlice<bf16>) -> Result<()> {
        let n = w.rows;
        assert_eq!(k, w.cols);
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32, n: m as i32, k: k as i32,
                    alpha: bf16::from_f32(1.0),
                    lda: k as i32, ldb: k as i32,
                    beta: bf16::from_f32(0.0), ldc: n as i32,
                },
                &w.data, x, out,
            )?;
        }
        Ok(())
    }

    /// silu_mul_split writing into a pre-allocated `out` buffer.  `out` must have `outer * inter` bf16 elements.
    pub fn silu_mul_split_into(&self, gu: &CudaSlice<bf16>, outer: usize, two_inter: usize, out: &mut CudaSlice<bf16>) -> Result<()> {
        let inter = two_inter / 2;
        let total = outer * inter;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let outer_i = outer as i32; let inter_i = inter as i32;
        let mut bb = self.stream.launch_builder(&self.k.silu_mul_split);
        bb.arg(out); bb.arg(gu); bb.arg(&outer_i); bb.arg(&inter_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// qkv_extract writing Q into pre-allocated `q_out` buffer.
    pub fn qkv_extract_into(&self,
        k_cache: &mut CudaSlice<bf16>, v_cache: &mut CudaSlice<bf16>,
        qkv: &CudaSlice<bf16>, qn_w: &CudaSlice<bf16>, kn_w: &CudaSlice<bf16>,
        cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
        nqh: usize, nkvh: usize, d: usize, q_dim: usize, kv_dim: usize,
        max_seq: usize, start: usize, pos_offset: usize, eps: f32,
        q_out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let b: usize = 1; let sl: usize = 1;
        let total_cols = q_dim + 2 * kv_dim;
        let bs = block_for_reduction(d);
        let cfg = LaunchConfig { grid_dim: ((b * sl) as u32, (nqh + nkvh) as u32, 1), block_dim: (bs, 1, 1), shared_mem_bytes: bs * 4 };
        let b_i = b as i32; let nqh_i = nqh as i32; let nkvh_i = nkvh as i32;
        let sl_i = sl as i32; let d_i = d as i32; let tot_i = total_cols as i32;
        let q_i = q_dim as i32; let kv_i = kv_dim as i32;
        let max_i = max_seq as i32; let start_i = start as i32; let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.qkv_extract_qkv_norm_rotary_cache);
        bb.arg(q_out); bb.arg(k_cache); bb.arg(v_cache); bb.arg(qkv);
        bb.arg(qn_w); bb.arg(kn_w); bb.arg(cos); bb.arg(sin);
        bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&nkvh_i); bb.arg(&sl_i); bb.arg(&d_i); bb.arg(&tot_i);
        bb.arg(&q_i); bb.arg(&kv_i);
        bb.arg(&max_i); bb.arg(&start_i); bb.arg(&po); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// fused_gqa_decode writing into a pre-allocated `out` buffer (flat, no GpuTensor).
    /// Single-kernel path for short-to-medium context (cur_len ≤ ~1024). Saves 1 launch
    /// vs the split path (p1+p2). Used by decode when cur_len is below the split threshold.
    pub fn fused_gqa_decode_into(&self, q: &CudaSlice<bf16>,
        k_cache: &CudaSlice<bf16>, v_cache: &CudaSlice<bf16>,
        nqh: usize, nkvh: usize, max_seq: usize, cur_len: usize, scale: f32,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let b: usize = 1;
        let d: usize = out.len() / nqh;  // nqh * d = out.len()
        // Adaptive block size: scale with cur_len. bs must be multiple of d, power of 2.
        let bs: u32 = if cur_len > 1024 { 1024 }
                      else if cur_len > 512 { 512 }
                      else { 256 };
        let t_chunks = (bs as usize / d).max(1);
        let smem_bytes = (cur_len + d * t_chunks) * 4;
        let cfg = LaunchConfig {
            grid_dim: ((b * nqh) as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: smem_bytes as u32,
        };
        let b_i = b as i32; let nqh_i = nqh as i32; let nkvh_i = nkvh as i32;
        let max_i = max_seq as i32; let d_i = d as i32; let cur_i = cur_len as i32;
        let mut bb = self.stream.launch_builder(&self.k.fused_gqa_decode);
        bb.arg(out); bb.arg(q); bb.arg(k_cache); bb.arg(v_cache);
        bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&nkvh_i); bb.arg(&max_i);
        bb.arg(&d_i); bb.arg(&cur_i); bb.arg(&scale);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// fused_gqa_decode_split writing into pre-allocated partial + output buffers.
    /// `max_chunks` is the fixed grid y-dimension (upper bound for graph stability).
    pub fn fused_gqa_decode_split_into(&self, q: &CudaSlice<bf16>,
        k_cache: &CudaSlice<bf16>, v_cache: &CudaSlice<bf16>,
        nkvh: usize, max_seq: usize, cur_len: usize, scale: f32, chunk_size: usize, max_chunks: usize,
        part_out: &mut CudaSlice<f32>, part_max: &mut CudaSlice<f32>, part_sum: &mut CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let n_chunks = (cur_len + chunk_size - 1) / chunk_size;
        let bs: u32 = 256;
        let t_split = (bs as usize / 128).max(1);
        let smem_bytes = (chunk_size + 128 * t_split) * 4;
        {
            // Grid y-dim uses max_chunks for stable topology; kernel handles excess chunks
            let cfg = LaunchConfig { grid_dim: (16, max_chunks as u32, 1), block_dim: (bs, 1, 1), shared_mem_bytes: smem_bytes as u32 };
            let nkvh_i = nkvh as i32; let max_i = max_seq as i32; let d_i = 128i32;
            let cur_i = cur_len as i32; let cs_i = chunk_size as i32; let nc_i = n_chunks as i32;
            let mut bb = self.stream.launch_builder(&self.k.fused_gqa_decode_split_p1);
            bb.arg(&mut *part_out); bb.arg(&mut *part_max); bb.arg(&mut *part_sum);
            bb.arg(q); bb.arg(k_cache); bb.arg(v_cache);
            bb.arg(&1i32); bb.arg(&16i32); bb.arg(&nkvh_i); bb.arg(&max_i); bb.arg(&d_i);
            bb.arg(&cur_i); bb.arg(&cs_i); bb.arg(&nc_i); bb.arg(&scale);
            unsafe { bb.launch(cfg) }?;
        }
        {
            let cfg = LaunchConfig { grid_dim: (16, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            let nc_i = n_chunks as i32;
            let mut bb = self.stream.launch_builder(&self.k.fused_gqa_decode_split_p2);
            bb.arg(out); bb.arg(&*part_out); bb.arg(&*part_max); bb.arg(&*part_sum);
            bb.arg(&1i32); bb.arg(&16i32); bb.arg(&nc_i); bb.arg(&128i32);
            unsafe { bb.launch(cfg) }?;
        }
        Ok(())
    }

    /// embed_id_from_gpu_slot writing into a pre-allocated `out` buffer.
    pub fn embed_id_from_gpu_slot_into(&self, table: &GpuWeight, token_buf: &CudaSlice<i32>, slot: usize, out: &mut CudaSlice<bf16>) -> Result<()> {
        let d = table.cols;
        let bs = (d as u32).min(1024);
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (bs, 1, 1), shared_mem_bytes: 0 };
        let slot_i = slot as i32; let d_i = d as i32;
        let mut bb = self.stream.launch_builder(&self.k.embed_lookup_single_i32);
        bb.arg(out); bb.arg(&table.data); bb.arg(token_buf); bb.arg(&slot_i); bb.arg(&d_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// linear_gpu_accum on raw CudaSlice (for decode scratch path).
    /// y = y + x @ W^T.  x has shape [1, 1, k], W is [n, k], y is [n].
    pub fn linear_gpu_accum_slice(&self, y: &mut CudaSlice<bf16>, x: &CudaSlice<bf16>, w: &GpuWeight) -> Result<()> {
        let k = w.cols;
        let n = w.rows;
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32, n: 1, k: k as i32,
                    alpha: bf16::from_f32(1.0),
                    lda: k as i32, ldb: k as i32,
                    beta: bf16::from_f32(1.0), ldc: n as i32,
                },
                &w.data, x, y,
            )?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Decoder Layer (GPU-resident)
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
struct GpuDecoderLayer {
    iln_w: CudaSlice<bf16>,
    pln_w: CudaSlice<bf16>,
    qn_w: CudaSlice<bf16>,
    kn_w: CudaSlice<bf16>,
    qkv_w: GpuWeight,
    o_w: GpuWeight,
    gu_w: GpuWeight,
    dp_w: GpuWeight,
    nqh: usize, nkvh: usize, hd: usize, hs: usize, eps: f32,
}

impl GpuDecoderLayer {
    fn load(w: &HashMap<String, RawTensor>, p: &str, cfg: &TextDecoderConfig, cuda: &CudaState) -> Result<Self> {
        Ok(Self {
            iln_w: load_gpu_vec(cuda, w, &format!("{}.input_layernorm.weight", p))?,
            pln_w: load_gpu_vec(cuda, w, &format!("{}.post_attention_layernorm.weight", p))?,
            qn_w: load_gpu_vec(cuda, w, &format!("{}.self_attn.q_norm.weight", p))?,
            kn_w: load_gpu_vec(cuda, w, &format!("{}.self_attn.k_norm.weight", p))?,
            qkv_w: load_fused_qkv_weight(w, &format!("{}.self_attn", p), cuda)?.0,
            o_w: load_gpu_weight(cuda, w, &format!("{}.self_attn.o_proj.weight", p))?,
            gu_w: load_fused_gate_up_weight(w, &format!("{}.mlp", p), cuda)?.0,
            dp_w: load_gpu_weight(cuda, w, &format!("{}.mlp.down_proj.weight", p))?,
            nqh: cfg.num_attention_heads,
            nkvh: cfg.num_key_value_heads,
            hd: cfg.head_dim,
            hs: cfg.hidden_size,
            eps: cfg.rms_norm_eps as f32,
        })
    }

    /// x: [b, s, hs]  (always GPU, consumed — reused as the residual stream so we
    /// don't allocate+memcpy a clone before the O-proj accumulation).
    /// cos/sin: [s, d]  device-resident slice for the positions of this call
    fn forward(&self, x: GpuTensor, cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
               kv: &mut GpuKvCache, layer_idx: usize, kv_start: usize, use_causal: bool,
               cuda: &CudaState) -> Result<GpuTensor>
    {
        let b = x.shape()[0]; let s = x.shape()[1];

        // 1. Input RMSNorm
        let normed = cuda.rms_norm(&x, &self.iln_w, self.eps)?;

        // 2. Fused QKV projection
        let qkv = cuda.linear_gpu(&normed, &self.qkv_w)?;
        let q_dim = self.nqh * self.hd;
        let kv_dim = self.nkvh * self.hd;

        // 3+4. Fused QKV: extract Q (norm+rotary), K (norm+rotary→cache), V (raw→cache).
        // One launch replaces qkv_extract_q_norm_rotary + qkv_extract_kv_norm_rotary_cache.
        let q = cuda.qkv_extract_qkv_norm_rotary_cache(
            &mut kv.k[layer_idx], &mut kv.v[layer_idx],
            &qkv, &self.qn_w, &self.kn_w, cos, sin,
            self.nqh, self.nkvh, self.hd, q_dim, kv_dim,
            kv.max_seq, kv_start, kv_start, self.eps,
        )?;
        drop(qkv);
        let cur_len = kv_start + s;

        // 6+7. Attention.  Three paths:
        //   - Prefill (s > 1): repeat_kv + cuBLAS strided batched GEMMs.
        //   - Decode short (s == 1, cur_len ≤ 1024): single-block fused_gqa_decode.
        //   - Decode long  (s == 1, cur_len > 1024): split-K fused_gqa_decode (multiple blocks
        //     per (b, q_head), online-softmax merge).  Cuts per-token attention latency by ~2x
        //     on long contexts where a single block bottlenecks on chunk-serial K reads.
        let attn_out = if s == 1 {
            let scale = 1.0f32 / (self.hd as f32).sqrt();
            const SPLIT_THRESHOLD: usize = 1024;
            // Adaptive chunk: smaller chunks = more parallelism but more merge overhead.
            // Empirically (P104, sm_61): 256 sweet spot until ~2048, then 512 wins on long ctx.
            let chunk_size: usize = if cur_len >= 2048 { 512 } else { 256 };
            if cur_len > SPLIT_THRESHOLD {
                cuda.fused_gqa_decode_split(&q, &kv.k[layer_idx], &kv.v[layer_idx],
                    self.nkvh, kv.max_seq, cur_len, scale, chunk_size)?
                    .reshape(vec![b, self.nqh, 1, self.hd])
            } else {
                cuda.fused_gqa_decode(&q, &kv.k[layer_idx], &kv.v[layer_idx],
                    self.nkvh, kv.max_seq, cur_len, scale)?
            }
        } else {
            let nr = self.nqh / self.nkvh;
            let k_rep = cuda.repeat_kv_from_cache(&kv.k[layer_idx], b, self.nkvh, kv.max_seq, self.hd, nr, cur_len)?;
            let v_rep = cuda.repeat_kv_from_cache(&kv.v[layer_idx], b, self.nkvh, kv.max_seq, self.hd, nr, cur_len)?;
            let scores = cuda.attention_qk(&q, &k_rep)?;
            let scale = 1.0f32 / (self.hd as f32).sqrt();
            let attn = cuda.softmax_scaled_causal(&scores, scale, use_causal && s > 1)?;
            drop(scores);
            cuda.attention_av(&attn, &v_rep)?
        };

        // 8. Reshape [b, h, s, d] → [b, s, h*d], then O projection with residual add (beta=1).
        //    For decode (s == 1), swap_dims_12 is a no-op (both layouts collapse to [b, h*d]),
        //    so we can skip the kernel and just reshape.
        let attn_flat = if s == 1 {
            attn_out.reshape(vec![b, 1, self.nqh * self.hd])
        } else {
            cuda.swap_dims_12(&attn_out)?.reshape(vec![b, s, self.nqh * self.hd])
        };
        // h = x.clone() then h += attn_flat @ O^T   (cuBLAS beta=1)
        let mut h = x;
        cuda.linear_gpu_accum(&mut h, &attn_flat, &self.o_w)?;
        drop(attn_flat);

        // 10. Post-attention RMSNorm
        let normed2 = cuda.rms_norm(&h, &self.pln_w, self.eps)?;

        // 11. Fused gate-up → SiLU·up
        let gu = cuda.linear_gpu(&normed2, &self.gu_w)?;
        let activated = cuda.silu_mul_split(&gu)?;
        drop(gu);

        // 12. Down projection with residual add (beta=1) — fused, no separate add_inplace.
        cuda.linear_gpu_accum(&mut h, &activated, &self.dp_w)?;

        let _ = self.hs;  // silence
        Ok(h)
    }

    /// Decode-only forward (s=1, b=1).  Zero-alloc path using DecodeScratch.
    /// `h` is the hidden state buffer [hs] (flat), modified in-place.
    fn forward_decode(&self, h: &mut CudaSlice<bf16>, cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
                      kv: &mut GpuKvCache, layer_idx: usize, kv_start: usize,
                      cuda: &CudaState, scratch: &mut DecodeScratch) -> Result<()>
    {
        let hs = self.hs;

        // 1. Input RMSNorm → scratch.norm1
        cuda.rms_norm_into_flat(h, hs, 1, &self.iln_w, self.eps, &mut scratch.norm1)?;

        // 2. Fused QKV projection → scratch.qkv
        cuda.linear_gpu_into_flat(&scratch.norm1, 1, hs, &self.qkv_w, &mut scratch.qkv)?;

        // 3+4. Extract Q+KV → scratch.q_out + KV cache write
        let q_dim = self.nqh * self.hd;
        let kv_dim = self.nkvh * self.hd;
        cuda.qkv_extract_into(
            &mut kv.k[layer_idx], &mut kv.v[layer_idx],
            &scratch.qkv, &self.qn_w, &self.kn_w, cos, sin,
            self.nqh, self.nkvh, self.hd, q_dim, kv_dim,
            kv.max_seq, kv_start, kv_start, self.eps,
            &mut scratch.q_out,
        )?;
        let cur_len = kv_start + 1;

        // 5. Attention. Single-block fused_gqa_decode for all context lengths — it is both
        //    correct and (empirically) the fastest path on this model/P104. The split-K path
        //    has a divergence bug and the matmul (repeat_kv+qk+softmax+av) path is ~2.5x
        //    slower due to repeat/alloc overhead. Optimizing long-context decode further is
        //    future work (e.g. a fixed flash-attention kernel).
        let scale = 1.0f32 / (self.hd as f32).sqrt();
        cuda.fused_gqa_decode_into(
            &scratch.q_out, &kv.k[layer_idx], &kv.v[layer_idx],
            self.nqh, self.nkvh, kv.max_seq, cur_len, scale,
            &mut scratch.attn_out,
        )?;

        // 6. O-proj + residual: h += attn_out @ O^T  (cuBLAS beta=1)
        cuda.linear_gpu_accum_slice(h, &scratch.attn_out, &self.o_w)?;

        // 7. Post-attention RMSNorm → scratch.norm2
        cuda.rms_norm_into_flat(h, hs, 1, &self.pln_w, self.eps, &mut scratch.norm2)?;

        // 8. Gate-up projection → scratch.gate_up
        cuda.linear_gpu_into_flat(&scratch.norm2, 1, hs, &self.gu_w, &mut scratch.gate_up)?;

        // 9. SiLU*up → scratch.activated
        cuda.silu_mul_split_into(&scratch.gate_up, 1, scratch.gate_up.len(), &mut scratch.activated)?;

        // 10. Down projection + residual: h += activated @ D^T  (cuBLAS beta=1)
        cuda.linear_gpu_accum_slice(h, &scratch.activated, &self.dp_w)?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Text Decoder
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub(crate) struct GpuTextDecoder {
    pub embed_table: GpuWeight,        // [vocab, hidden]  (also reused as lm_head)
    layers: Vec<GpuDecoderLayer>,
    norm_w: CudaSlice<bf16>,            // [hidden]
    eps: f32,
    pub config: TextDecoderConfig,
    pub cuda: Arc<CudaState>,
}

impl GpuTextDecoder {
    pub fn load_with(cuda: Arc<CudaState>, weights: &HashMap<String, RawTensor>, prefix: &str, config: &TextDecoderConfig) -> Result<Self> {
        let (embed_f16, embed_shape) = get_weight_f16(weights, &format!("{}.embed_tokens.weight", prefix))?;
        let embed_dev = cuda.upload_f16(&embed_f16)?;
        let embed_table = GpuWeight { data: embed_dev, rows: embed_shape[0], cols: embed_shape[1] };

        let norm_w = load_gpu_vec(&cuda, weights, &format!("{}.norm.weight", prefix))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(GpuDecoderLayer::load(weights, &format!("{}.layers.{}", prefix, i), config, &cuda)?);
        }

        Ok(Self { embed_table, layers, norm_w, eps: config.rms_norm_eps as f32, config: config.clone(), cuda })
    }

    pub fn embed_ids(&self, ids: &[i64]) -> Result<GpuTensor> {
        let ids_gpu = self.cuda.upload_i64(ids)?;
        self.cuda.embed_lookup(&self.embed_table, &ids_gpu)
    }

    /// Forward pass.
    /// hs: [1, sl, hidden] (GPU). cos/sin: [sl, head_dim] (GPU).
    /// kv_start: how many positions are already in the cache (0 for prefill).
    /// Returns logits as a GpuTensor of shape [1, out_sl, vocab] (out_sl = 1 if llo).
    pub fn forward(&self, hs: GpuTensor, cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
                   kv: &mut GpuKvCache, kv_start: usize, use_causal: bool, llo: bool) -> Result<GpuTensor>
    {
        let sl = hs.shape()[1];
        let mut h = hs;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h, cos, sin, kv, i, kv_start, use_causal, &self.cuda)?;
        }
        kv.cur_len = kv_start + sl;

        // Final RMSNorm
        let h = self.cuda.rms_norm(&h, &self.norm_w, self.eps)?;

        // Low-latency optimization: slice last token for prefill
        let h = if llo && sl > 1 {
            self.slice_last_token(&h)?
        } else {
            h
        };

        // LM head (shared with embed_table)
        self.cuda.linear_gpu(&h, &self.embed_table)
    }

    /// Zero-alloc decode: runs all layers via scratch buffers, final norm + lm_head,
    /// writes argmax into `token_buf[slot]`.  The hidden state `h` must be [hs] bf16.
    pub fn forward_decode_scratch(&self, h: &mut CudaSlice<bf16>, cos: &CudaSlice<bf16>, sin: &CudaSlice<bf16>,
                                  kv: &mut GpuKvCache, kv_start: usize,
                                  token_buf: &mut CudaSlice<i32>,
                                  scratch: &mut DecodeScratch) -> Result<()>
    {
        // Run all layers, modifying h in-place
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_decode(h, cos, sin, kv, i, kv_start, &self.cuda, scratch)?;
        }
        kv.cur_len = kv_start + 1;

        // Final RMSNorm → scratch.final_norm
        self.cuda.rms_norm_into_flat(h, scratch.hs, 1, &self.norm_w, self.eps, &mut scratch.final_norm)?;

        // LM head → scratch.logits
        self.cuda.linear_gpu_into_flat(&scratch.final_norm, 1, scratch.hs, &self.embed_table, &mut scratch.logits)?;

        // Argmax into token_buf
        self.cuda.argmax_into_flat(&scratch.logits, scratch.vocab, token_buf, 0)?;

        Ok(())
    }

    fn slice_last_token(&self, h: &GpuTensor) -> Result<GpuTensor> {
        let s = h.shape();
        assert_eq!(s.len(), 3);
        let (b, sl, hidden) = (s[0], s[1], s[2]);
        // For [1, sl, hidden] the last token's contiguous row sits at offset (sl-1)*hidden.
        // We allocate a fresh buffer and ask the stream to copy device→device.
        let mut out = self.cuda.alloc_uninit_f16(b * hidden)?;
        let src_offset = (sl - 1) * hidden;
        let src_view = h.data.slice(src_offset..src_offset + b * hidden);
        self.cuda.stream.memcpy_dtod(&src_view, &mut out)?;
        Ok(GpuTensor::new(out, vec![b, 1, hidden]))
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Standard RoPE cos/sin precompute (CPU side, then upload to GPU)
// ═══════════════════════════════════════════════════════════════════════

/// Standard RoPE cos/sin as bf16 CpuTensors of shape [seq_len, head_dim],
/// half-mirrored layout (matches rotary_emb_f16 / rms_norm_rotary_f16 kernels).
pub(crate) fn compute_rope_cos_sin_f16(
    pos: &[i64], head_dim: usize, rope_theta: f64,
) -> (CpuTensor, CpuTensor) {
    let (cv, sv) = crate::mrope::compute_rope_cos_sin(pos, head_dim, rope_theta);
    let sl = pos.len();
    let cos = CpuTensor::new(cv.iter().map(|&v| bf16::from_f32(v)).collect(), vec![sl, head_dim]);
    let sin = CpuTensor::new(sv.iter().map(|&v| bf16::from_f32(v)).collect(), vec![sl, head_dim]);
    (cos, sin)
}
