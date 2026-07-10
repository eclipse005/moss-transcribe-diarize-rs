//! GPU Whisper-Medium audio encoder, mirror of the CPU `whisper.rs`.
//!
//! Reuses the generic cudarc kernels: layer_norm, conv1d_k3_gelu (GELU fused),
//! linear_gpu, attention_qk/av, softmax_scaled_causal (causal=false), add_pe,
//! swap_dims_12, add_inplace, gelu_inplace.
//! Whisper attention is full/bidirectional.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::f16;
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::WhisperAudioConfig;
use crate::cudarc_engine::{CudaState, GpuTensor, GpuWeight};
use crate::raw_tensor::RawTensor;

pub struct GpuWhisperEncoder {
    cuda: Arc<CudaState>,
    conv1_w: CudaSlice<f16>,   // [out_ch, in_ch, 3] flat
    conv1_b: CudaSlice<f16>,
    conv2_w: CudaSlice<f16>,
    conv2_b: CudaSlice<f16>,
    embed_positions: CudaSlice<f16>,
    layers: Vec<GpuWhisperLayerData>,
    ln_w: CudaSlice<f16>,
    ln_b: CudaSlice<f16>,
    d_model: usize,
    n_heads: usize,
    ffn: usize,
    eps: f32,
}

struct GpuWhisperLayerData {
    sln_w: CudaSlice<f16>, sln_b: CudaSlice<f16>,
    fln_w: CudaSlice<f16>, fln_b: CudaSlice<f16>,
    q_w: GpuWeight, q_b: CudaSlice<f16>,
    k_w: GpuWeight, k_b: CudaSlice<f16>,
    v_w: GpuWeight, v_b: CudaSlice<f16>,
    o_w: GpuWeight, o_b: CudaSlice<f16>,
    fc1_w: GpuWeight, fc1_b: CudaSlice<f16>,
    fc2_w: GpuWeight, fc2_b: CudaSlice<f16>,
}

impl GpuWhisperEncoder {
    pub fn synchronize(&self) -> Result<()> { self.cuda.synchronize() }

    pub fn load(cuda: Arc<CudaState>, weights: &HashMap<String, RawTensor>, prefix: &str, cfg: &WhisperAudioConfig) -> Result<Self> {
        let g = |name: &str| -> Result<GpuWeight> { crate::cudarc_engine::load_gpu_weight(&cuda, weights, name) };
        let gv = |name: &str| -> Result<CudaSlice<f16>> { crate::cudarc_engine::load_gpu_vec(&cuda, weights, name) };
        let p = |s: &str| format!("{}.{}", prefix, s);
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        // Load all weights first, then move cuda into the struct.
        // Conv1d weights are 3-D [out_ch, in_ch, 3] — load as flat f16 vectors
        // (the kernel reads them row-major directly).
        let conv1_w = gv(&p("conv1.weight"))?;
        let conv1_b = gv(&p("conv1.bias"))?;
        let conv2_w = gv(&p("conv2.weight"))?;
        let conv2_b = gv(&p("conv2.bias"))?;
        let embed_positions = gv(&p("embed_positions.weight"))?;
        let ln_w = gv(&p("layer_norm.weight"))?;
        let ln_b = gv(&p("layer_norm.bias"))?;
        for i in 0..cfg.encoder_layers {
            let lp = format!("{}.layers.{}", prefix, i);
            let g2 = |s: &str| -> Result<GpuWeight> { crate::cudarc_engine::load_gpu_weight(&cuda, weights, &format!("{}.{}", lp, s)) };
            let gv2 = |s: &str| -> Result<CudaSlice<f16>> { crate::cudarc_engine::load_gpu_vec(&cuda, weights, &format!("{}.{}", lp, s)) };
            // k_proj has bias=False in Whisper; fall back to a zero vector if absent.
            let k_b = match gv2("self_attn.k_proj.bias") {
                Ok(b) => b,
                Err(_) => cuda.upload_f16(&vec![f16::from_f32(0.0); cfg.d_model])?,
            };
            layers.push(GpuWhisperLayerData {
                sln_w: gv2("self_attn_layer_norm.weight")?,
                sln_b: gv2("self_attn_layer_norm.bias")?,
                fln_w: gv2("final_layer_norm.weight")?,
                fln_b: gv2("final_layer_norm.bias")?,
                q_w: g2("self_attn.q_proj.weight")?,
                q_b: gv2("self_attn.q_proj.bias")?,
                k_w: g2("self_attn.k_proj.weight")?,
                k_b,
                v_w: g2("self_attn.v_proj.weight")?,
                v_b: gv2("self_attn.v_proj.bias")?,
                o_w: g2("self_attn.out_proj.weight")?,
                o_b: gv2("self_attn.out_proj.bias")?,
                fc1_w: g2("fc1.weight")?,
                fc1_b: gv2("fc1.bias")?,
                fc2_w: g2("fc2.weight")?,
                fc2_b: gv2("fc2.bias")?,
            });
        }
        Ok(Self {
            cuda,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            embed_positions,
            layers,
            ln_w,
            ln_b,
            d_model: cfg.d_model,
            n_heads: cfg.encoder_attention_heads,
            ffn: cfg.encoder_ffn_dim,
            eps: 1e-5,
        })
    }

    /// mel: [1, num_mel, nb_max_frames] f32 host → uploaded. Returns host [1, T_out*d_model] f32.
    pub fn forward_f32(&self, mel: &[f32], num_mel: usize, nb_frames: usize) -> Result<Vec<f32>> {
        let b = 1usize;
        let mel_f16: Vec<f16> = mel.iter().map(|&v| f16::from_f32(v)).collect();
        let mel_gpu = self.cuda.upload_f16(&mel_f16)?;
        let mut x = GpuTensor::new(mel_gpu, vec![b, num_mel, nb_frames]);
        // conv1 (stride 1) + GELU, conv2 (stride 2) + GELU
        x = self.cuda.conv1d_k3_gelu(&x, &self.conv1_w, &self.conv1_b, num_mel, self.d_model, 1)?;
        x = self.cuda.conv1d_k3_gelu(&x, &self.conv2_w, &self.conv2_b, self.d_model, self.d_model, 2)?;
        // permute [b, d, t] → [b, t, d]
        let mut h = self.cuda.permute_bct_to_btc(&x)?;
        let t_out = h.shape()[1];
        // + learned positional embeddings (rows 0..t_out)
        h = self.cuda.add_pe(&h, &self.embed_positions, t_out)?;
        // encoder layers
        for layer in &self.layers {
            h = self.layer_forward(layer, h)?;
        }
        // final layer_norm
        h = self.cuda.layer_norm(&h, &self.ln_w, &self.ln_b, self.eps)?;
        // download → f32
        let host = self.cuda.download_tensor(&h)?;
        Ok(host.data.iter().map(|&v| v.to_f32()).collect())
    }

    fn layer_forward(&self, layer: &GpuWhisperLayerData, x: GpuTensor) -> Result<GpuTensor> {
        let b = x.shape()[0]; let s = x.shape()[1];
        let head_dim = self.d_model / self.n_heads;
        // pre-LN attn
        let normed = self.cuda.layer_norm(&x, &layer.sln_w, &layer.sln_b, self.eps)?;
        let mut q = self.cuda.linear_gpu(&normed, &layer.q_w)?;
        self.cuda.add_bias_inplace(&mut q, &layer.q_b)?;
        let mut k = self.cuda.linear_gpu(&normed, &layer.k_w)?;
        self.cuda.add_bias_inplace(&mut k, &layer.k_b)?;
        let mut v = self.cuda.linear_gpu(&normed, &layer.v_w)?;
        self.cuda.add_bias_inplace(&mut v, &layer.v_b)?;
        // reshape [b, s, d] → [b, h, s, hd]
        let q4 = self.cuda.swap_dims_12(&q.reshape(vec![b, s, self.n_heads, head_dim]))?;
        let k4 = self.cuda.swap_dims_12(&k.reshape(vec![b, s, self.n_heads, head_dim]))?;
        let v4 = self.cuda.swap_dims_12(&v.reshape(vec![b, s, self.n_heads, head_dim]))?;
        let scores = self.cuda.attention_qk(&q4, &k4)?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let attn = self.cuda.softmax_scaled_causal(&scores, scale, false)?; // bidirectional
        let out = self.cuda.attention_av(&attn, &v4)?; // [b, h, s, hd]
        // swap back [b,h,s,hd] → [b,s,h,hd] → reshape [b, s, d]
        let out = self.cuda.swap_dims_12(&out)?.reshape(vec![b, s, self.d_model]);
        let mut out = self.cuda.linear_gpu(&out, &layer.o_w)?;
        self.cuda.add_bias_inplace(&mut out, &layer.o_b)?;
        // residual
        let mut h = x;
        self.cuda.add_inplace(&mut h, &out)?;
        // pre-LN FFN
        let normed2 = self.cuda.layer_norm(&h, &layer.fln_w, &layer.fln_b, self.eps)?;
        let mut hidden = self.cuda.linear_gpu(&normed2, &layer.fc1_w)?;
        self.cuda.add_bias_inplace(&mut hidden, &layer.fc1_b)?;
        self.cuda.gelu_inplace(&mut hidden)?;
        let mut ffn_out = self.cuda.linear_gpu(&hidden, &layer.fc2_w)?;
        self.cuda.add_bias_inplace(&mut ffn_out, &layer.fc2_b)?;
        self.cuda.add_inplace(&mut h, &ffn_out)?;
        Ok(h)
    }
}
