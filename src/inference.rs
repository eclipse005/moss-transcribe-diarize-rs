//! Inference orchestration: load model → preprocess audio → encode → scatter → decode.
//!
//! CPU path first (node-by-node alignment with Python). CUDA path mirrors the
//! structure once parity is proven.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use crate::adaptor::{time_merge, CpuVqAdaptor};
use crate::config::{MossConfig, ProcessorConfig};
use crate::cpu_engine::{self, CpuKvCache, CpuTensor, CpuTextDecoder};
use crate::error::AsrError;
use crate::mel::{load_audio_wav, MelExtractor, MEL_SAMPLE_RATE, N_FFT, HOP_LENGTH};
use crate::processor::{audio_span_ids, audios_to_input_features};
use crate::raw_tensor::RawTensor;
use crate::whisper::{CpuEncTensor, CpuWhisperEncoder};

/// Token ids needed at prompt-build time (from tokenizer_config / added_tokens).
pub struct SpecialTokens {
    pub im_start: i64,      // 151644
    pub im_end: i64,        // 151645
    pub endoftext: i64,     // 151643
    pub audio_start: i64,   // 151669
    pub audio_pad: i64,     // 151671
    pub audio_end: i64,     // 151670
    pub newline: i64,       // 198
    pub digits: [i64; 10],
}

pub struct AsrInference {
    inner: Mutex<AsrInner>,
}

/// Build the MelExtractor using the exact mel filterbank the Python
/// WhisperFeatureExtractor computes (dumped to data/whisper_mel_filters.bin).
/// Using HF's slaney filterbank bit-for-bit eliminates the largest source of
/// mel divergence. Falls back to the librosa-style generated filterbank if the
/// data file is missing.
fn load_mel_extractor(n_fft: usize, hop_length: usize, num_mel_bins: usize) -> MelExtractor {
    let path = std::path::Path::new("data/whisper_mel_filters.bin");
    if let Ok(bytes) = std::fs::read(path) {
        let n_freqs = n_fft / 2 + 1;
        let expected = num_mel_bins * n_freqs * 4;
        if bytes.len() == expected {
            let filters: Vec<f32> = bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            log::info!("loaded Python-exact mel filterbank from {} ({} filters x {} freqs)",
                path.display(), num_mel_bins, n_freqs);
            return MelExtractor::new_with_filters(n_fft, hop_length, num_mel_bins, filters);
        }
        log::warn!("mel filterbank file size mismatch ({} vs {}), falling back to generated", bytes.len(), expected);
    } else {
        log::warn!("mel filterbank file not found at {}, using generated (slaney) filterbank", path.display());
    }
    MelExtractor::new(n_fft, hop_length, num_mel_bins, MEL_SAMPLE_RATE)
}

struct AsrInner {
    config: MossConfig,
    pc: ProcessorConfig,
    tokenizer: tokenizers::Tokenizer,
    mel: MelExtractor,
    encoder: CpuWhisperEncoder,
    adaptor: CpuVqAdaptor,
    decoder: CpuTextDecoder,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuStack>,
    weights: HashMap<String, RawTensor>,  // kept for node-dump alignment
    tokens: SpecialTokens,
}

/// GPU-resident encoder + decoder sharing one CudaState.
#[cfg(feature = "cuda")]
struct GpuStack {
    cuda: std::sync::Arc<crate::cudarc_engine::CudaState>,
    encoder: crate::gpu_whisper::GpuWhisperEncoder,
    adaptor: crate::cudarc_engine::GpuVqAdaptor,
    decoder: crate::cudarc_engine::GpuTextDecoder,
}

unsafe impl Send for AsrInner {}

impl AsrInference {
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::load_with_backend(model_dir, "cpu")
    }

    /// `backend` = "cpu" | "cuda" | "auto". cuda runs the Whisper encoder AND
    /// the Qwen3 decoder on GPU; the VQAdaptor + time-merge still run on CPU
    /// (small, alignment-proven).
    pub fn load_with_backend(model_dir: &Path, backend: &str) -> Result<Self> {
        let config = MossConfig::from_file(&model_dir.join("config.json"))?;
        let weight_data = crate::weights::load_weights(model_dir)?;
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer load failed: {}", e))?;

        let mel = load_mel_extractor(N_FFT, HOP_LENGTH, config.audio_config.num_mel_bins);
        let pc = ProcessorConfig::default();

        let encoder = CpuWhisperEncoder::load(&weight_data, "model.whisper_encoder", &config.audio_config)?;
        let adaptor = CpuVqAdaptor::load(
            &weight_data, "model.vq_adaptor",
            config.text_config.hidden_size,
            config.text_config.rms_norm_eps as f32,
        )?;
        let decoder = CpuTextDecoder::load(&weight_data, "model.language_model", &config.text_config)?;

        let tokens = build_special_tokens(&tokenizer, &config);

        // GPU stack (optional): shared CudaState drives both encoder + decoder.
        #[cfg(feature = "cuda")]
        let gpu = {
            let want_cuda = matches!(backend, "cuda" | "auto");
            if want_cuda {
                match crate::cudarc_engine::CudaState::new(0) {
                    Ok(state) => {
                        let cuda = std::sync::Arc::new(state);
                        let gpu_enc = match crate::gpu_whisper::GpuWhisperEncoder::load(cuda.clone(), &weight_data, "model.whisper_encoder", &config.audio_config) {
                            Ok(e) => e,
                            Err(e) => { log::warn!("GPU encoder load failed ({e}); using CPU"); return Self::finish_cpu(config, pc, tokenizer, mel, encoder, adaptor, decoder, weight_data, tokens); }
                        };
                        let gpu_dec = match crate::cudarc_engine::GpuTextDecoder::load_with(cuda.clone(), &weight_data, "model.language_model", &config.text_config) {
                            Ok(d) => d,
                            Err(e) => { log::warn!("GPU decoder load failed ({e}); using CPU"); return Self::finish_cpu(config, pc, tokenizer, mel, encoder, adaptor, decoder, weight_data, tokens); }
                        };
                        let gpu_adaptor = match crate::cudarc_engine::GpuVqAdaptor::load(
                            cuda.clone(), &weight_data, "model.vq_adaptor",
                            config.text_config.hidden_size, config.text_config.rms_norm_eps as f32,
                        ) {
                            Ok(a) => a,
                            Err(e) => { log::warn!("GPU adaptor load failed ({e}); using CPU"); return Self::finish_cpu(config, pc, tokenizer, mel, encoder, adaptor, decoder, weight_data, tokens); }
                        };
                        log::info!("GPU stack loaded: Whisper encoder + VQAdaptor + Qwen3 decoder (cuda backend)");
                        Some(GpuStack { cuda, encoder: gpu_enc, adaptor: gpu_adaptor, decoder: gpu_dec })
                    }
                    Err(e) => { log::warn!("CUDA init failed ({e}); using CPU"); None }
                }
            } else { None }
        };

        Ok(Self {
            inner: Mutex::new(AsrInner {
                config, pc, tokenizer, mel, encoder, adaptor, decoder,
                #[cfg(feature = "cuda")]
                gpu,
                weights: weight_data, tokens,
            }),
        })
    }

    #[cfg(feature = "cuda")]
    fn finish_cpu(config: MossConfig, pc: ProcessorConfig, tokenizer: tokenizers::Tokenizer, mel: MelExtractor,
                  encoder: CpuWhisperEncoder, adaptor: CpuVqAdaptor, decoder: CpuTextDecoder,
                  weight_data: HashMap<String, RawTensor>, tokens: SpecialTokens) -> Result<Self> {
        Ok(Self {
            inner: Mutex::new(AsrInner {
                config, pc, tokenizer, mel, encoder, adaptor, decoder, gpu: None, weights: weight_data, tokens,
            }),
        })
    }

    /// Full transcription. Returns decoded text.
    pub fn transcribe(&self, audio_path: &str, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let samples = load_audio_wav(audio_path, MEL_SAMPLE_RATE).map_err(|e| match e {
            AsrError::AudioDecode(inner) => inner,
            other => anyhow!("{}", other),
        })?;
        self.transcribe_samples(&samples, prompt, max_new_tokens)
    }

    pub fn transcribe_samples(&self, samples: &[f32], prompt: &str, max_new_tokens: usize) -> Result<String> {
        let inner = self.inner.lock().map_err(|_| anyhow!("mutex poisoned"))?;
        inner.run(samples, prompt, max_new_tokens)
    }
}

fn build_special_tokens(tok: &tokenizers::Tokenizer, config: &MossConfig) -> SpecialTokens {
    let conv_id = |s: &str| -> i64 {
        tok.token_to_id(s).map(|v| v as i64).unwrap_or(0)
    };
    // For non-added literal tokens (e.g. "\n") encode and take the single-token id.
    let single_id = |s: &str| -> i64 {
        if let Ok(e) = tok.encode(s, false) {
            let ids = e.get_ids();
            if ids.len() == 1 { return ids[0] as i64; }
        }
        conv_id(s)
    };
    // audio tokens: from added_tokens, ids are fixed (151669/151670/151671).
    let audio_start = conv_id("<|audio_start|>");
    let audio_end = conv_id("<|audio_end|>");
    let audio_pad = config.audio_token_id;
    let mut digits = [0i64; 10];
    for d in 0..10 {
        digits[d] = single_id(&d.to_string());
    }
    SpecialTokens {
        im_start: conv_id("<|im_start|>"),
        im_end: conv_id("<|im_end|>"),
        endoftext: conv_id("<|endoftext|>"),
        audio_start, audio_end, audio_pad,
        newline: single_id("\n"),
        digits,
    }
}

impl AsrInner {
    fn run(&self, samples: &[f32], prompt: &str, max_new_tokens: usize) -> Result<String> {
        let bench = std::env::var("MOSS_BENCH").is_ok();
        let t_total = std::time::Instant::now();
        let mut t_preproc = std::time::Duration::ZERO;
        let mut t_encode = std::time::Duration::ZERO;
        let mut t_prompt = std::time::Duration::ZERO;
        let mut t_prefill = std::time::Duration::ZERO;
        let mut t_decode = std::time::Duration::ZERO;

        // 1. preprocess: mel + chunking + token lengths
        let tt = std::time::Instant::now();
        let audios = vec![samples.to_vec()];
        let (input_features, n_chunks, num_mel, feature_lengths, chunk_mapping) =
            audios_to_input_features(&self.mel, &audios, &self.pc);

        // token count per audio (scatter_add equivalent; here single audio → sum of lengths)
        let num_audios = audios.len();
        let mut audio_token_counts = vec![0usize; num_audios];
        for (i, &ci) in chunk_mapping.iter().enumerate() {
            audio_token_counts[ci] += feature_lengths[i];
        }
        t_preproc = tt.elapsed();

        // 2. Whisper encoder over each chunk → trim → concat per audio → cast → time_merge → adaptor
        let tt = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        if let Some(gpu) = self.gpu.as_ref() { gpu.encoder.synchronize().ok(); }
        #[cfg(feature = "cuda")]
        let audio_embeds = if self.gpu.is_some() {
            self.encode_audio_gpu(&input_features, num_mel, &feature_lengths, &chunk_mapping)?
        } else {
            self.encode_audio(&input_features, n_chunks, num_mel, &feature_lengths, &chunk_mapping)?
        };
        #[cfg(not(feature = "cuda"))]
        let audio_embeds = self.encode_audio(
            &input_features, n_chunks, num_mel, &feature_lengths, &chunk_mapping,
        )?;
        #[cfg(feature = "cuda")]
        if let Some(gpu) = self.gpu.as_ref() { gpu.encoder.synchronize().ok(); }
        t_encode = tt.elapsed();

        // 3. Build prompt: chat template + audio span (with time markers)
        let tt = std::time::Instant::now();
        let (input_ids, _audio_start_pos) = self.build_prompt(prompt, audio_token_counts[0], audio_embeds.len() / self.config.text_config.hidden_size)?;
        let seq_len = input_ids.len();
        let hidden = self.config.text_config.hidden_size;
        let audio_token_id = self.config.audio_token_id;

        // Optional node-dump for alignment testing (env MOSS_DUMP=path).
        if let Ok(dir) = std::env::var("MOSS_DUMP") {
            let dir = std::path::Path::new(&dir);
            let _ = std::fs::create_dir_all(dir);
            dump_f32(dir.join("rust_input_features.bin"), &input_features);
            dump_f32(dir.join("rust_audio_embeds.bin"), &audio_embeds);
            dump_i64(dir.join("rust_input_ids.bin"), &input_ids);
            log::info!("MOSS_DUMP: wrote input_features({}), audio_embeds({}), input_ids({}) to {}",
                input_features.len(), audio_embeds.len(), input_ids.len(), dir.display());
        }

        // 4. Embed ALL input_ids, then masked_scatter audio embeds into every
        //    audio_token_id position (matching Python's inputs_embeds.masked_scatter).
        //    Digit time-marker positions keep their own token embeddings.
        let all_emb = self.decoder.embed_ids(&input_ids);
        let mut hs_data = all_emb.data;
        let mut ai = 0usize;
        for pos in 0..seq_len {
            if input_ids[pos] == audio_token_id {
                let src = &audio_embeds[ai * hidden..(ai + 1) * hidden];
                hs_data[pos * hidden..(pos + 1) * hidden].copy_from_slice(src);
                ai += 1;
            }
        }
        assert_eq!(ai, audio_embeds.len() / hidden, "audio embed count != audio_token positions");
        let hidden_states = CpuTensor::new(hs_data, vec![1, seq_len, hidden]);
        t_prompt = tt.elapsed();

        // 5. RoPE tables
        let total_positions = seq_len + max_new_tokens;
        let all_pos: Vec<i64> = (0..total_positions as i64).collect();
        let (cos_table, sin_table) = cpu_engine::compute_rope_cos_sin(
            &all_pos, self.config.text_config.head_dim, self.config.text_config.rope_theta,
        );

        // 6. Prefill + decode (GPU path when available, else CPU).
        // Match Python generation_config.json: eos_token_id = 151645 (<|im_end|>) only.
        // 151643 (<|endoftext|>) is the pad/bos token, NOT an eos — stopping on it would
        // truncate generation if the model ever emits it mid-stream.
        let eos_ids: &[i64] = &[self.tokens.im_end];
        log::debug!("eos_ids im_end={} audio_pad={}", self.tokens.im_end, self.config.audio_token_id);
        let mut generated: Vec<u32> = Vec::new();
        let mut current_pos = seq_len;

        #[cfg(feature = "cuda")]
        let (t_prefill_dur, t_decode_dur, gen_ids, cp) = if let Some(gpu) = self.gpu.as_ref() {
            self.generate_gpu(gpu, &hidden_states, &all_pos, seq_len, max_new_tokens, eos_ids)?
        } else {
            self.generate_cpu(&hidden_states, &cos_table, &sin_table, total_positions, seq_len, max_new_tokens, eos_ids)?
        };
        #[cfg(not(feature = "cuda"))]
        let (t_prefill_dur, t_decode_dur, gen_ids, cp) = self.generate_cpu(
            &hidden_states, &cos_table, &sin_table, total_positions, seq_len, max_new_tokens, eos_ids,
        )?;
        generated = gen_ids;
        current_pos = cp;
        t_prefill = t_prefill_dur;
        t_decode = t_decode_dur;
        let _ = current_pos;
        log::debug!("generated last 8 ids: {:?}", generated.iter().rev().take(8).copied().collect::<Vec<_>>());

        // 7. Decode token ids → text
        let text = self.tokenizer.decode(&generated, true)
            .map_err(|e| anyhow!("decode: {}", e))?;
        if bench {
            let total = t_total.elapsed().as_secs_f64();
            let n = generated.len().max(1);
            log::info!(
                "BENCH total={:.0}ms | preproc={:.0}ms encode={:.0}ms prompt={:.0}ms prefill={:.0}ms decode={:.0}ms ({} tok, {:.2}ms/tok) | seq_len={} backend={}",
                total*1000.0,
                t_preproc.as_secs_f64()*1000.0, t_encode.as_secs_f64()*1000.0, t_prompt.as_secs_f64()*1000.0,
                t_prefill.as_secs_f64()*1000.0, t_decode.as_secs_f64()*1000.0, n, t_decode.as_secs_f64()*1000.0/(n as f64), seq_len,
                self.backend_tag(),
            );
        }
        Ok(text.trim().to_string())
    }

    fn backend_tag(&self) -> &'static str {
        #[cfg(feature = "cuda")]
        { if self.gpu.is_some() { "cuda" } else { "cpu" } }
        #[cfg(not(feature = "cuda"))]
        { "cpu" }
    }

    /// CPU prefill + decode loop. Returns (prefill_dur, decode_dur, generated_ids, final_pos).
    fn generate_cpu(
        &self,
        hidden_states: &CpuTensor,
        cos_table: &[f32], sin_table: &[f32],
        total_positions: usize, seq_len: usize, max_new_tokens: usize,
        eos_ids: &[i64],
    ) -> Result<(std::time::Duration, std::time::Duration, Vec<u32>, usize)> {
        let mut kv = CpuKvCache::new(
            self.config.text_config.num_hidden_layers, 1,
            self.config.text_config.num_key_value_heads, total_positions,
            self.config.text_config.head_dim,
        );
        let tt = std::time::Instant::now();
        let logits = self.decoder.forward(
            CpuTensor::new(hidden_states.data.clone(), hidden_states.shape.clone()),
            cos_table, sin_table, &mut kv, 0, true, true,
        );
        let mut next_token = cpu_engine::argmax(&logits.data) as i64;
        let t_prefill = tt.elapsed();
        let hidden = self.config.text_config.hidden_size;
        let mut generated: Vec<u32> = Vec::new();
        let mut current_pos = seq_len;
        let td = std::time::Instant::now();
        for _ in 0..max_new_tokens {
            if eos_ids.contains(&next_token) { break; }
            generated.push(next_token as u32);
            let ne = self.decoder.embed_ids(&[next_token]);
            let ne = CpuTensor::new(ne.data, vec![1, 1, hidden]);
            let sl = self.decoder.forward(ne, cos_table, sin_table, &mut kv, current_pos, true, true);
            next_token = cpu_engine::argmax(&sl.data) as i64;
            current_pos += 1;
        }
        Ok((t_prefill, td.elapsed(), generated, current_pos))
    }

    /// GPU prefill + zero-alloc decode loop. `hidden_states` is the [1, seq_len, hidden] f32
    /// CPU-assembled scatter; we upload it once. Returns (prefill, decode, ids, final_pos).
    #[cfg(feature = "cuda")]
    fn generate_gpu(
        &self,
        gpu: &GpuStack,
        hidden_states: &CpuTensor,
        all_pos: &[i64],
        seq_len: usize, max_new_tokens: usize, eos_ids: &[i64],
    ) -> Result<(std::time::Duration, std::time::Duration, Vec<u32>, usize)> {
        use crate::cudarc_engine::{
            compute_rope_cos_sin_f16, CpuTensor as GpuCpuTensor, DecodeScratch, GpuKvCache,
        };
        use half::bf16;

        let cuda = &gpu.cuda;
        let decoder = &gpu.decoder;
        let text_cfg = &self.config.text_config;
        let head_dim = text_cfg.head_dim;

        // Upload hidden_states [1, seq_len, hidden] as bf16.
        let hs_f16: Vec<bf16> = hidden_states.data.iter().map(|&v| bf16::from_f32(v)).collect();
        let hs_gpu = cuda.upload_tensor(&GpuCpuTensor::new(hs_f16, vec![1, seq_len, text_cfg.hidden_size]))?;

        // RoPE cos/sin tables over all positions, uploaded as bf16.
        let (cos_cpu, sin_cpu) = compute_rope_cos_sin_f16(all_pos, head_dim, text_cfg.rope_theta);
        let cos = cuda.upload_f16(&cos_cpu.data)?;
        let sin = cuda.upload_f16(&sin_cpu.data)?;

        let total_positions = all_pos.len();
        let mut kv = GpuKvCache::new(
            cuda, text_cfg.num_hidden_layers, 1,
            text_cfg.num_key_value_heads, total_positions, head_dim,
        )?;

        // Prefill: full forward, then argmax the last token's logits.
        let tp = std::time::Instant::now();
        let logits = decoder.forward(hs_gpu, &cos, &sin, &mut kv, 0, true, true)?;
        let mut token_buf = cuda.alloc_uninit_i32(1)?;
        cuda.argmax_into(&logits, &mut token_buf, 0)?;
        let mut pinned = unsafe { cuda.ctx.alloc_pinned::<i32>(1)? };
        cuda.download_i32_into_pinned(&token_buf, &mut pinned)?;
        let mut next_token = unsafe { *pinned.as_ptr()? } as i64;
        let t_prefill = tp.elapsed();

        // Decode loop: zero-alloc via DecodeScratch.
        let mut scratch = DecodeScratch::new(cuda, total_positions, text_cfg)?;
        let mut h_buf = scratch.embed_out.clone();
        let td = std::time::Instant::now();
        let mut generated: Vec<u32> = Vec::new();
        let mut current_pos = seq_len;
        loop {
            if eos_ids.contains(&next_token) { break; }
            generated.push(next_token as u32);
            if generated.len() >= max_new_tokens { break; }
            // embed next token (read id from token_buf on device) → h_buf
            cuda.embed_id_from_gpu_slot_into(&decoder.embed_table, &token_buf, 0, &mut h_buf)?;
            decoder.forward_decode_scratch(&mut h_buf, &cos, &sin, &mut kv, current_pos, &mut token_buf, &mut scratch)?;
            cuda.download_i32_into_pinned(&token_buf, &mut pinned)?;
            next_token = unsafe { *pinned.as_ptr()? } as i64;
            current_pos += 1;
        }
        cuda.synchronize()?;
        Ok((t_prefill, td.elapsed(), generated, current_pos))
    }


    /// GPU encode: encoder → trim → concat → time_merge(reshape) → GPU VQAdaptor.
    /// All on GPU; downloads only the final [nat, hidden] audio embeds once.
    #[cfg(feature = "cuda")]
    fn encode_audio_gpu(
        &self,
        input_features: &[f32],
        num_mel: usize,
        feature_lengths: &[usize],
        chunk_mapping: &[usize],
    ) -> Result<Vec<f32>> {
        use crate::cudarc_engine::GpuTensor;
        use half::bf16;
        let gpu = self.gpu.as_ref().unwrap();
        let cuda = &gpu.cuda;
        let nb_frames = self.pc.nb_max_frames;
        let d_model = self.config.audio_config.d_model;
        let merge = self.pc.audio_merge_size;
        let hidden = self.config.text_config.hidden_size;

        let num_audios = chunk_mapping.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let mut per_audio_chunks: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_audios];
        for (ci, &tl) in feature_lengths.iter().enumerate() {
            per_audio_chunks[chunk_mapping[ci]].push((ci, tl));
        }

        let mut adapted_all = Vec::new();
        for parts in &per_audio_chunks {
            // Encode each chunk on GPU → trim to tl*4 → collect trimmed GPU tensors.
            let mut cat_t = 0usize;
            let mut cat_chunks: Vec<GpuTensor> = Vec::new();
            for &(ci, tl) in parts {
                let chunk_mel = &input_features[ci * num_mel * nb_frames..(ci + 1) * num_mel * nb_frames];
                let enc = gpu.encoder.forward_gpu(chunk_mel, num_mel, nb_frames)?; // [1, T_out, d_model]
                let t_out = enc.shape()[1];
                let keep = (tl * 4).min(t_out);
                // slice [1, keep, d_model] from [1, t_out, d_model]
                let sliced = cuda.slice_dim1(&enc, 0, keep)?;
                cat_t += keep;
                cat_chunks.push(sliced);
            }
            // Concat along dim 1 → [1, cat_t, d_model]
            let mut cat_data = cat_chunks[0].data.clone();
            let total_cat = cat_t * d_model;
            let mut full = unsafe { cuda.stream.alloc::<bf16>(total_cat)? };
            let mut offset = 0usize;
            for chunk in &cat_chunks {
                let n = chunk.data.len();
                cuda.stream.memcpy_dtod(&chunk.data.slice(..), &mut full.slice_mut(offset..offset + n))?;
                offset += n;
            }
            let cat = GpuTensor::new(full, vec![1, cat_t, d_model]);
            // time_merge: (1, cat_t, d_model) → (1, cat_t//merge, d_model*merge)
            // This is a pure reshape (contiguous frames grouped). Trim to multiple of merge first.
            let t_trim = (cat_t / merge) * merge;
            let trimmed = if t_trim == cat_t { cat } else { cuda.slice_dim1(&cat, 0, t_trim)? };
            let merged = trimmed.reshape(vec![1, t_trim / merge, d_model * merge]);
            // GPU VQAdaptor: [1, out_t, d_model*merge] → [1, out_t, hidden]
            let adapted = gpu.adaptor.forward(cuda, &merged)?;
            let out_t = adapted.shape()[1];
            // Download once
            let host = cuda.download_tensor(&adapted)?;
            adapted_all.extend(host.data.iter().take(out_t * hidden).map(|&v| v.to_f32()));
        }
        Ok(adapted_all)
    }

    /// Encode audio: Whisper encoder per chunk → trim → concat → time_merge → VQAdaptor.
    /// Returns [nat, hidden] f32 (hidden = text hidden_size).
    fn encode_audio(
        &self,
        input_features: &[f32],
        n_chunks: usize,
        num_mel: usize,
        feature_lengths: &[usize],
        chunk_mapping: &[usize],
    ) -> Result<Vec<f32>> {
        let nb_frames = self.pc.nb_max_frames;
        let d_model = self.config.audio_config.d_model;
        let merge = self.pc.audio_merge_size;
        let dtype_hidden = self.config.text_config.hidden_size;

        // group chunks per audio
        let num_audios = chunk_mapping.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let mut per_audio_chunks: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_audios]; // (chunk_idx, token_len)
        for (ci, &tl) in feature_lengths.iter().enumerate() {
            let sidx = chunk_mapping[ci];
            per_audio_chunks[sidx].push((ci, tl));
        }

        let mut adapted_all = Vec::new();
        for parts in &per_audio_chunks {
            // run encoder per chunk, trim to token_len*4 frames, concat along time
            let mut cat: Vec<f32> = Vec::new();
            let mut cat_t = 0usize;
            for &(ci, tl) in parts {
                let chunk_mel = &input_features[ci * num_mel * nb_frames..(ci + 1) * num_mel * nb_frames];
                // GPU path returns host f32 [1, T_out, d_model]; CPU path returns CpuEncTensor.
                #[cfg(feature = "cuda")]
                let (enc_data, t_out): (Vec<f32>, usize) = if let Some(gpu) = self.gpu.as_ref() {
                    let out = gpu.encoder.forward_f32(chunk_mel, num_mel, nb_frames)?;
                    let t = out.len() / d_model;
                    (out, t)
                } else {
                    let mel_t = CpuEncTensor::new(chunk_mel.to_vec(), vec![1, num_mel, nb_frames]);
                    let enc = self.encoder.forward(&mel_t); // [1, T_out, d_model]
                    (enc.data, enc.shape[1])
                };
                #[cfg(not(feature = "cuda"))]
                let (enc_data, t_out): (Vec<f32>, usize) = {
                    let mel_t = CpuEncTensor::new(chunk_mel.to_vec(), vec![1, num_mel, nb_frames]);
                    let enc = self.encoder.forward(&mel_t);
                    (enc.data, enc.shape[1])
                };
                let keep = (tl * 4).min(t_out);
                cat.extend_from_slice(&enc_data[..keep * d_model]);
                cat_t += keep;
            }
            // cast to model dtype (f32 here; Python casts bf16/fp16 then merges)
            // time_merge (B=1, T=cat_t, D=d_model) → (1, cat_t/merge, d_model*merge)
            let (merged, out_t, out_d) = time_merge(&cat, 1, cat_t, d_model, merge);
            // adaptor: [1, out_t, out_d] → [1, out_t, hidden]
            let adapted = self.adaptor.forward(&merged, 1, out_t, out_d);
            adapted_all.extend_from_slice(&adapted[..out_t * dtype_hidden]);
        }
        Ok(adapted_all)
    }

    /// Build the full input_ids sequence (chat template) and return the audio span start position.
    /// Format (from chat_template.jinja, add_generation_prompt=True, single user turn):
    ///   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    ///   <|im_start|>user\n<|audio_start|>{audio_span}<|audio_end|>\n{prompt}<|im_end|>\n
    ///   <|im_start|>assistant\n
    fn build_prompt(&self, prompt: &str, audio_token_count: usize, nat: usize) -> Result<(Vec<i64>, usize)> {
        let t = &self.tokens;
        // audio span (with time markers)
        let span = audio_span_ids(nat, t.audio_pad, &t.digits, &self.pc);

        // Encode text pieces via tokenizer (no special tokens for the literal text portions).
        let enc = |s: &str| -> Result<Vec<i64>> {
            let e = self.tokenizer.encode(s, false).map_err(|x| anyhow!("encode: {}", x))?;
            Ok(e.get_ids().iter().map(|&id| id as i64).collect())
        };

        let mut ids: Vec<i64> = Vec::new();
        // system
        ids.push(t.im_start);
        ids.extend(enc("system")?);
        ids.push(t.newline);
        ids.extend(enc("You are a helpful assistant.")?);
        ids.push(t.im_end);
        ids.push(t.newline);
        // user
        ids.push(t.im_start);
        ids.extend(enc("user")?);
        ids.push(t.newline);
        ids.push(t.audio_start);
        let audio_start_pos = ids.len();
        ids.extend_from_slice(&span);
        ids.push(t.audio_end);
        ids.push(t.newline);
        ids.extend(enc(prompt.trim())?);
        ids.push(t.im_end);
        ids.push(t.newline);
        // assistant
        ids.push(t.im_start);
        ids.extend(enc("assistant")?);
        ids.push(t.newline);

        let _ = audio_token_count; // already used to derive nat
        Ok((ids, audio_start_pos))
    }
}

// ─── dump helpers (raw binary: little-endian f32 / i64, row-major) ─────

fn dump_f32(path: std::path::PathBuf, data: &[f32]) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::File::create(&path) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = f.write_all(&bytes);
    }
}

fn dump_i64(path: std::path::PathBuf, data: &[i64]) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::File::create(&path) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = f.write_all(&bytes);
    }
}

