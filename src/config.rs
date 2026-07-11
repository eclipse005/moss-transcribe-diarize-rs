//! Config types for MOSS-Transcribe-Diarize (Whisper-Medium encoder + Qwen3-0.6B decoder).
//!
//! Mirrors `configuration_moss_transcribe_diarize.py`. We only deserialize the
//! fields the inference path touches; the rest are ignored.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // fields deserialized for forward-compat with HF config.json
pub struct MossConfig {
    pub text_config: Qwen3TextConfig,
    pub audio_config: WhisperAudioConfig,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: i64,
    #[serde(default = "default_audio_merge_size")]
    pub audio_merge_size: usize,
    #[serde(default = "default_adaptor_input_dim")]
    pub adaptor_input_dim: usize,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: i64,
}

fn default_audio_token_id() -> i64 { 151671 }
fn default_audio_merge_size() -> usize { 4 }
fn default_adaptor_input_dim() -> usize { 4096 }
fn default_tie_word_embeddings() -> bool { true }
fn default_pad_token_id() -> i64 { 151643 }

impl MossConfig {
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Qwen3TextConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

/// Alias used by the reused decoder engines (cudarc_engine.rs / cpu_engine.rs),
/// which were written against qwen3-asr-rs's `TextDecoderConfig`. Same fields.
pub type TextDecoderConfig = Qwen3TextConfig;

fn default_vocab_size() -> usize { 151936 }
fn default_hidden_size() -> usize { 1024 }
fn default_intermediate_size() -> usize { 3072 }
fn default_num_hidden_layers() -> usize { 28 }
fn default_num_attention_heads() -> usize { 16 }
fn default_num_key_value_heads() -> usize { 8 }
fn default_head_dim() -> usize { 128 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f64 { 1_000_000.0 }
fn default_max_position_embeddings() -> usize { 131072 }

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct WhisperAudioConfig {
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,
    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: usize,
    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: usize,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
    #[serde(default = "default_scale_embedding")]
    pub scale_embedding: bool,
}

fn default_num_mel_bins() -> usize { 80 }
fn default_d_model() -> usize { 1024 }
fn default_encoder_layers() -> usize { 24 }
fn default_encoder_attention_heads() -> usize { 16 }
fn default_encoder_ffn_dim() -> usize { 4096 }
fn default_max_source_positions() -> usize { 1500 }
fn default_scale_embedding() -> bool { false }

/// Processor constants (from processor_config.json + preprocessor_config.json).
/// These are fixed for this model; kept here as a single source of truth.
#[allow(dead_code)]
pub struct ProcessorConfig {
    pub sampling_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_samples: usize,             // 30s @ 16kHz = 480000
    pub nb_max_frames: usize,         // 3000
    pub audio_tokens_per_second: f64, // 12.5
    pub audio_merge_size: usize,      // 4
    pub time_marker_every_seconds: usize, // 5
    pub enable_time_marker: bool,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            n_samples: 480000,
            nb_max_frames: 3000,
            audio_tokens_per_second: 12.5,
            audio_merge_size: 4,
            time_marker_every_seconds: 5,
            enable_time_marker: true,
        }
    }
}
