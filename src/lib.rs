//! MOSS-Transcribe-Diarize Rust + CUDA inference port.
//!
//! Architecture: Whisper-Medium encoder + VQAdaptor + Qwen3-0.6B decoder.
//! CPU path is the alignment reference; CUDA path mirrors it for speed.

pub mod adaptor;
pub mod backend;
pub mod config;
pub mod cpu_engine;
pub mod error;
pub mod inference;
pub mod mel;
pub mod mrope;
pub mod processor;
pub mod raw_tensor;
pub mod transcript;
pub mod whisper;
#[cfg(feature = "cuda")]
pub mod cudarc_engine;
#[cfg(feature = "cuda")]
pub mod gpu_whisper;
pub mod weights;

pub use error::{AsrError, Result};
pub use inference::AsrInference;
pub use mel::load_audio_wav;
