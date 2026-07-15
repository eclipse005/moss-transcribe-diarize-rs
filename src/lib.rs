//! MOSS-Transcribe-Diarize Rust + CUDA inference port.
//!
//! Architecture: Whisper-Medium encoder + VQAdaptor + Qwen3-0.6B decoder.
//! CPU path is the alignment reference; CUDA path mirrors it for speed.

// Internal modules: not part of the stable public API.
pub(crate) mod adaptor;
pub(crate) mod config;
pub(crate) mod cpu_engine;
pub(crate) mod mel;
pub(crate) mod mrope;
pub(crate) mod processor;
pub(crate) mod raw_tensor;
pub(crate) mod resampler;
/// Long-form pipeline scaffolding (not wired into the public CLI path).
pub(crate) mod transcript;
pub(crate) mod whisper;
pub(crate) mod weights;

#[cfg(feature = "cuda")]
pub(crate) mod cudarc_engine;
#[cfg(feature = "cuda")]
pub(crate) mod gpu_whisper;
#[cfg(feature = "cuda")]
pub(crate) mod prebuilt_ptx;

// Public surface.
pub mod backend;
pub mod error;
pub mod inference;

pub use backend::Backend;
pub use error::{AsrError, Result};
pub use inference::AsrInference;
pub use mel::load_audio_wav;
