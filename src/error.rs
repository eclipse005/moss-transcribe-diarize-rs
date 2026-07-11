//! Domain error type for the public library surface.
//!
//! Binary / CLI code may still use `anyhow` at the process boundary; library
//! callers get structured `AsrError` variants.

use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum AsrError {
    #[error("Model load failed: {0}")]
    ModelLoad(#[source] anyhow::Error),

    #[error("Audio decode failed: {0}")]
    AudioDecode(#[source] anyhow::Error),

    #[error("Inference failed: {0}")]
    Inference(#[source] anyhow::Error),

    #[error("Invalid backend: {0}")]
    InvalidBackend(String),

    /// Filesystem / raw I/O failures (also via `From<std::io::Error>`).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Model `config.json` (or similar) parse / schema failures.
    #[error("Config error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, AsrError>;

impl AsrError {
    pub fn model_load(err: impl Into<anyhow::Error>) -> Self {
        Self::ModelLoad(err.into())
    }

    pub fn audio_decode(err: impl Into<anyhow::Error>) -> Self {
        Self::AudioDecode(err.into())
    }

    pub fn inference(err: impl Into<anyhow::Error>) -> Self {
        Self::Inference(err.into())
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
}

impl From<anyhow::Error> for AsrError {
    fn from(err: anyhow::Error) -> Self {
        // Default mapping for internal `?` at the library boundary.
        Self::Inference(err)
    }
}
