//! Backend selection — pure tag enum, no internal types.
//!
//! `Backend` is a lightweight selection tag.  Pass it to [`crate::AsrInference::load_with`]
//! to choose CPU or GPU.  The heavy lifting lives in `cudarc_engine` (GPU) and
//! `cpu_engine` (CPU); this module just owns the dispatch.
//!
//! This type is intentionally free of CLI framework traits (`clap`, etc.).
//! Parse strings with [`std::str::FromStr`] (or the binary's own value parser).

use std::str::FromStr;

#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::error::{AsrError, Result};

/// Compute backend selection.  Pass to [`crate::AsrInference::load_with`] to choose.
///
/// `Auto` detects the best available backend at load time (prefers CUDA when
/// the `cuda` feature is enabled and a device is present; falls back to CPU).
///
/// **CUDA init failure contract (historical):** both `Auto` and explicit `Cuda`
/// fall back to CPU with a warning if the device cannot be opened. Component
/// loads (encoder / decoder / adaptor) that fail after a successful device open
/// also fall back to CPU — see `AsrInference::load_with`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Detect the best available backend at load time.
    #[default]
    Auto,
    /// Force CPU inference.
    Cpu,
    /// Prefer CUDA (GPU 0). If device init fails, falls back to CPU (historical).
    #[cfg(feature = "cuda")]
    Cuda,
}

/// Internal resolved backend — carries `Arc<CudaState>` when CUDA is selected.
/// Never exposed in the public API.
pub(crate) enum ResolvedBackend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(Arc<crate::cudarc_engine::CudaState>),
}

impl Backend {
    /// Resolve to a concrete backend handle.
    ///
    /// Never returns `Err` for device open failures: CUDA init errors become
    /// [`ResolvedBackend::Cpu`] with a warning (matches the original stringly
    /// `"cuda"` / `"auto"` load path). Callers may still fail later on weight load.
    pub(crate) fn resolve(self) -> ResolvedBackend {
        match self {
            Backend::Cpu => ResolvedBackend::Cpu,
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                use crate::cudarc_engine::CudaState;
                match CudaState::new(0) {
                    Ok(state) => {
                        log::info!("Backend::Cuda: using CUDA device 0");
                        ResolvedBackend::Cuda(Arc::new(state))
                    }
                    Err(e) => {
                        log::warn!("CUDA init failed ({e}); using CPU");
                        ResolvedBackend::Cpu
                    }
                }
            }
            Backend::Auto => {
                #[cfg(feature = "cuda")]
                {
                    use crate::cudarc_engine::CudaState;
                    match CudaState::new(0) {
                        Ok(state) => {
                            log::info!("Auto: selected CUDA device 0");
                            ResolvedBackend::Cuda(Arc::new(state))
                        }
                        Err(e) => {
                            log::warn!("Auto: CUDA init failed ({e}); falling back to CPU");
                            ResolvedBackend::Cpu
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    log::info!("Auto: no GPU backend available, using CPU");
                    ResolvedBackend::Cpu
                }
            }
        }
    }

    /// Prefer the best available backend (`Auto`).
    pub fn best() -> Self {
        Backend::Auto
    }

    /// Short human label — useful for logs.
    pub fn tag(&self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            Backend::Cuda => "cuda:0",
        }
    }
}

impl FromStr for Backend {
    type Err = AsrError;

    /// Parse `"auto" | "cpu" | "cuda" | "gpu"` (case-insensitive).
    ///
    /// Unknown tags and `"cuda"`/`"gpu"` on a build without the `cuda` feature
    /// return [`AsrError::InvalidBackend`] (stricter than the old silent-CPU path).
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Backend::Auto),
            "cpu" => Ok(Backend::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" | "gpu" => Ok(Backend::Cuda),
            #[cfg(not(feature = "cuda"))]
            "cuda" | "gpu" => Err(AsrError::InvalidBackend(
                "cuda requested but this build has no `cuda` feature".into(),
            )),
            other => Err(AsrError::InvalidBackend(format!(
                "unknown backend {other:?}; expected auto|cpu{}",
                if cfg!(feature = "cuda") {
                    "|cuda|gpu"
                } else {
                    ""
                }
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_backend_names() {
        assert_eq!(Backend::from_str("auto").unwrap(), Backend::Auto);
        assert_eq!(Backend::from_str("CPU").unwrap(), Backend::Cpu);
        #[cfg(feature = "cuda")]
        {
            assert_eq!(Backend::from_str("cuda").unwrap(), Backend::Cuda);
            assert_eq!(Backend::from_str("gpu").unwrap(), Backend::Cuda);
        }
        assert!(Backend::from_str("tpu").is_err());
    }

    #[test]
    fn default_is_auto() {
        assert_eq!(Backend::default(), Backend::Auto);
    }
}
