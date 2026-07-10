//! Backend selection — pure tag enum, no internal types.
//!
//! `Backend` is a lightweight selection tag.  Pass it to [`AsrInference::load`]
//! to choose CPU or GPU.  The heavy lifting lives in `cudarc_engine` (GPU) and
//! `cpu_engine` (CPU); this module just owns the dispatch.

#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Compute backend selection.  Pass to [`crate::AsrInference::load`] to choose.
///
/// `Auto` detects the best available backend at load time (prefers CUDA when
/// the `cuda` feature is enabled and a device is present; falls back to CPU).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Detect the best available backend at load time.
    Auto,
    /// Force CPU inference.
    Cpu,
    /// Force CUDA inference (GPU 0).
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
    /// Resolve `Auto` to a concrete backend; leave explicit choices unchanged.
    pub(crate) fn resolve(self) -> anyhow::Result<ResolvedBackend> {
        match self {
            Backend::Cpu => Ok(ResolvedBackend::Cpu),
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                use crate::cudarc_engine::CudaState;
                let state = CudaState::new(0)?;
                Ok(ResolvedBackend::Cuda(Arc::new(state)))
            }
            Backend::Auto => {
                #[cfg(feature = "cuda")]
                {
                    use crate::cudarc_engine::CudaState;
                    match CudaState::new(0) {
                        Ok(state) => {
                            log::info!("Auto: selected CUDA device 0");
                            Ok(ResolvedBackend::Cuda(Arc::new(state)))
                        }
                        Err(e) => {
                            log::warn!("Auto: CUDA init failed ({e}); falling back to CPU");
                            Ok(ResolvedBackend::Cpu)
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    log::info!("Auto: no GPU backend available, using CPU");
                    Ok(ResolvedBackend::Cpu)
                }
            }
        }
    }

    /// Backward-compatible convenience: returns `Backend::Auto`.
    pub fn best() -> anyhow::Result<Self> {
        Ok(Backend::Auto)
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
