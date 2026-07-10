//! Raw on-disk tensor view, loaded directly from safetensors bytes.
//!
//! Used only for weight loading — never as a GPU/CPU computation tensor (those
//! live in `cudarc_engine` / `cpu_engine`).  `RawTensor` is a deserialization
//! intermediate; the engines consume the raw bytes and upload them to their
//! respective devices.
//!
//! `data` is a `bytes::Bytes` so it can share one mmap region across all
//! tensors via O(1) refcount+range slices (see `weights.rs`). `Bytes` derefs
//! to `&[u8]`, so the conversion paths below are unchanged from the old
//! `Vec<u8>` field and remain bit-exact.

use anyhow::{anyhow, Result};
use bytes::Bytes;
use safetensors::Dtype;

/// One tensor as it sits in the safetensors file: raw bytes + shape + dtype.
#[derive(Debug, Clone)]
pub struct RawTensor {
    /// Raw little-endian bytes (f32 = 4 bytes, f16/bf16 = 2 bytes, etc.).
    /// Backed by a refcounted slice of the mmap'd file; cheap to clone.
    pub data: Bytes,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl RawTensor {
    /// Convert raw bytes to a `Vec<f32>`. Supports F32 / F16 / BF16. Endianness: native (== LE on all supported targets).
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        match self.dtype {
            Dtype::F32 => Ok(self.data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            Dtype::F16 => Ok(self.data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]).to_f32())
                .collect()),
            Dtype::BF16 => Ok(self.data
                .chunks_exact(2)
                .map(|c| {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    f32::from_bits((b as u32) << 16)
                })
                .collect()),
            other => Err(anyhow!("unsupported dtype {:?} for to_f32_vec", other)),
        }
    }

    /// Convert raw bytes to a `Vec<half::f16>`. Supports F32 / F16 / BF16.
    ///
    /// NOTE: BF16→F16 is a **lossy** conversion (different exponent/mantissa split).
    /// For bit-exact alignment with a BF16 Python model, use `to_bf16_vec()` instead.
    pub fn to_f16_vec(&self) -> Result<Vec<half::f16>> {
        match self.dtype {
            Dtype::F16 => Ok(self.data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]))
                .collect()),
            Dtype::F32 => Ok(self.data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .map(half::f16::from_f32)
                .collect()),
            Dtype::BF16 => Ok(self.data
                .chunks_exact(2)
                .map(|c| {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    half::f16::from_f32(f32::from_bits((b as u32) << 16))
                })
                .collect()),
            other => Err(anyhow!("unsupported dtype {:?} for to_f16_vec", other)),
        }
    }

    /// Convert raw bytes to a `Vec<half::bf16>`, preserving BF16 bit-exactly.
    /// This is the correct loader for models stored/trained in BF16 (e.g. this
    /// MOSS model), so GPU compute in BF16 matches the Python original node-by-node.
    pub fn to_bf16_vec(&self) -> Result<Vec<half::bf16>> {
        match self.dtype {
            Dtype::BF16 => Ok(self.data
                .chunks_exact(2)
                .map(|c| half::bf16::from_ne_bytes([c[0], c[1]]))
                .collect()),
            Dtype::F32 => Ok(self.data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .map(half::bf16::from_f32)
                .collect()),
            Dtype::F16 => Ok(self.data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]).to_f32())
                .map(half::bf16::from_f32)
                .collect()),
            other => Err(anyhow!("unsupported dtype {:?} for to_bf16_vec", other)),
        }
    }

    /// (f32_data, shape) — convenience for loaders that need both.
    pub fn as_f32(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        Ok((self.to_f32_vec()?, self.shape.clone()))
    }

    /// (f16_data, shape) — convenience for loaders that need both.
    pub fn as_f16(&self) -> Result<(Vec<half::f16>, Vec<usize>)> {
        Ok((self.to_f16_vec()?, self.shape.clone()))
    }

    /// (bf16_data, shape) — convenience for loaders that need both.
    pub fn as_bf16(&self) -> Result<(Vec<half::bf16>, Vec<usize>)> {
        Ok((self.to_bf16_vec()?, self.shape.clone()))
    }
}
