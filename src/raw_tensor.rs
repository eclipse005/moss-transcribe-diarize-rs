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
use rayon::prelude::*;
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

/// Bulk-copy raw LE bytes into a `Vec<T>` when `T` is a transparent 2- or 4-byte float.
/// Bit-exact; faster than per-element `from_ne_bytes` on large weight tensors.
///
/// Safety: `T` must be a plain POD of size `elem_size` with the same bit layout as the
/// on-disk little-endian encoding (true for `f32`, `half::f16`, `half::bf16` on LE hosts).
unsafe fn bulk_copy_pod<T: Copy>(bytes: &[u8], elem_size: usize) -> Result<Vec<T>> {
    if bytes.len() % elem_size != 0 {
        return Err(anyhow!(
            "tensor byte length {} is not a multiple of {}",
            bytes.len(),
            elem_size
        ));
    }
    let n = bytes.len() / elem_size;
    let mut out = Vec::with_capacity(n);
    // SAFETY: caller guarantees T bit-layout matches `bytes`; lengths match.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        out.set_len(n);
    }
    Ok(out)
}

impl RawTensor {
    /// Convert raw bytes to a `Vec<f32>`. Supports F32 / F16 / BF16. Endianness: native (== LE on all supported targets).
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        match self.dtype {
            // Bit-exact bulk path for native f32 weights.
            Dtype::F32 => unsafe { bulk_copy_pod::<f32>(&self.data, 4) },
            Dtype::F16 => Ok(self
                .data
                .par_chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]).to_f32())
                .collect()),
            Dtype::BF16 => Ok(self
                .data
                .par_chunks_exact(2)
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
            // Bit-exact bulk path for native f16 weights.
            Dtype::F16 => unsafe { bulk_copy_pod::<half::f16>(&self.data, 2) },
            Dtype::F32 => Ok(self
                .data
                .par_chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .map(half::f16::from_f32)
                .collect()),
            // Same lossy path as before (bf16→f32→f16); parallel only — bit-identical per element.
            Dtype::BF16 => Ok(self
                .data
                .par_chunks_exact(2)
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
            // Bit-exact bulk memcpy — dominant GPU load path for this checkpoint.
            Dtype::BF16 => unsafe { bulk_copy_pod::<half::bf16>(&self.data, 2) },
            Dtype::F32 => Ok(self
                .data
                .par_chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .map(half::bf16::from_f32)
                .collect()),
            Dtype::F16 => Ok(self
                .data
                .par_chunks_exact(2)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_bulk_copy_is_bit_exact() {
        let vals: Vec<u16> = (0u16..64).map(|i| 0x3F80 + i).collect(); // arbitrary bf16 bit patterns
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = RawTensor {
            data: Bytes::from(bytes),
            shape: vec![vals.len()],
            dtype: Dtype::BF16,
        };
        let out = t.to_bf16_vec().unwrap();
        assert_eq!(out.len(), vals.len());
        for (a, &b) in out.iter().zip(vals.iter()) {
            assert_eq!(a.to_bits(), b);
        }
    }

    #[test]
    fn f32_bulk_copy_is_bit_exact() {
        let vals: Vec<f32> = (0..32).map(|i| i as f32 * 0.125).collect();
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = RawTensor {
            data: Bytes::from(bytes),
            shape: vec![vals.len()],
            dtype: Dtype::F32,
        };
        let out = t.to_f32_vec().unwrap();
        assert_eq!(out, vals);
    }
}
