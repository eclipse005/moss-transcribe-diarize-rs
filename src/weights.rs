//! Safetensors weight loading — single-file and sharded index.
//!
//! Weights are mmap'd and shared across all tensors via `bytes::Bytes` O(1)
//! refcount+range slices, avoiding the previous `std::fs::read` (full file
//! into host RAM) + per-tensor `to_vec()` copy. On the 0.6B checkpoint this
//! drops `load_weights` from ~1.1s / ~3.6GB peak to mmap cost only.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::anyhow;
use bytes::Bytes;
use memmap2::Mmap;

use crate::raw_tensor::RawTensor;

pub(crate) fn load_weights(model_dir: &Path) -> anyhow::Result<HashMap<String, RawTensor>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let wm = idx["weight_map"].as_object().ok_or_else(|| anyhow!("invalid index.json"))?;
        let mut sf: std::collections::HashSet<String> = std::collections::HashSet::new();
        for v in wm.values() { if let Some(s) = v.as_str() { sf.insert(s.to_string()); } }
        let mut all = HashMap::new();
        for s in sf { all.extend(load_safetensors_file(&model_dir.join(&s))?); }
        return Ok(all);
    }
    load_safetensors_file(&model_dir.join("model.safetensors"))
}

fn load_safetensors_file(path: &Path) -> anyhow::Result<HashMap<String, RawTensor>> {
    let file = File::open(path).map_err(|e| anyhow!("open {:?}: {}", path, e))?;
    // SAFETY: the file is a read-only checkpoint we do not mutate while mapped.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| anyhow!("mmap {:?}: {}", path, e))?;
    #[cfg(unix)]
    {
        // Linux/macOS: pages will be demand-paged from disk on first touch;
        // Sequential hints the pager we read end-to-end. No-op on Windows.
        let _ = mmap.advise(memmap2::Advice::Sequential);
    }

    // Hand ownership of the mmap to Bytes; every tensor slice below keeps the
    // whole region alive via refcount. `from_owner` requires its owner to be
    // Send + 'static — Mmap satisfies both.
    let buf: Bytes = Bytes::from_owner(mmap);
    let st = safetensors::SafeTensors::deserialize(&buf)
        .map_err(|e| anyhow!("safetensors: {}", e))?;

    // Anchor: base address of the mmap region, to recover each tensor's byte
    // offset within `buf` for an O(1) slice (zero-copy).
    let base = buf.as_ptr() as usize;
    let mut weights = HashMap::with_capacity(st.len());
    for (name, view) in st.iter() {
        let view_data = view.data();
        let offset = view_data.as_ptr() as usize - base;
        let len = view_data.len();
        weights.insert(name.to_string(), RawTensor {
            data: buf.slice(offset..offset + len),
            shape: view.shape().to_vec(),
            dtype: view.dtype(),
        });
    }
    Ok(weights)
}
