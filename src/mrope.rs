//! Standard 1-D Rotary Position Embedding for the Qwen3-0.6B text decoder.
//!
//! Replaces the reference qwen3-asr-rs MRoPE: this model has no `rope_scaling`,
//! so every head-dim index `j` uses the single scalar position `pos[t]` (not a
//! 3-D t/h/w stream). The **half-mirrored** table layout is preserved because
//! both `apply_rotary_row` (cpu_engine.rs) and `rotary_emb_f16` (kernels.cu)
//! read `cos[i]` and `cos[i+half]` separately and require them to be equal.
//!
//! `compute_rope_cos_sin` returns `(cos, sin)` each of shape `[seq_len, head_dim]`,
//! with rows `cos[t, j] == cos[t, j + half]` for `j < half` (mirroring MRoPE).

/// Build standard RoPE cosine/sine tables.
/// Returns `(cos, sin)` each `[seq_len, head_dim]` with the half-mirrored layout.
pub(crate) fn compute_rope_cos_sin(pos: &[i64], head_dim: usize, rope_theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let sl = pos.len();
    // inv_freq[i] = 1 / theta^(2i / head_dim),  i in [0, half)
    let inv: Vec<f64> = (0..half)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();
    let mut cos = vec![0.0f32; sl * head_dim];
    let mut sin = vec![0.0f32; sl * head_dim];
    for t in 0..sl {
        let p = pos[t] as f64;
        for j in 0..half {
            let a = p * inv[j];
            let (c, s) = (a.cos(), a.sin());
            cos[t * head_dim + j] = c as f32;
            sin[t * head_dim + j] = s as f32;
            cos[t * head_dim + j + half] = c as f32;
            sin[t * head_dim + j + half] = s as f32;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_mirrored_layout() {
        let (cos, sin) = compute_rope_cos_sin(&[0, 1, 2], 8, 1_000_000.0);
        for t in 0..3 {
            for j in 0..4 {
                assert!((cos[t * 8 + j] - cos[t * 8 + j + 4]).abs() < 1e-6);
                assert!((sin[t * 8 + j] - sin[t * 8 + j + 4]).abs() < 1e-6);
            }
        }
        // position 0 → cos=1, sin=0
        for j in 0..8 {
            assert!((cos[j] - 1.0).abs() < 1e-6);
            assert!(sin[j].abs() < 1e-6);
        }
    }
}
