# moss-transcribe-diarize-rs

Rust + hand-written CUDA inference port of [MOSS-Transcribe-Diarize](..), an ASR
+ speaker-diarization model (Whisper-Medium encoder + VQAdaptor + Qwen3-0.6B
decoder). Zero deep-learning-framework dependency — matmuls go through cuBLAS
(GPU) or `gemm`+`rayon` (CPU); every element-wise op is a hand-written CUDA
kernel or a rayon-parallel Rust routine.

> **Phase 1 status: node-by-node alignment with the Python original is proven.**
> The Rust output matches the Python reference text on `15s.wav`, `22s.wav`
> (Chinese), and `90s.wav` (multi-chunk) on both CPU and CUDA backends. CPU/GPU
> speedups (RTFx) are the next phase.

## Architecture

```
16kHz WAV
  → mel (80-bin, hop=160, n_fft=400, 30s chunks)            [mel.rs]
  → Whisper-Medium encoder (24 layers, d_model=1024, GELU,  [whisper.rs / gpu_whisper.rs]
       learned positional embedding, full bidirectional attn)
  → per-chunk trim to token_len*4 frames → concat per audio
  → 4× time-merge  (B,T,1024)→(B,T/4,4096)                  [adaptor.rs]
  → VQAdaptor  Linear(4096→1024)→SiLU→Linear→LayerNorm
  → masked_scatter into Qwen3 text embeddings                [inference.rs]
       at <|audio_pad|> positions (digit time-markers keep their own embeds)
  → Qwen3-0.6B decoder (28 layers, GQA 16q/8kv, SwiGLU,     [cpu_engine.rs / cudarc_engine.rs]
       RMSNorm, Q/K-norm, standard RoPE θ=1e6)
  → tied lm_head → greedy decode
```

This is a **different architecture** from the `qwen3-asr-rs` reference (Qwen3-ASR
uses an audio_tower + mRoPE thinker). The reusable parts of that reference are
the generic CUDA/CPU infrastructure (cuBLAS wrappers, weight loading, mel
extraction, RMSNorm/LayerNorm/GQA/RoPE kernels, KV cache); the Whisper encoder,
VQAdaptor, time-merge, and the standard-RoPE table builder are new.

## Build

```bash
# CUDA backend (default) — needs CUDA 12.8+
cargo build --release

# CPU-only backend
cargo build --release --no-default-features
```

## Usage

```bash
# CUDA
cargo run --release -- transcribe ../15s.wav --backend cuda --max-new-tokens 512

# CPU
cargo run --release --no-default-features -- transcribe ../15s.wav --backend cpu

# Custom prompt / model path
cargo run --release -- transcribe ../90s.wav --model /path/to/model --max-new-tokens 2048
```

`--backend auto` (default feature build) prefers CUDA and falls back to CPU.

## Alignment testing

The Python reference dumps intermediate tensors + text:

```bash
conda activate moss-transcribe-diarize
python rust/scripts/dump_reference.py 15s.wav --max-new-tokens 512
```

Rust node-dump: set `MOSS_DUMP=<dir>` to write `rust_input_ids.bin`,
`rust_audio_embeds.bin`, `rust_input_features.bin`. Verified on `15s.wav`:

- `input_ids` are **byte-exact** (222 tokens, identical to Python).
- `audio_embeds` match to max-abs-diff ≈ 1.2, mean ≈ 0.018 — the expected
  bf16-vs-f32 gap (Python runs the encoder in bf16; Rust runs f32).

End-to-end test:

```bash
cargo test --release --test test_alignment -- --ignored --nocapture
```

## Module map

| file | role |
|------|------|
| `config.rs` | `MossConfig` (Qwen3 + Whisper sub-configs) + `ProcessorConfig` |
| `mel.rs` | librosa-style 80-bin mel + WAV loader (copied from qwen3-asr-rs) |
| `processor.rs` | 30s chunking, token-length math, time-marker audio span |
| `whisper.rs` | CPU Whisper-Medium encoder (conv1d stem + 24 layers + final LN) |
| `adaptor.rs` | VQAdaptor + 4× time-merge |
| `mrope.rs` | standard 1-D RoPE cos/sin table (half-mirrored layout) |
| `cpu_engine.rs` | CPU Qwen3 decoder (gemm + rayon; copied from qwen3-asr-rs) |
| `cudarc_engine.rs` | CUDA engine: cuBLAS wrappers + all element-wise kernels |
| `kernels/kernels.cu` | CUDA kernels (rms_norm, layer_norm, gelu, conv1d, RoPE, GQA, …) |
| `gpu_whisper.rs` | GPU Whisper encoder (reuses cudarc kernels) |
| `inference.rs` | orchestration: load → preprocess → encode → scatter → decode |
| `weights.rs` | mmap safetensors loading (zero-copy) |

## Deferred (Phase 2+)

- Full-GPU decoder path (currently the Qwen3 decoder runs on CPU even with
  `--backend cuda`; the GPU encoder feeds a CPU time-merge + adaptor + decoder).
- CPU-specific speedups (INT8 weight quant, AVX2 GEMV — present in `cpu_engine.rs`
  but the Whisper path uses the f32 reference).
- Long-audio VAD chunking (`transcript.rs` scaffold) and the
  `[start][Sxx]text[end]` transcript parser.
- Video container decoding (PyAV) — WAV only for now.
