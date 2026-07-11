# moss-transcribe-diarize-rs

[MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) 的 Rust + 手写 CUDA 推理引擎。零深度学习框架依赖——矩阵乘法走 cuBLAS（GPU）或 `gemm`+`rayon`（CPU），所有逐元素运算都是手写 CUDA kernel 或 rayon 并行的 Rust 实现。

模型架构：Whisper-Medium 编码器 + VQAdaptor + Qwen3-0.6B 解码器，支持语音转写、说话人分离（diarization）、时间戳标注、热词提示。

## 特性

- **完全 GPU 推理**：编码器、VQAdaptor、解码器全在 GPU 上运行，bf16 精度
- **与 Python 原版 token 级对齐**：相同 prompt 下，转录文本、说话人标签、时间戳逐段一致
- **CPU + CUDA 双后端**：CUDA 不可用时自动回退到 CPU
- **多语言**：中文、英语、日语等
- **热词支持**：通过 prompt 注入热词，纠正专有名词识别
- **soxr 高质量重采样**：精确匹配 librosa 的 `soxr_hq`，支持 16/24/32/48kHz 输入

## 安装

```bash
# 需要 Rust 1.85+ (MSRV) 和 CUDA 12.8+（默认 feature）
cargo build --release          # CUDA 后端（默认 features）
cargo build --release --no-default-features  # 仅 CPU 后端
```

## 使用

### 命令行

```bash
# 模型路径：--model 或环境变量 MOSS_MODEL_DIR
export MOSS_MODEL_DIR=/path/to/moss-transcribe-diarize

# 基本转写（默认英语 prompt，带时间戳和说话人；默认 --backend cpu）
cargo run --release -- transcribe audio.wav --model "$MOSS_MODEL_DIR"

# 指定后端（auto | cpu | cuda | gpu；库侧 FromStr，CLI value_parser）
cargo run --release -- transcribe audio.wav --model "$MOSS_MODEL_DIR" --backend cuda
cargo run --release --no-default-features -- transcribe audio.wav --model "$MOSS_MODEL_DIR" --backend cpu

# 热词（通过 prompt 注入）
cargo run --release -- transcribe audio.wav --model "$MOSS_MODEL_DIR" --prompt "Transcribe the audio. For each segment, start with the timestamp and speaker ID ([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp. Hotwords: order block, swing trading"

# 性能分析
MOSS_BENCH=1 cargo run --release -- transcribe audio.wav --model "$MOSS_MODEL_DIR" --backend cuda
```

### 作为库

```rust
use moss_transcribe_diarize_rs::{AsrInference, Backend};

let infer = AsrInference::load_with(std::path::Path::new("path/to/model"), Backend::Cuda)?;
let text = infer.transcribe("audio.wav", &prompt, 2048)?;
println!("{}", text);
```

输出格式（紧凑时间戳）：

```
[0.13][S01]我现在在天海酒吧[1.37][1.41][S01]限你十分钟内到我面前[2.91][2.94][S02]十分钟 二十公里呢
```

每段：`[起始时间戳][说话人]正文[结束时间戳]`，说话人标签为 `[S01]`、`[S02]`、`[S03]` …

## 模型架构

```
16kHz WAV
  → soxr 重采样到 16kHz mono（如需要）           [resampler.rs]
  → mel 特征提取（80-bin, f64 FFT, Hann 窗）      [mel.rs]
  → 30s 分块                                     [processor.rs]
  → Whisper-Medium 编码器（24层, d_model=1024,    [gpu_whisper.rs]
       GELU, 学习位置编码, 双向注意力）
  → 按 token 长度裁剪 → 拼接
  → 4× 时间合并 (B,T,1024)→(B,T/4,4096)
  → VQAdaptor Linear(4096→1024)→SiLU→Linear→LayerNorm  [cudarc_engine.rs]
  → masked_scatter 注入 Qwen3 文本嵌入             [inference.rs]
  → Qwen3-0.6B 解码器（28层, GQA 16q/8kv, SwiGLU,  [cudarc_engine.rs]
       RMSNorm, Q/K-norm, 标准 RoPE θ=1e6）
  → 绑定的 lm_head → 贪心解码
```

## 模块说明

| 文件 | 职责 |
|------|------|
| `config.rs` | 模型配置解析（Whisper + Qwen3 子配置）|
| `mel.rs` | mel 特征提取 + WAV 音频加载 + soxr 重采样调度 |
| `resampler.rs` | soxr_hq 兼容的多相 Kaiser 重采样器 |
| `processor.rs` | 30s 分块、token 长度计算、时间标记音频跨度 |
| `whisper.rs` | CPU Whisper 编码器（对齐参考实现）|
| `gpu_whisper.rs` | GPU Whisper 编码器（conv1d + 注意力 + FFN）|
| `cudarc_engine.rs` | CUDA 引擎：cuBLAS 封装 + 全部 GPU kernel + GPU VQAdaptor + GPU 解码器 |
| `cpu_engine.rs` | CPU Qwen3 解码器（gemm + rayon）|
| `kernels/kernels.cu` | CUDA kernel（rms_norm, layer_norm, gelu, conv1d, RoPE, GQA, silu, …）|
| `mrope.rs` | 标准 RoPE cos/sin 表（半镜像布局）|
| `inference.rs` | 推理编排：加载 → 预处理 → 编码 → 散射 → 解码 |
| `weights.rs` | safetensors 权重加载（mmap 零拷贝）|
| `raw_tensor.rs` | safetensors 张量视图（支持 f32/f16/bf16）|

## 精度对齐

本项目的核心目标之一是与 Python 原版 **token 级对齐**。通过以下措施实现：

- **bf16 全链路**：权重加载、cuBLAS 计算、CUDA kernel 均使用 bfloat16（匹配 Python `dtype=bfloat16`）
- **f64 FFT**：mel 提取的 STFT 使用 float64 精度（匹配 NumPy `np.fft` 的内部行为）
- **Periodic Hann 窗**：使用 `2πi/N`（非 `2πi/(N-1)`），匹配 `transformers` 的 `window_function`
- **soxr 重采样**：精确复刻 librosa 的 `soxr_hq` 重采样算法
- **Python filterbank**：直接加载 HF `WhisperFeatureExtractor` 的 slaney mel filterbank

验证结果（与 Python 原版逐段对比）：

| 音频 | 语言 | 说话人 | 段数 | 精确匹配 |
|------|------|--------|------|---------|
| ja.wav (89s, 16kHz) | 日语 | 3 人 | 25 | **25/25** |
| 180s.wav (180s) | 中文 | 4 人 | 91 | **91/91** |
| 90s.wav (90s) | 英语 | 1 人 | 19 | **19/19** |

"精确匹配"指每段的说话人标签、起止时间戳、转录文本完全一致。

## 模型下载

从 HuggingFace 下载：

- [OpenMOSS/moss-transcribe-diarize](https://huggingface.co/OpenMOSS/moss-transcribe-diarize)

模型目录需包含 `config.json`、`model*.safetensors`、`tokenizer.json`、`chat_template.jinja` 等文件。

## License

MIT
