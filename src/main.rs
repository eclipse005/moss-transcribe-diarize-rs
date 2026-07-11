use std::io::{self, Write};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use moss_transcribe_diarize_rs::{AsrError, AsrInference, Backend};

#[derive(Parser, Debug)]
#[command(name = "moss-transcribe-diarize-rs")]
#[command(about = "Rust + CUDA port of MOSS-Transcribe-Diarize ASR")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Transcribe a single WAV file.
    Transcribe(TranscribeArgs),
}

#[derive(Parser, Debug, Clone)]
struct TranscribeArgs {
    #[arg(value_name = "AUDIO")]
    audio: PathBuf,

    /// Model directory (config.json + safetensors + tokenizer).
    /// Defaults to `MOSS_MODEL_DIR` when set; otherwise required.
    #[arg(long, env = "MOSS_MODEL_DIR")]
    model: Option<PathBuf>,

    /// Compute backend: auto | cpu | cuda | gpu (default: cpu).
    #[arg(long, default_value = "cpu", value_parser = parse_backend)]
    backend: Backend,

    #[arg(long, default_value_t = 2048)]
    max_new_tokens: usize,

    /// Override the instruction prompt (default = official English diarize prompt).
    #[arg(long)]
    prompt: Option<String>,

    /// Stream decoded text to stdout as tokens are generated (same bytes as final text; no extra newlines).
    #[arg(long, default_value_t = false)]
    stream: bool,
}

fn parse_backend(s: &str) -> std::result::Result<Backend, String> {
    Backend::from_str(s).map_err(|e: AsrError| e.to_string())
}

fn main() -> Result<()> {
    // Logs on stderr so --stream can own stdout cleanly.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Stderr)
        .init();
    let cli = Cli::parse();
    match cli.command {
        Commands::Transcribe(args) => {
            let model = args.model.ok_or_else(|| {
                anyhow::anyhow!("model path required: pass --model PATH or set MOSS_MODEL_DIR")
            })?;
            if !model.is_dir() {
                bail!("model path is not a directory: {}", model.display());
            }
            let infer = AsrInference::load_with(&model, args.backend)
                .with_context(|| format!("load model from {}", model.display()))?;
            let prompt = args.prompt.unwrap_or_else(default_en_prompt);

            if args.stream {
                let mut out = io::stdout().lock();
                let mut on_delta = |delta: &str| {
                    let _ = out.write_all(delta.as_bytes());
                    let _ = out.flush();
                };
                let text = infer
                    .transcribe(
                        &args.audio,
                        &prompt,
                        args.max_new_tokens,
                        Some(&mut on_delta),
                    )
                    .with_context(|| format!("transcribe {}", args.audio.display()))?;
                // Match non-stream CLI: one trailing newline after the body.
                let _ = writeln!(out);
                let _ = text; // body already written via deltas
            } else {
                let text = infer
                    .transcribe(&args.audio, &prompt, args.max_new_tokens, None)
                    .with_context(|| format!("transcribe {}", args.audio.display()))?;
                println!("{}", text);
            }
        }
    }
    Ok(())
}

fn default_en_prompt() -> String {
    "Transcribe the audio. For each segment, start with the timestamp and speaker ID ([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp.".to_string()
}
