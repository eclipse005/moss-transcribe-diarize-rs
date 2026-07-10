use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use moss_transcribe_diarize_rs::AsrInference;

mod transcript;

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

    #[arg(long, default_value = r"D:\MOSS-Transcribe-Diarize\pretrained\moss-transcribe-diarize")]
    model: PathBuf,

    /// Backend: auto | cpu | cuda (cpu is the Phase-1 alignment reference)
    #[arg(long, default_value = "cpu")]
    backend: String,

    #[arg(long, default_value_t = 2048)]
    max_new_tokens: usize,

    /// Override the instruction prompt (default = official English transcription prompt).
    #[arg(long)]
    prompt: Option<String>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match cli.command {
        Commands::Transcribe(args) => {
            let infer = AsrInference::load_with_backend(&args.model, &args.backend)?;
            let prompt = args.prompt.unwrap_or_else(default_en_prompt);
            let text = infer.transcribe(
                args.audio.to_str().unwrap(),
                &prompt,
                args.max_new_tokens,
            )?;
            println!("{}", text);
        }
    }
    Ok(())
}

fn default_en_prompt() -> String {
    "Transcribe the audio. For each segment, start with the timestamp and speaker ID ([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp.".to_string()
}
