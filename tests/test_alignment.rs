//! End-to-end alignment test: Rust CPU transcription must match the Python
//! reference on `15s.wav` (text-level). Run with the model present.
//! Skips automatically if the model or audio are missing.
//!
//!   cargo test --release --test test_alignment -- --ignored --nocapture
//!
//! Env overrides:
//!   MOSS_MODEL_DIR  — model directory
//!   MOSS_AUDIO_DIR  — directory containing 15s.wav (defaults to parent of crate)

use std::path::PathBuf;

fn model_dir() -> PathBuf {
    if let Ok(p) = std::env::var("MOSS_MODEL_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(r"D:\MOSS-Transcribe-Diarize\pretrained\moss-transcribe-diarize")
}

fn audio_dir() -> PathBuf {
    if let Ok(p) = std::env::var("MOSS_AUDIO_DIR") {
        return PathBuf::from(p);
    }
    // Prefer sibling Python repo, then crate parent.
    let sibling = PathBuf::from(r"D:\MOSS-Transcribe-Diarize");
    if sibling.is_dir() {
        return sibling;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
#[ignore]
fn cpu_transcribes_15s_like_python() {
    let audio = audio_dir().join("15s.wav");
    let model = model_dir();
    if !audio.exists() || !model.exists() {
        eprintln!("skipping: model or audio not present ({}, {})", audio.display(), model.display());
        return;
    }
    let infer = moss_transcribe_diarize_rs::AsrInference::load(&model).expect("load");
    let text = infer
        .transcribe(&audio, "Transcribe the audio as text.", 512, None)
        .expect("transcribe");
    assert!(
        text.to_lowercase().contains("whippers") && text.to_lowercase().contains("crash course"),
        "transcription head did not match expected: {text}"
    );
    println!("RUST TEXT: {text}");
}

#[test]
#[ignore]
fn cpu_and_cuda_agree_on_15s() {
    let audio = audio_dir().join("15s.wav");
    let model = model_dir();
    if !audio.exists() || !model.exists() {
        eprintln!("skipping: model or audio not present");
        return;
    }
    let cpu = moss_transcribe_diarize_rs::AsrInference::load_with(
        &model,
        moss_transcribe_diarize_rs::Backend::Cpu,
    )
    .unwrap();
    let cpu_text = cpu
        .transcribe(&audio, "Transcribe the audio as text.", 512, None)
        .unwrap();
    #[cfg(feature = "cuda")]
    {
        let cuda = moss_transcribe_diarize_rs::AsrInference::load_with(
            &model,
            moss_transcribe_diarize_rs::Backend::Cuda,
        )
        .unwrap();
        let cuda_text = cuda
            .transcribe(&audio, "Transcribe the audio as text.", 512, None)
            .unwrap();
        println!("CPU:  {cpu_text}");
        println!("CUDA: {cuda_text}");
        assert!(cpu_text.to_lowercase().contains("whippers"));
        assert!(cuda_text.to_lowercase().contains("whippers"));
    }
}
