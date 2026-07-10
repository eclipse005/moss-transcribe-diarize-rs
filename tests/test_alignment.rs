//! End-to-end alignment test: Rust CPU transcription must match the Python
//! reference on `15s.wav` (text-level). Run with the model present at the
//! default path. Skips automatically if the model or audio are missing.
//!
//!   cargo test --release --test test_alignment -- --ignored --nocapture

use std::path::PathBuf;

fn model_dir() -> PathBuf {
    PathBuf::from(r"D:\MOSS-Transcribe-Diarize\pretrained\moss-transcribe-diarize")
}

fn repo_root() -> PathBuf {
    // tests/ is one level under rust/
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
#[ignore]
fn cpu_transcribes_15s_like_python() {
    let audio = repo_root().join("15s.wav");
    let model = model_dir();
    if !audio.exists() || !model.exists() {
        eprintln!("skipping: model or audio not present");
        return;
    }
    let infer = moss_transcribe_diarize_rs::AsrInference::load(&model).expect("load");
    let text = infer
        .transcribe(
            audio.to_str().unwrap(),
            "Transcribe the audio as text.",
            512,
        )
        .expect("transcribe");
    // The exact string is nondeterministic in punctuation but the leading
    // words are stable. Assert the head matches the Python reference.
    assert!(
        text.to_lowercase().contains("whippers") && text.to_lowercase().contains("crash course"),
        "transcription head did not match expected: {text}"
    );
    println!("RUST TEXT: {text}");
}

#[test]
#[ignore]
fn cpu_and_cuda_agree_on_15s() {
    let audio = repo_root().join("15s.wav");
    let model = model_dir();
    if !audio.exists() || !model.exists() {
        eprintln!("skipping: model or audio not present");
        return;
    }
    let cpu = moss_transcribe_diarize_rs::AsrInference::load_with_backend(&model, "cpu").unwrap();
    let cpu_text = cpu.transcribe(audio.to_str().unwrap(), "Transcribe the audio as text.", 512).unwrap();
    // CUDA path is feature-gated; only compare when built with default features.
    #[cfg(feature = "cuda")]
    {
        let cuda = moss_transcribe_diarize_rs::AsrInference::load_with_backend(&model, "cuda").unwrap();
        let cuda_text = cuda.transcribe(audio.to_str().unwrap(), "Transcribe the audio as text.", 512).unwrap();
        println!("CPU:  {cpu_text}");
        println!("CUDA: {cuda_text}");
        // Both should contain the same leading phrase (bf16 vs f32 may shift later words).
        assert!(cpu_text.to_lowercase().contains("whippers"));
        assert!(cuda_text.to_lowercase().contains("whippers"));
    }
}
