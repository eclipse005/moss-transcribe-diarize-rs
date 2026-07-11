//! Chunking / transcript helpers (long-form pipeline scaffolding).
#![allow(dead_code)]

use anyhow::{anyhow, Result};
use regex::Regex;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptPlan {
    pub audio: String,
    pub model: String,
    pub mode: String,
    pub lang: String,
    pub prompt: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SilenceGap {
    pub start: f64,
    pub end: f64,
    pub dur_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlannedChunk {
    pub index: usize,
    pub start: f64,
    pub end: f64,
    pub cut_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AsrWindow {
    pub index: usize,
    pub plan_start: f64,
    pub plan_end: f64,
    pub audio_start: f64,
    pub audio_end: f64,
    pub overlap_sec: f64,
    pub pad_left: f64,
    pub pad_right: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Segment {
    pub id: String,
    pub start: f64,
    pub end: f64,
    pub speaker: String,
    pub text: String,
    pub chunk: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionResult {
    pub raw_transcript: String,
    pub plain_text: String,
    pub segments: Vec<Segment>,
    pub chunks: Vec<PlannedChunk>,
    pub asr_windows: Vec<AsrWindow>,
    pub duration_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptPlanFull {
    pub audio: String,
    pub model: String,
    pub mode: String,
    pub lang: String,
    pub prompt: String,
    pub duration_sec: f64,
    pub chunk_sec: f64,
    pub overlap_sec: f64,
    pub min_silence_ms: f64,
    pub min_silence_fallback_ms: f64,
    pub silence_lookback_sec: f64,
    pub chunks: Vec<PlannedChunk>,
    pub asr_windows: Vec<AsrWindow>,
    pub status: String,
}

pub fn load_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let chans = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            let raw: Result<Vec<f32>, _> = reader.samples::<f32>().collect();
            raw?
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample.max(1) as u32;
            // Match Python PyAV path: int / 32768.0 for 16-bit, no clamp.
            // The divisor is 1<<(bits-1) to map the signed range to [-1, ~1).
            let max = ((1i64 << (bits.saturating_sub(1))) - 1) as f32;
            let raw: Result<Vec<i32>, _> = reader.samples::<i32>().collect();
            raw?.into_iter().map(|s| s as f32 / max).collect()
        }
    };
    let mono = if chans == 1 { samples } else { samples.chunks(chans).map(|frame| frame.iter().copied().sum::<f32>() / chans as f32).collect() };
    Ok((mono, sr))
}

pub fn duration_sec(samples: &[f32], sr: u32) -> f64 { samples.len() as f64 / sr as f64 }

pub fn build_prompt(mode: &str, lang: &str, hotwords: Option<&str>, prompt_override: Option<&str>) -> Result<String> {
    if let Some(prompt) = prompt_override { return Ok(prompt.trim().to_string()); }
    let prompt = match (mode, lang) {
        ("default", "zh") => "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。",
        ("default", "en") => "Transcribe the audio. For each segment, start with the timestamp and speaker ID ([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp.",
        ("speaker", "zh") => "转录为文本，使用 [S01] [S02] [S03]等说话人标签。",
        ("speaker", "en") => "Transcribe the audio as text using speaker labels such as [S01], [S02], and [S03].",
        ("hotword", "zh") => {
            let hw = hotwords.ok_or_else(|| anyhow!("--mode hotword requires --hotwords"))?;
            return Ok(format!("请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。热词提示：{}", hw.trim()));
        }
        ("hotword", "en") => {
            let hw = hotwords.ok_or_else(|| anyhow!("--mode hotword requires --hotwords"))?;
            return Ok(format!("Transcribe the audio. For each segment, start with the timestamp and speaker ID ([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp. Hotwords: {}", hw.trim()));
        }
        _ => return Err(anyhow!("unknown mode/lang: {}/{}", mode, lang)),
    };
    Ok(prompt.to_string())
}

pub fn strip_tags(text: &str) -> String {
    let re_speaker = Regex::new(r"\[S\d+\]").unwrap();
    let re_ts = Regex::new(r"\[\d+(?:\.\d+)?\]").unwrap();
    re_ts.replace_all(&re_speaker.replace_all(text, ""), "").replace("  ", " ").trim().to_string()
}

pub fn shift_timestamps(text: &str, offset: f64) -> String {
    if offset.abs() < 1e-9 { return text.to_string(); }
    let ts = Regex::new(r"\[(\d+(?:\.\d+)?)\]").unwrap();
    ts.replace_all(text, |caps: &regex::Captures| {
        let t: f64 = caps[1].parse().unwrap_or(0.0);
        format!("[{:.2}]", t + offset)
    }).to_string()
}

pub fn segments_to_raw_transcript(segments: &[Segment]) -> String {
    let mut out = String::new();
    for seg in segments {
        let speaker = if seg.speaker.starts_with("[S") && seg.speaker.ends_with(']') {
            seg.speaker.trim_matches(['[', ']'].as_ref()).to_string()
        } else if seg.speaker.starts_with('S') {
            seg.speaker.clone()
        } else {
            "S00".to_string()
        };
        out.push_str(&format!("[{:.2}][{}]{}[{:.2}]", seg.start, speaker, seg.text, seg.end));
    }
    out
}

pub fn plan_chunks(duration: f64, chunk_sec: f64) -> Vec<PlannedChunk> {
    if duration <= chunk_sec + 1e-3 {
        return vec![PlannedChunk { index: 0, start: 0.0, end: duration, cut_reason: "short".to_string() }];
    }
    let mut out = vec![];
    let mut start = 0.0_f64;
    let mut idx = 0usize;
    while start < duration - 1e-3 {
        let end = (start + chunk_sec).min(duration);
        out.push(PlannedChunk { index: idx, start: (start * 1000.0).round() / 1000.0, end: (end * 1000.0).round() / 1000.0, cut_reason: if idx == 0 { "start".to_string() } else { "hard".to_string() } });
        idx += 1;
        start = end;
    }
    out
}

pub fn expand_chunks_with_overlap(chunks: &[PlannedChunk], duration: f64, overlap_sec: f64) -> Vec<AsrWindow> {
    if chunks.is_empty() { return vec![]; }
    if overlap_sec <= 1e-6 || chunks.len() == 1 {
        return chunks.iter().map(|c| AsrWindow { index: c.index, plan_start: c.start, plan_end: c.end, audio_start: c.start, audio_end: c.end, overlap_sec: 0.0, pad_left: 0.0, pad_right: 0.0 }).collect();
    }
    let half = overlap_sec / 2.0;
    let mut out = Vec::with_capacity(chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        let plan_len = (c.end - c.start).max(0.0);
        let side = half.min((plan_len * 0.25).max(0.5));
        let audio_start = if i == 0 { c.start } else { (c.start - side).max(0.0) };
        let audio_end = if i + 1 == chunks.len() { c.end } else { (c.end + side).min(duration) };
        out.push(AsrWindow { index: c.index, plan_start: c.start, plan_end: c.end, audio_start, audio_end, overlap_sec, pad_left: (c.start - audio_start).max(0.0), pad_right: (audio_end - c.end).max(0.0) });
    }
    out
}

pub fn dummy_transcribe(plan: &TranscriptPlanFull) -> TranscriptionResult {
    let segments = plan
        .chunks
        .iter()
        .map(|c| Segment {
            id: format!("seg_{:04}", c.index + 1),
            start: c.start,
            end: c.end,
            speaker: "S00".to_string(),
            text: String::new(),
            chunk: Some(c.index),
        })
        .collect::<Vec<_>>();
    let raw_transcript = segments_to_raw_transcript(&segments);
    let plain_text = strip_tags(&raw_transcript);
    TranscriptionResult {
        raw_transcript,
        plain_text,
        segments,
        chunks: plan.chunks.clone(),
        asr_windows: plan.asr_windows.clone(),
        duration_sec: plan.duration_sec,
    }
}
