//! Preprocessing mirror of `processing_moss_transcribe_diarize.py`:
//!  - 30s chunking of raw audio
//!  - per-chunk mel (via mel::MelExtractor) padded to n_samples
//!  - audio token length math (WHISPER_ENCODER_STRIDE=2)
//!  - time-marker audio span (`_audio_span_ids`)
//!  - chat-template prompt building + padding
//!
//! Output: input_ids, attention_mask, mel input_features, audio_feature_lengths,
//! audio_chunk_mapping — exactly what the model `forward` consumes.

use crate::config::ProcessorConfig;
use crate::mel::MelExtractor;

pub const WHISPER_ENCODER_STRIDE: usize = 2;

/// Per the Python `_compute_audio_token_length`:
///   stride = hop_length * WHISPER_ENCODER_STRIDE * audio_merge_size
///   token_len = (num_samples - 1) // stride + 1
pub fn compute_audio_token_length(num_samples: usize, pc: &ProcessorConfig) -> usize {
    let stride = pc.hop_length * WHISPER_ENCODER_STRIDE * pc.audio_merge_size;
    (num_samples - 1) / stride + 1
}

/// 30s chunking: split audio into consecutive n_samples chunks (last padded).
/// Returns (chunks, token_lengths) where each chunk is n_samples long.
pub fn chunk_audio(audio: &[f32], pc: &ProcessorConfig) -> (Vec<Vec<f32>>, Vec<usize>) {
    let n_samples = pc.n_samples;
    let mut chunks = Vec::new();
    let mut token_lengths = Vec::new();
    let mut start = 0;
    while start < audio.len() {
        let end = (start + n_samples).min(audio.len());
        let chunk_len = end - start;
        token_lengths.push(compute_audio_token_length(chunk_len, pc));
        // pad to n_samples with zeros
        let mut chunk = vec![0.0f32; n_samples];
        chunk[..chunk_len].copy_from_slice(&audio[start..end]);
        chunks.push(chunk);
        start += n_samples;
    }
    if chunks.is_empty() {
        // degenerate empty audio → one zero chunk
        token_lengths.push(compute_audio_token_length(n_samples, pc));
        chunks.push(vec![0.0f32; n_samples]);
    }
    (chunks, token_lengths)
}

/// Build the per-chunk mel + token lengths + chunk mapping (one per audio).
/// Returns (input_features [n_chunks, num_mel, nb_max_frames], feature_lengths, chunk_mapping).
pub fn audios_to_input_features(
    mel: &MelExtractor,
    audios: &[Vec<f32>],
    pc: &ProcessorConfig,
) -> (Vec<f32>, usize, usize, Vec<usize>, Vec<usize>) {
    // input_features layout: [n_chunks, num_mel, nb_max_frames]
    let num_mel = mel.mel_bins();
    let nb_frames = pc.nb_max_frames;
    let mut feature_batches: Vec<Vec<f32>> = Vec::new();
    let mut feature_lengths: Vec<usize> = Vec::new();
    let mut chunk_mapping: Vec<usize> = Vec::new();

    for (audio_idx, audio) in audios.iter().enumerate() {
        let (chunks, token_lengths) = chunk_audio(audio, pc);
        for (ci, chunk) in chunks.iter().enumerate() {
            // extract mel → [num_mel, n_frames]
            let (mel_data, _n_mels, n_frames) = mel.extract(chunk).expect("mel extract failed");
            // pad/truncate to nb_max_frames, store row-major [num_mel, nb_frames]
            let mut padded = vec![0.0f32; num_mel * nb_frames];
            for m in 0..num_mel {
                for f in 0..n_frames.min(nb_frames) {
                    padded[m * nb_frames + f] = mel_data[m * n_frames + f];
                }
            }
            feature_batches.push(padded);
            feature_lengths.push(token_lengths[ci]);
            chunk_mapping.push(audio_idx);
        }
    }

    let n_chunks = feature_batches.len();
    let mut input_features = vec![0.0f32; n_chunks * num_mel * nb_frames];
    for (ci, feat) in feature_batches.iter().enumerate() {
        input_features[ci * num_mel * nb_frames..(ci + 1) * num_mel * nb_frames].copy_from_slice(feat);
    }
    (input_features, n_chunks, num_mel, feature_lengths, chunk_mapping)
}

/// Build the audio-span token ids with optional time markers (mirror of `_audio_span_ids`).
/// Produces a vector of length `audio_seq_len` filled mostly with audio_token_id,
/// but with digit tokens inserted at positions marking each `time_marker_every_seconds`.
pub fn audio_span_ids(audio_seq_len: usize, audio_token_id: i64,
                      digit_token_ids: &[i64; 10], pc: &ProcessorConfig) -> Vec<i64> {
    if !pc.enable_time_marker || audio_seq_len == 0 || pc.time_marker_every_seconds == 0 {
        return vec![audio_token_id; audio_seq_len];
    }
    let tokens_per_marker =
        (pc.audio_tokens_per_second * pc.time_marker_every_seconds as f64) as usize;
    if tokens_per_marker == 0 {
        return vec![audio_token_id; audio_seq_len];
    }
    let duration = audio_seq_len as f64 / pc.audio_tokens_per_second;
    let mut output: Vec<i64> = Vec::with_capacity(audio_seq_len);
    let mut consumed = 0usize;
    let mut sec = pc.time_marker_every_seconds;
    while sec <= duration as usize {
        let pos = (sec / pc.time_marker_every_seconds) * tokens_per_marker;
        let segment_len = pos.saturating_sub(consumed);
        if segment_len > 0 {
            output.extend(std::iter::repeat(audio_token_id).take(segment_len));
            consumed += segment_len;
        }
        // marker digits for `sec`
        for ch in sec.to_string().chars() {
            let d = ch.to_digit(10).expect("digit") as usize;
            output.push(digit_token_ids[d]);
        }
        sec += pc.time_marker_every_seconds;
    }
    let remainder = audio_seq_len.saturating_sub(consumed);
    if remainder > 0 {
        output.extend(std::iter::repeat(audio_token_id).take(remainder));
    }
    // Note: the Python original does NOT clamp to audio_seq_len — when time markers
    // are present the actual span length can exceed it slightly (markers occupy
    // extra positions). Return the full output to match the reference exactly.
    output
}
