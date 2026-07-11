use anyhow::Result;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

pub(crate) const MEL_SAMPLE_RATE: u32 = 16000;
pub(crate) const N_FFT: usize = 400;
pub(crate) const HOP_LENGTH: usize = 160;

fn hann_window(n: usize) -> Vec<f32> {
    // Periodic Hann window (2πi/N), matching Python's window_function(n, "hann")
    // which is periodic — NOT symmetric (2πi/(N-1)). Using symmetric was a source
    // of mel divergence vs the Python reference.
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

fn reflection_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    let mut padded = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad.min(n - 1)).rev() {
        padded.push(signal[i]);
    }
    padded.extend_from_slice(signal);
    let right_start = if n > pad { n - pad - 1 } else { 0 };
    for i in (right_start..n - 1).rev() {
        padded.push(signal[i]);
    }
    padded
}

fn compute_power_stft(
    signal: &[f32],
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
) -> (Vec<f32>, usize, usize) {
    let n_freqs = n_fft / 2 + 1;
    let n_frames = if signal.len() >= n_fft {
        (signal.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    // Use f64 for the FFT, matching Python's np.fft which promotes to float64
    // internally. f32 FFT accumulates enough twiddle-factor error to diverge
    // in the power spectrum (max-abs ~0.8 in the resulting mel), which then
    // shifts rare timestamp tokens vs the Python reference.
    //
    // The FFT plan is Sync (read-only after creation), so frames are computed
    // in parallel via rayon — each thread gets its own scratch frame buffer.
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let window_f64: Vec<f64> = window.iter().map(|&v| v as f64).collect();

    // power is column-major [n_freqs, n_frames]. We compute per-frame into a
    // temporary row-major buffer, then scatter into the column-major layout.
    // Parallelize across frames (each thread gets its own frame_buf).
    let frame_results: Vec<Vec<f32>> = (0..n_frames)
        .into_par_iter()
        .map(|i| {
            let start = i * hop_length;
            let mut frame_buf = vec![Complex::new(0.0f64, 0.0f64); n_fft];
            for j in 0..n_fft {
                frame_buf[j] = Complex::new(signal[start + j] as f64 * window_f64[j], 0.0);
            }
            fft.process(&mut frame_buf);
            let mut row = vec![0.0f32; n_freqs];
            for k in 0..n_freqs {
                let re = frame_buf[k].re;
                let im = frame_buf[k].im;
                row[k] = (re * re + im * im) as f32;
            }
            row
        })
        .collect();

    let mut power = vec![0.0f32; n_freqs * n_frames];
    for (i, row) in frame_results.iter().enumerate() {
        for k in 0..n_freqs {
            power[k * n_frames + i] = row[k];
        }
    }

    (power, n_freqs, n_frames)
}

fn create_mel_filterbank(
    num_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    fmin: f64,
    fmax: f64,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let sr = sample_rate as f64;

    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;

    let hz_to_mel = |f: f64| -> f64 {
        if f < min_log_hz {
            f / f_sp
        } else {
            min_log_mel + (f / min_log_hz).ln() / logstep
        }
    };

    let mel_to_hz = |m: f64| -> f64 {
        if m < min_log_mel {
            f_sp * m
        } else {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let filter_freqs: Vec<f64> = (0..num_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (num_mels + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    let all_freqs: Vec<f64> = (0..n_freqs).map(|j| j as f64 * sr / n_fft as f64).collect();
    let f_diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    let mut filters = vec![0.0f32; num_mels * n_freqs];

    for j in 0..n_freqs {
        for i in 0..num_mels {
            let down = (all_freqs[j] - filter_freqs[i]) / f_diff[i];
            let up = (filter_freqs[i + 2] - all_freqs[j]) / f_diff[i + 1];
            let val = down.min(up).max(0.0);
            filters[i * n_freqs + j] = val as f32;
        }
    }

    for i in 0..num_mels {
        let enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);
        for j in 0..n_freqs {
            filters[i * n_freqs + j] *= enorm as f32;
        }
    }

    filters
}

pub struct MelExtractor {
    n_fft: usize,
    hop_length: usize,
    num_mel_bins: usize,
    mel_filters: Vec<f32>,
    n_freqs: usize,
}

impl MelExtractor {
    pub fn new(
        n_fft: usize,
        hop_length: usize,
        num_mel_bins: usize,
        sample_rate: u32,
    ) -> Self {
        let n_freqs = n_fft / 2 + 1;
        let mel_filters = create_mel_filterbank(
            num_mel_bins,
            n_fft,
            sample_rate,
            0.0,
            sample_rate as f64 / 2.0,
        );
        Self {
            n_fft,
            hop_length,
            num_mel_bins,
            mel_filters,
            n_freqs,
        }
    }

    /// Build with a pre-computed mel filterbank, laid out row-major as
    /// `[num_mel_bins, n_freqs]` (one filter per row). This lets us load the
    /// exact same filterbank the Python WhisperFeatureExtractor uses, so the
    /// mel step is bit-identical (the generated filterbank is the main source
    /// of mel divergence vs HF's slaney implementation).
    pub fn new_with_filters(
        n_fft: usize,
        hop_length: usize,
        num_mel_bins: usize,
        mel_filters: Vec<f32>,
    ) -> Self {
        let n_freqs = n_fft / 2 + 1;
        assert_eq!(mel_filters.len(), num_mel_bins * n_freqs,
            "mel_filters must be [num_mel_bins={}, n_freqs={}] = {} elems, got {}",
            num_mel_bins, n_freqs, num_mel_bins * n_freqs, mel_filters.len());
        Self { n_fft, hop_length, num_mel_bins, mel_filters, n_freqs }
    }

    pub fn mel_bins(&self) -> usize { self.num_mel_bins }

    pub(crate) fn extract(&self, samples: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        // Do NOT zero-pad to a hop_length multiple first — Python's spectrogram()
        // reflects the raw waveform and then drops any trailing partial frame
        // via num_frames = 1 + floor((len - frame_length)/hop). Zero-padding
        // before reflection changes the boundary samples and was a source of
        // mel divergence vs the Python reference.
        let pad = self.n_fft / 2;
        let padded_signal = reflection_pad(samples, pad);

        let window = hann_window(self.n_fft);

        let (power, _n_freqs, n_frames_with_last) =
            compute_power_stft(&padded_signal, self.n_fft, self.hop_length, &window);

        // DIAG: dump raw power spectrum [n_freqs, n_frames_with_last] for alignment comparison
        if let Ok(dir) = std::env::var("MOSS_POWER_DUMP") {
            let _ = std::fs::create_dir_all(&dir);
            use std::io::Write;
            if let Ok(mut f) = std::fs::File::create(std::path::Path::new(&dir).join("rust_power.bin")) {
                let bytes: Vec<u8> = power.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = f.write_all(&bytes);
            }
        }

        let n_frames = if n_frames_with_last > 0 {
            n_frames_with_last - 1
        } else {
            0
        };

        let mut mel_spec = vec![0.0f32; self.num_mel_bins * n_frames];
        for m in 0..self.num_mel_bins {
            let filter_row = &self.mel_filters[m * self.n_freqs..(m + 1) * self.n_freqs];
            let out_row = &mut mel_spec[m * n_frames..(m + 1) * n_frames];
            for f in 0..self.n_freqs {
                let w = filter_row[f];
                if w == 0.0 { continue; }
                let power_row = &power[f * n_frames_with_last..f * n_frames_with_last + n_frames];
                for (t, &p) in power_row.iter().enumerate() {
                    out_row[t] += w * p;
                }
            }
        }

        let log10_factor = 1.0 / 10.0f32.ln();
        let mut max_val = f32::NEG_INFINITY;
        for v in mel_spec.iter_mut() {
            *v = v.max(1e-10f32).ln() * log10_factor;
            if *v > max_val { max_val = *v; }
        }

        let min_val = max_val - 8.0;
        for v in mel_spec.iter_mut() {
            *v = (v.max(min_val) + 4.0) / 4.0;
        }

        Ok((mel_spec, self.num_mel_bins, n_frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_edges() {
        let w = hann_window(5);
        assert_eq!(w.len(), 5);
        assert!(w[0].abs() < 1e-6);
        assert!(w[4].abs() < 1e-6);
        assert!((w[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reflection_pad_basic() {
        let signal = vec![1.0f32, 2.0, 3.0];
        let padded = reflection_pad(&signal, 1);
        assert_eq!(padded, vec![2.0f32, 1.0, 2.0, 3.0, 2.0]);
    }

    #[test]
    fn test_mel_filterbank_shape_and_nonneg() {
        let num_mels = 128;
        let n_fft = 400;
        let sr = 16000u32;
        let n_freqs = n_fft / 2 + 1;
        let filters = create_mel_filterbank(num_mels, n_fft, sr, 0.0, sr as f64 / 2.0);
        assert_eq!(filters.len(), num_mels * n_freqs);
        assert!(filters.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_mel_extractor_silent_signal() {
        let samples = vec![0.0f32; 16000];
        let extractor = MelExtractor::new(400, 160, 128, 16000);
        let (mel, n_mels, n_frames) = extractor.extract(&samples).unwrap();
        assert_eq!(n_mels, 128);
        assert!(n_frames > 0);
        assert_eq!(mel.len(), n_mels * n_frames);
        assert!(mel.iter().all(|v| v.is_finite()));
    }
}

pub fn load_audio_wav(path: &str, target_sr: u32) -> crate::Result<Vec<f32>> {
    load_audio_wav_impl(path, target_sr).map_err(crate::error::AsrError::AudioDecode)
}

fn load_audio_wav_impl(path: &str, target_sr: u32) -> anyhow::Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples_f32: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("WAV read error: {}", e))?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i64 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("WAV read error: {}", e))?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    let mono: Vec<f32> = if channels == 1 {
        samples_f32
    } else {
        samples_f32
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    if sr == target_sr {
        return Ok(mono);
    }

    // Use the soxr_hq-compatible polyphase Kaiser resampler (matches librosa's
    // soxr_hq output for node-level alignment with the Python reference).
    // rubato's sinc resampler produced different samples → mel divergence.
    Ok(crate::resampler::resample_mono(&mono, sr, target_sr))
}
