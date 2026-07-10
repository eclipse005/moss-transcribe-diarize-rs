//! High-quality audio resampler for matching librosa's `soxr_hq` output.
//!
//! - **Integer-ratio** decimation (e.g. 48 kHz → 16 kHz, M=3): polyphase
//!   Kaiser-windowed sinc.
//! - **Rational-ratio** resampling (e.g. 24 kHz → 16 kHz, L=2/M=3):
//!   zero-stuff + Kaiser-windowed sinc convolution + decimation.
//! - **Arbitrary ratios** (e.g. 44.1 kHz → 16 kHz): direct Kaiser-windowed
//!   sinc interpolation.

/// Resample a mono f32 signal to the target rate.
pub fn resample_mono(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }
    if source_rate > target_rate && source_rate % target_rate == 0 {
        return decimate_integer(samples, source_rate, target_rate);
    }
    // Try rational ratio L/M where source*L = target*M (e.g. 24k→16k: L=2,M=3)
    if source_rate > target_rate {
        if let Some((l, m)) = rational_ratio(source_rate, target_rate) {
            if l <= 16 && m <= 16 {
                return decimate_rational(samples, l, m);
            }
        }
    }
    resample_arbitrary(samples, source_rate, target_rate)
}

/// Find the smallest L/M such that source*L = target*M (both > 0).
fn rational_ratio(source_rate: u32, target_rate: u32) -> Option<(u32, u32)> {
    let g = gcd(source_rate, target_rate);
    let l = target_rate / g; // 16000/8000 = 2
    let m = source_rate / g; // 24000/8000 = 3
    if l > 0 && m > 0 {
        Some((l, m))
    } else {
        None
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Rational-ratio decimation by L/M (up-sample by L, low-pass, down-sample by M).
///
/// Two-step approach: zero-stuff by L, convolve with Kaiser-windowed sinc,
/// then pick every M-th sample.  This avoids polyphase phase-alignment issues
/// for even L and produces correct linear-phase output.
fn decimate_rational(samples: &[f32], l: u32, m: u32) -> Vec<f32> {
    let l = l as usize;
    let m = m as usize;

    // Design the anti-alias / anti-imaging filter at the up-sampled rate.
    let fn_val = l.max(m) as f64;
    let lin_to_db_2 = 6.020599913279624_f64;
    let rej = SOXR_PRECISION * lin_to_db_2;
    let passband_end = 1.0 - 0.05 / lsx_to_3db(rej);
    let fp0 = passband_end;
    let fs0 = SOXR_STOPBAND_BEGIN;
    let att = (SOXR_PRECISION + 1.0) * lin_to_db_2;

    let fp_n = fp0 / fn_val;
    let fs_n = fs0 / fn_val;
    let modulo = 4;

    let mut tr_bw = 0.5 * (fs_n - fp_n);
    tr_bw = tr_bw.min(0.5 * fs_n);
    let fc = fs_n - tr_bw;

    let beta = kaiser_beta(att);
    let att_for_taps = if att < 60.0 {
        (att - 7.95) / (2.285 * std::f64::consts::PI * 2.0)
    } else {
        ((0.0007528358 - 1.577737e-05 * beta) * beta + 0.6248022) * beta + 0.06186902
    };
    let num_taps = (att_for_taps / tr_bw + 1.0).ceil() as usize;
    let num_taps = ((num_taps + modulo - 2) / modulo) * modulo + 1;

    let rho = if att < 120.0 { 0.63 } else { 0.75 };
    // scale = L: DC gain = L, compensating the 1/L energy from zero-stuffing.
    let h = make_lpf(num_taps, fc, beta, rho, l as f64);

    let half = num_taps / 2;
    let half_i = half as i64;
    let output_len = (samples.len() * l) / m;
    let mut out = Vec::with_capacity(output_len);

    // Output i corresponds to up-sampled position pos = i * M.
    // The filter is centred at pos: convolution reads up[pos - half + k].
    // For pos < half, negative indices read as zero (natural preload).
    let mut pos = 0i64;
    for _ in 0..output_len {
        let mut acc = 0.0f64;
        for k in 0..num_taps {
            let up_idx = pos - half_i + k as i64;
            if up_idx >= 0 && up_idx % l as i64 == 0 {
                let s_idx = (up_idx / l as i64) as usize;
                if s_idx < samples.len() {
                    acc += h[k] * samples[s_idx] as f64;
                }
            }
        }
        out.push(acc as f32);
        pos += m as i64;
    }

    out
}

/// Build a Kaiser-windowed sinc low-pass — port of libsoxr `lsx_make_lpf`.
///
/// `fc` is the normalised cutoff (0.5 = Nyquist) relative to the design
/// rate (which is the up-sampled rate for rational ratios).  `scale` sets
/// the DC gain (use `L` for rational ratios to compensate zero-stuffing).
fn make_lpf(num_taps: usize, fc: f64, beta: f64, rho: f64, scale: f64) -> Vec<f64> {
    let m = num_taps - 1;
    let mult = scale / bessel_i0(beta);
    let mult1 = 1.0 / (0.5 * m as f64 + rho);
    let mut h = vec![0.0f64; num_taps];
    let half = m as f64 / 2.0;
    for i in 0..=m / 2 {
        let z = i as f64 - half;
        let x = z * std::f64::consts::PI;
        let y = z * mult1;
        let sinc = if x.abs() < 1e-20 { fc } else { (fc * x).sin() / x };
        let window = bessel_i0(beta * (1.0 - y * y).sqrt().max(0.0)) * mult;
        let coeff = sinc * window;
        h[i] = coeff;
        if i != m - i {
            h[m - i] = coeff;
        }
    }
    h
}

/// Integer-ratio decimation by factor M = source_rate / target_rate.
fn decimate_integer(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    let m = (source_rate / target_rate) as usize;
    let (h, half) = design_kaiser_lowpass(source_rate, target_rate, Some(m));

    assert_eq!(
        half % m,
        0,
        "filter half-length must be divisible by decimation factor"
    );
    let p = half / m;

    let output_len = samples.len() / m;
    let mut out = vec![0.0f32; output_len];

    for m_out in 0..output_len {
        let mut acc = 0.0f64;
        for r in 0..m {
            let mut i = r;
            let mut j = 0usize;
            while i < h.len() {
                let x_idx = ((m_out as isize - j as isize + p as isize) * m as isize)
                    - r as isize;
                if x_idx >= 0 && x_idx < samples.len() as isize {
                    acc += f64::from(h[i]) * f64::from(samples[x_idx as usize]);
                }
                i += m;
                j += 1;
            }
        }
        out[m_out] = acc as f32;
    }

    out
}

/// Arbitrary-ratio resampling using Kaiser-windowed sinc interpolation.
fn resample_arbitrary(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    let fs = source_rate as f64;
    let ft = target_rate as f64;
    let ratio = fs / ft;

    let (_h, half) = design_kaiser_lowpass(source_rate, target_rate, None);
    let beta = kaiser_beta(6.0 * SOXR_PRECISION);
    let fc = cutoff_frequency(source_rate, target_rate);
    let sinc_scale = 2.0 * fc / fs;

    let output_len = ((samples.len() as f64 / ratio) + 0.5) as usize;
    let mut out = Vec::with_capacity(output_len);

    for m in 0..output_len {
        let pos = m as f64 * ratio;
        let base = pos.floor() as isize;
        let frac = pos - base as f64;

        let mut acc = 0.0f64;
        for k in -(half as isize)..=(half as isize) {
            let x_idx = base - k;
            if x_idx >= 0 && x_idx < samples.len() as isize {
                let tau = frac + k as f64;
                let h_val = sinc_kaiser_value(tau, sinc_scale, half as f64, beta);
                acc += f64::from(samples[x_idx as usize]) * h_val;
            }
        }
        out.push(acc as f32);
    }

    out
}

const SOXR_PASSBAND_END: f64 = 0.913;
const SOXR_STOPBAND_BEGIN: f64 = 1.0;
const SOXR_PRECISION: f64 = 20.0;

/// `lsx_to_3dB(a) = 1 - lsx_inv_f_resp(-3, a)` from libsoxr filter.c.
/// Used to compute the dynamic passband_end for HQ quality.
fn lsx_to_3db(a: f64) -> f64 {
    1.0 - lsx_inv_f_resp(-3.0, a)
}

/// `lsx_inv_f_resp` from libsoxr filter.c — inverse frequency-response model.
fn lsx_inv_f_resp(drop: f64, a: f64) -> f64 {
    let x = sine_phi(a);
    let drop = (drop * std::f64::consts::LN_10 * 0.05).exp(); // dB_to_linear
    let s = if drop > 0.5 { 1.0 - drop } else { drop };
    let x = s.powf(1.0 / sine_pow(x)).asin() / x;
    if drop > 0.5 { x } else { 1.0 - x }
}

fn sine_phi(a: f64) -> f64 {
    ((2.0517e-07 * a - 1.1303e-04) * a + 0.023154) * a + 0.55924
}
fn sine_pow(x: f64) -> f64 {
    0.5_f64.ln() / ((x * 0.5).sin()).ln()
}

fn cutoff_frequency(source_rate: u32, target_rate: u32) -> f64 {
    let nyquist_min = (target_rate.min(source_rate) as f64) / 2.0;
    let fp = SOXR_PASSBAND_END * nyquist_min;
    let fsb = SOXR_STOPBAND_BEGIN * nyquist_min;
    (fp + fsb) / 2.0
}

fn design_kaiser_lowpass(
    source_rate: u32,
    target_rate: u32,
    decimation_factor: Option<usize>,
) -> (Vec<f32>, usize) {
    let fs = source_rate as f64;
    let fc = cutoff_frequency(source_rate, target_rate);
    let nyquist_min = (target_rate.min(source_rate) as f64) / 2.0;
    let fp = SOXR_PASSBAND_END * nyquist_min;
    let fsb = SOXR_STOPBAND_BEGIN * nyquist_min;
    let delta_f = fsb - fp;

    let attenuation_db = 6.0 * SOXR_PRECISION;
    let beta = kaiser_beta(attenuation_db);

    let delta_omega = 2.0 * std::f64::consts::PI * delta_f / fs;
    let mut n = ((attenuation_db - 7.95) / (2.285 * delta_omega)).ceil() as usize;
    if n % 2 == 0 {
        n += 1;
    }

    if let Some(m) = decimation_factor {
        let half = (n - 1) / 2;
        let half = if half % m == 0 {
            half
        } else {
            half + (m - half % m)
        };
        n = 2 * half + 1;
    }

    let half = (n - 1) / 2;
    let alpha = half as f64;
    let mut h = Vec::with_capacity(n);
    let sinc_scale = 2.0 * fc / fs;

    for i in 0..n {
        let x = i as f64 - alpha;
        let sinc = if x.abs() < 1e-12 {
            sinc_scale
        } else {
            let arg = std::f64::consts::PI * 2.0 * fc * x / fs;
            sinc_scale * arg.sin() / arg
        };
        let window = kaiser_window_discrete(i, n, beta);
        h.push((sinc * window) as f32);
    }

    (h, half)
}

fn sinc_kaiser_value(tau: f64, sinc_scale: f64, half: f64, beta: f64) -> f64 {
    let sinc = if tau.abs() < 1e-12 {
        sinc_scale
    } else {
        let arg = std::f64::consts::PI * sinc_scale * tau;
        sinc_scale * arg.sin() / arg
    };
    let window = kaiser_window_continuous(tau, half, beta);
    sinc * window
}

fn kaiser_beta(attenuation_db: f64) -> f64 {
    if attenuation_db > 50.0 {
        0.1102 * (attenuation_db - 8.7)
    } else if attenuation_db >= 21.0 {
        0.5842 * (attenuation_db - 21.0).powf(0.4)
            + 0.07886 * (attenuation_db - 21.0)
    } else {
        0.0
    }
}

fn kaiser_window_discrete(i: usize, n: usize, beta: f64) -> f64 {
    let alpha = (n - 1) as f64 / 2.0;
    let x = (i as f64 - alpha) / alpha;
    kaiser_window_continuous(x * alpha, alpha, beta)
}

fn kaiser_window_continuous(tau: f64, half: f64, beta: f64) -> f64 {
    if half <= 0.0 {
        return 1.0;
    }
    let x = tau / half;
    if x.abs() >= 1.0 {
        return 0.0;
    }
    let arg = (1.0 - x * x).sqrt() * beta;
    bessel_i0(arg) / bessel_i0(beta)
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x2 = x * x / 4.0;
    for k in 1..=32 {
        term *= x2 / (k * k) as f64;
        sum += term;
        if term.abs() < 1e-20 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimate_by_three_produces_expected_length() {
        let x: Vec<f32> = (0..9000).map(|i| (i as f32).sin()).collect();
        let y = decimate_integer(&x, 48000, 16000);
        assert_eq!(y.len(), x.len() / 3);
    }

    #[test]
    fn arbitrary_resample_produces_expected_length() {
        let x: Vec<f32> = (0..44100).map(|i| (i as f32).sin()).collect();
        let y = resample_arbitrary(&x, 44100, 16000);
        assert!(y.len() > 0);
        let expected = (x.len() as f64 * 16000.0 / 44100.0 + 0.5) as usize;
        assert!((y.len() as isize - expected as isize).abs() <= 1);
    }

    #[test]
    fn rational_24k_to_16k_produces_expected_length() {
        let x: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.01).sin()).collect();
        let y = decimate_rational(&x, 2, 3);
        assert_eq!(y.len(), 16000);
    }

    /// DC gain must be 1.0 for all paths (no amplitude scaling).
    #[test]
    fn dc_gain_is_unity_rational() {
        let x: Vec<f32> = vec![1.0; 48000];
        let y = decimate_rational(&x, 2, 3);
        let dc = y[y.len() / 2..].iter().sum::<f32>() / (y.len() / 2) as f32;
        assert!((dc - 1.0).abs() < 1e-3, "DC gain = {}", dc);
    }

    #[test]
    fn dc_gain_is_unity_integer() {
        let x: Vec<f32> = vec![1.0; 48000];
        let y = decimate_integer(&x, 48000, 16000);
        let dc = y[y.len() / 2..].iter().sum::<f32>() / (y.len() / 2) as f32;
        assert!((dc - 1.0).abs() < 1e-3, "DC gain = {}", dc);
    }

    /// 24k→16k: 验证通带内幅度无显著衰减。
    /// 这是一个幅度统计测试（非逐样本 bit-exact 回归）。
    #[test]
    fn rational_24k_preserves_passband_amplitude() {
        let sr = 24000.0f64;
        let input: Vec<f32> = (0..300)
            .map(|i| (2.0 * std::f64::consts::PI * 2000.0 * i as f64 / sr).sin() as f32)
            .collect();
        let out = decimate_rational(&input, 2, 3);
        // 稳态幅度（跳过首尾各 30 样本以避免暂态）。
        let steady = &out[30..out.len() - 30];
        let amp: f64 = steady.iter().map(|v| v.abs() as f64).sum::<f64>() / steady.len() as f64;
        // libsoxr soxr_hq 对同一段信号给出 0.603577；我们验证通带幅度一致。
        assert!(
            (amp - 0.6036).abs() < 1e-3,
            "passband amplitude {} differs from expected 0.6036",
            amp
        );
    }
}
