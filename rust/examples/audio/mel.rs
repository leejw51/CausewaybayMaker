//! Log-mel spectrogram, matching mlx_whisper/audio.py.

use anyhow::{Result, bail};
use mlx_rs::ops::{maximum, zeros};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, fft};

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
/// Samples in one 30 s window: the encoder always sees exactly this much audio.
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE;
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;

/// librosa slaney mel filterbank (sr=16000, n_fft=400), shape (n_mels, 201).
pub fn mel_filters(n_mels: i32) -> Result<Array> {
    let bytes: &[u8] = match n_mels {
        80 => include_bytes!("../../assets/mel_80.npy"),
        128 => include_bytes!("../../assets/mel_128.npy"),
        n => bail!("unsupported n_mels {n}"),
    };
    crate::npy::parse_npy(bytes)
}

/// Periodic Hann window (numpy.hanning(N + 1)[:-1]).
fn hanning(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .map(|w| w as f32)
        .collect()
}

/// Pad or trim `audio` (16 kHz mono) to one 30 s window and return the
/// (3000, n_mels) float32 log-mel spectrogram.
pub fn log_mel_spectrogram(audio: &[f32], n_mels: i32) -> Result<Array> {
    let mut x: Vec<f32> = audio.iter().copied().take(N_SAMPLES).collect();
    x.resize(N_SAMPLES, 0.0);

    // Reflect-pad N_FFT/2 samples on both sides (torch.stft center=True).
    let pad = N_FFT / 2;
    let mut padded = Vec::with_capacity(N_SAMPLES + 2 * pad);
    padded.extend(x[1..=pad].iter().rev());
    padded.extend_from_slice(&x);
    padded.extend(x[N_SAMPLES - pad - 1..N_SAMPLES - 1].iter().rev());

    // Windowed frames: (N_FRAMES + 1, N_FFT); the last frame is dropped after the FFT.
    let window = hanning(N_FFT);
    let n_frames = (padded.len() - N_FFT + HOP_LENGTH) / HOP_LENGTH;
    let mut frames = Vec::with_capacity(n_frames * N_FFT);
    for t in 0..n_frames {
        let start = t * HOP_LENGTH;
        frames.extend(padded[start..start + N_FFT].iter().zip(&window).map(|(s, w)| s * w));
    }
    let frames = Array::from_slice(&frames, &[n_frames as i32, N_FFT as i32]);

    let spec = fft::rfft(&frames, None, None)?;
    let magnitudes = spec.index((0..N_FRAMES as i32, ..)).abs()?.square()?;
    let mel = magnitudes.matmul(mel_filters(n_mels)?.t())?;

    let log_spec = maximum(&mel, Array::from_f32(1e-10))?.log10()?;
    let floor = log_spec.max(None)?.subtract(Array::from_f32(8.0))?;
    let log_spec = maximum(&log_spec, &floor)?;
    let log_spec = log_spec.add(Array::from_f32(4.0))?.divide(Array::from_f32(4.0))?;
    log_spec.eval()?;
    Ok(log_spec)
}

/// Zero spectrogram placeholder, used to warm up kernels.
pub fn silence(n_mels: i32) -> Result<Array> {
    Ok(zeros::<f32>(&[N_FRAMES as i32, n_mels])?)
}
