"""Contrast pair construction: A = clean audio, B = Gaussian noise matched in length and RMS."""
import zlib

import numpy as np

from caga import config


def _rng_for(key):
    """Deterministic RNG per item, seeded from `key` and CONTRAST_SEED."""
    h = (zlib.crc32(str(key).encode("utf-8")) ^ (int(config.CONTRAST_SEED) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return np.random.default_rng(h)


def make_gaussian_noise(wav, seed_key, snr_db):
    """White Gaussian noise of the same length as `wav`, with RMS = RMS(wav) * 10^(-snr_db / 20)."""
    wav = np.asarray(wav, dtype=np.float32)
    rng = _rng_for(seed_key)
    audio_rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-12
    target_rms = audio_rms * (10.0 ** (-float(snr_db) / 20.0))
    n = rng.standard_normal(len(wav)).astype(np.float32)
    n = n / (np.sqrt(np.mean(n ** 2)) + 1e-12) * target_rms
    return np.clip(n, -1.0, 1.0).astype(np.float32)


def load_audio_pair(item, sr=None):
    """Return (clean, noise, sr) for a manifest item. Both arrays have the same length, so A and B
    tokenize to the same number of audio tokens and their activations can be subtracted per position.
    The noise seed comes from the audio path or the item id (config.CONTRAST_SEED_KEY)."""
    import librosa
    sr = sr or config.SAMPLE_RATE
    audio_path = item["audio_path"]
    clean, sr = librosa.load(audio_path, sr=sr, mono=True)
    key = audio_path if config.CONTRAST_SEED_KEY == "path" else item["id"]
    noise = make_gaussian_noise(clean, seed_key=key, snr_db=config.CONTRAST_SNR_DB)
    return clean.astype(np.float32), noise, sr
