"""
utils.py
-------------------------------------------------------------------------------
Requirements:
    pip install musdb librosa matplotlib numpy pandas soundfile
    ffmpeg debian package (sudo apt install -y ffmpeg)
-------------------------------------------------------------------------------
Audio utility functions:
- loading either the official 7-second sample snippets or a local full dataset
- plotting mixture / vocal spectrograms
- extracting compact spectral features
- identifying vocal-heavy regions with an ideal ratio mask

Examples:
    # Use official 7-second MUSDB snippets
    python musdb_tester.py --mode sample

    # Use a specific sample track by exact name
    python musdb_tester.py --mode sample --track "Actions - One Minute Smile"

    # Use a locally downloaded MUSDB18 dataset
    python musdb_tester.py --mode local --root /path/to/musdb18 --track "Actions - One Minute Smile"

    # Use local test subset
    python musdb_tester.py --mode local --root /path/to/musdb18 --subset test
-------------------------------------------------------------------------------
REPET utility functions:
- computing a REPET-style beat spectrum from the mixture power spectrogram
- estimating the repeating period from beat-spectrum peaks
- segmenting the mixture spectrogram into repeating-length blocks
- building a repeating background model with the elementwise median
- generating a repeating-background mask and complementary foreground/vocal mask
- evaluating the REPET vocal-oriented mask against the ideal vocal mask
- saving REPET masks, beat-spectrum plots, and period-score plots

Examples:
    # Run REPET mask generation on official 7-second MUSDB snippets
    python repet_tester.py --mode sample

    # Run REPET on a specific sample track
    python repet_tester.py --mode sample --track "Actions - One Minute Smile"

    # Run REPET on a locally downloaded MUSDB18 dataset
    python repet_tester.py --mode local --root /path/to/musdb18 --track "Actions - One Minute Smile"

    # Adjust REPET period-search settings
    python repet_tester.py --mode sample --track "Actions - One Minute Smile" --min_period_sec 0.25 --max_period_sec 2.0
-------------------------------------------------------------------------------
Imports:
-------------------------------------------------------------------------------
"""
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.ensemble import HistGradientBoostingRegressor

import os

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import librosa
import librosa.display
import musdb
import soundfile as sf
"""
-------------------------------------------------------------------------------
Audio Utility Functions
-------------------------------------------------------------------------------
"""
def downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def get_database(
    mode: str,
    root: Optional[str] = None,
    subset: str = "train",
    is_wav: bool = True
) -> musdb.DB:
    """
    Create a MUSDB database handle.

    mode='sample':
        Uses musdb's sample download behavior.

    mode='local':
        Uses a locally downloaded MUSDB18 dataset at root.
    """
    if mode == "sample":
        return musdb.DB(download=True)

    if mode == "local":
        if root is None:
            raise ValueError("For mode='local', you must provide --root")
        return musdb.DB(root=root, subsets=subset, is_wav=is_wav)

    raise ValueError(f"Unknown mode: {mode}")


def list_track_names(db: musdb.DB) -> list[str]:
    """Return all track names in the dataset handle."""
    return [track.name for track in db]


def select_track(db: musdb.DB, track_name: Optional[str] = None):
    """
    Select a track by exact name, or use the first track if not provided.
    """
    if track_name is None:
        return db[0]

    for track in db:
        if track.name == track_name:
            return track

    available = list_track_names(db)
    preview = "\n".join(available[:20])
    raise ValueError(
        f'Track "{track_name}" not found.\n'
        f"Here are some available tracks:\n{preview}"
    )


def load_track_audio(track) -> Dict[str, Any]:
    """
    Extract mixture and vocals from a MUSDB track.
    """
    mixture = track.audio
    vocals = track.targets["vocals"].audio
    bass = track.targets["bass"].audio
    drums = track.targets["drums"].audio
    other = track.targets["other"].audio

    accompaniment = bass + drums + other

    return {
        "track": track,
        "track_name": track.name,
        "sample_rate": track.rate,
        "mixture_stereo": mixture,
        "vocals_stereo": vocals,
        "bass_stereo": bass,
        "drums_stereo": drums,
        "other_stereo": other,
        "accompaniment_stereo": accompaniment,
        "mixture_mono": downmix_to_mono(mixture),
        "vocals_mono": downmix_to_mono(vocals),
        "bass_mono": downmix_to_mono(bass),
        "drums_mono": downmix_to_mono(drums),
        "other_mono": downmix_to_mono(other),
        "accompaniment_mono": downmix_to_mono(accompaniment),
    }


def compute_stft_features(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048
) -> Dict[str, Any]:
    """
    Compute STFT and a compact set of exploratory spectral features.
    """
    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True
    )

    mag = np.abs(stft)
    phase = np.angle(stft)
    log_mag = np.log1p(mag)
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)

    harmonic, percussive = librosa.decompose.hpss(mag)

    rms = librosa.feature.rms(S=mag)[0]
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=mag, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=mag)[0]

    harmonic_energy = np.sum(harmonic, axis=0)
    percussive_energy = np.sum(percussive, axis=0)
    hp_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-8)

    return {
        "stft": stft,
        "mag": mag,
        "phase": phase,
        "log_mag": log_mag,
        "mag_db": mag_db,
        "harmonic": harmonic,
        "percussive": percussive,
        "rms": rms,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness,
        "harmonic_energy": harmonic_energy,
        "percussive_energy": percussive_energy,
        "hp_ratio": hp_ratio,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
    }


def ideal_ratio_mask(
    vocals_mag: np.ndarray,
    accomp_mag: np.ndarray,
    p: int = 1,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Simple ideal ratio mask approximation:
        M = |V|^p / (|V|^p + |I|^p + eps)
    """
    return (vocals_mag ** p) / (vocals_mag ** p + accomp_mag ** p + eps)


def build_frame_feature_table(mix_features: Dict[str, Any]) -> pd.DataFrame:
    """Create a per-frame feature table."""
    n_frames = len(mix_features["rms"])

    return pd.DataFrame({
        "frame": np.arange(n_frames),
        "rms": mix_features["rms"],
        "spectral_centroid": mix_features["centroid"],
        "spectral_bandwidth": mix_features["bandwidth"],
        "spectral_rolloff": mix_features["rolloff"],
        "spectral_flatness": mix_features["flatness"],
        "harmonic_energy": mix_features["harmonic_energy"],
        "percussive_energy": mix_features["percussive_energy"],
        "harmonic_to_total_ratio": mix_features["hp_ratio"],
    })


def summarize_high_vocal_regions(
    irm: np.ndarray,
    sr: int,
    hop_length: int,
    threshold: float = 0.6
) -> pd.DataFrame:
    """
    Identify frames whose mean vocal mask exceeds a threshold.
    """
    mean_mask_per_frame = np.mean(irm, axis=0)
    active = mean_mask_per_frame >= threshold
    frame_ids = np.where(active)[0]
    times = librosa.frames_to_time(frame_ids, sr=sr, hop_length=hop_length)

    return pd.DataFrame({
        "frame": frame_ids,
        "time_sec": times,
        "avg_vocal_mask": mean_mask_per_frame[frame_ids],
    })


def plot_spectrogram(
    S_plot: np.ndarray,
    sr: int,
    hop_length: int,
    title: str,
    save_path: Optional[str] = None,
    db_scale: bool = False
) -> None:
    """Plot a spectrogram or mask."""
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(
        S_plot,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log"
    )
    plt.colorbar(format="%+2.0f dB" if db_scale else "%0.2f")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    #plt.show()
    plt.close()


def plot_feature_curves(feature_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot key frame-level features."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(feature_df["rms"])
    axes[0].set_ylabel("RMS")
    axes[0].set_title("Frame-Level Features")

    axes[1].plot(feature_df["spectral_centroid"])
    axes[1].set_ylabel("Centroid (Hz)")

    axes[2].plot(feature_df["harmonic_to_total_ratio"])
    axes[2].set_ylabel("Harmonic Ratio")

    axes[3].plot(feature_df["spectral_flatness"])
    axes[3].set_ylabel("Flatness")
    axes[3].set_xlabel("Frame")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    #plt.show()
    plt.close()


def save_outputs(
    outdir: str,
    track_name: str,
    mix_feat: Dict[str, Any],
    voc_feat: Dict[str, Any],
    irm: np.ndarray,
    feature_df: pd.DataFrame,
    vocal_regions: pd.DataFrame,
    sr: int,
    hop_length: int
) -> None:
    """Save plots and csv outputs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    safe_track_name = track_name.replace("/", "_").replace("\\", "_")

    feature_csv = os.path.join(outdir, f"{safe_track_name}_frame_features.csv")
    vocal_regions_csv = os.path.join(outdir, f"{safe_track_name}_vocal_heavy_frames.csv")

    feature_df.to_csv(feature_csv, index=False)
    vocal_regions.to_csv(vocal_regions_csv, index=False)

    plot_spectrogram(
        mix_feat["mag_db"],
        sr,
        hop_length,
        title=f"Mixture Spectrogram: {track_name}",
        save_path=os.path.join(outdir, f"{safe_track_name}_mixture_spectrogram.png"),
        db_scale=True
    )

    plot_spectrogram(
        voc_feat["mag_db"],
        sr,
        hop_length,
        title=f"Vocal Spectrogram: {track_name}",
        save_path=os.path.join(outdir, f"{safe_track_name}_vocal_spectrogram.png"),
        db_scale=True
    )

    plot_spectrogram(
        irm,
        sr,
        hop_length,
        title=f"Ideal Ratio Mask (Vocals): {track_name}",
        save_path=os.path.join(outdir, f"{safe_track_name}_ideal_ratio_mask.png"),
        db_scale=False
    )

    plot_feature_curves(
        feature_df,
        save_path=os.path.join(outdir, f"{safe_track_name}_feature_curves.png")
    )

    print(f"Saved feature table to: {feature_csv}")
    print(f"Saved vocal-heavy frame summary to: {vocal_regions_csv}")

import soundfile as sf

def estimate_stems_from_mask(
    mix_feat: Dict[str, Any],
    vocal_mask: np.ndarray,
    length: Optional[int] = None,
    mask_name: str = "vocal_mask"
) -> Dict[str, Any]:
    """
    Reconstruct time-domain vocal and accompaniment estimates from a soft mask.
    """
    mix_stft = np.asarray(mix_feat["stft"])
    mask = np.asarray(vocal_mask, dtype=np.float32)

    if mask.shape != mix_stft.shape:
        raise ValueError(
            f"{mask_name} shape {mask.shape} does not match mixture STFT shape {mix_stft.shape}"
        )

    mask = np.clip(mask, 0.0, 1.0)
    background_mask = 1.0 - mask

    vocal_stft = mix_stft * mask
    accompaniment_stft = mix_stft * background_mask

    istft_kwargs = dict(
        hop_length=mix_feat["hop_length"],
        win_length=mix_feat["win_length"],
        window="hann",
        center=True,
        length=length,
    )

    vocals_est = librosa.istft(vocal_stft, **istft_kwargs)
    accompaniment_est = librosa.istft(accompaniment_stft, **istft_kwargs)
    remix_est = vocals_est + accompaniment_est

    return {
        "vocal_mask": mask,
        "background_mask": background_mask,
        "vocal_stft": vocal_stft,
        "accompaniment_stft": accompaniment_stft,
        "vocals_est": vocals_est,
        "accompaniment_est": accompaniment_est,
        "remix_est": remix_est,
    }


def save_estimated_stems(
    outdir: str,
    track_name: str,
    stems: Dict[str, Any],
    sr: int,
    prefix: str
) -> Dict[str, str]:
    """Save reconstructed stems as WAV files."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_name = track_name.replace("/", "_").replace("\\", "_")

    prefix = prefix.strip().lower()
    if prefix:
        prefix = f"_{prefix}"

    vocal_path = outdir / f"{safe_name}{prefix}_vocals_est.wav"
    accomp_path = outdir / f"{safe_name}{prefix}_accompaniment_est.wav"
    remix_path = outdir / f"{safe_name}{prefix}_remix_est.wav"

    sf.write(vocal_path, np.asarray(stems["vocals_est"], dtype=np.float32), sr)
    sf.write(accomp_path, np.asarray(stems["accompaniment_est"], dtype=np.float32), sr)
    sf.write(remix_path, np.asarray(stems["remix_est"], dtype=np.float32), sr)

    return {
        "vocals_path": str(vocal_path),
        "accompaniment_path": str(accomp_path),
        "remix_path": str(remix_path),
    }


def stem_metrics(
    estimated_stem: np.ndarray,
    reference_stem: np.ndarray,
    eps: float = 1e-8
) -> dict:
    """
    Compute basic waveform-domain metrics for one estimated stem against
    its reference stem.

    Parameters
    ----------
    estimated_stem : np.ndarray
        Estimated waveform.
    reference_stem : np.ndarray
        Ground-truth waveform.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    dict
        {
            "mae",
            "mse",
            "rmse",
            "snr_db",
            "si_sdr_db",
            "corr",
            "est_energy",
            "ref_energy",
            "energy_ratio",
        }
    """
    est = np.asarray(estimated_stem, dtype=np.float64).squeeze()
    ref = np.asarray(reference_stem, dtype=np.float64).squeeze()

    if est.shape != ref.shape:
        raise ValueError(
            f"Shape mismatch: estimated_stem {est.shape} vs reference_stem {ref.shape}"
        )

    err = est - ref

    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    ref_energy = float(np.sum(ref ** 2))
    est_energy = float(np.sum(est ** 2))
    noise_energy = float(np.sum(err ** 2))

    snr_db = float(10.0 * np.log10((ref_energy + eps) / (noise_energy + eps)))

    # SI-SDR
    scale = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = scale * ref
    residual = est - target

    target_energy = float(np.sum(target ** 2))
    residual_energy = float(np.sum(residual ** 2))
    si_sdr_db = float(10.0 * np.log10((target_energy + eps) / (residual_energy + eps)))

    if np.std(est) > 0 and np.std(ref) > 0:
        corr = float(np.corrcoef(est, ref)[0, 1])
    else:
        corr = 0.0

    energy_ratio = float(est_energy / (ref_energy + eps))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "snr_db": snr_db,
        "si_sdr_db": si_sdr_db,
        "corr": corr,
        "est_energy": est_energy,
        "ref_energy": ref_energy,
        "energy_ratio": energy_ratio,
    }
"""
-------------------------------------------------------------------------------
REPET Utility Functions
-------------------------------------------------------------------------------
"""
def _safe_divide(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return num / np.maximum(den, eps)


def repet_beat_spectrum(power_spec: np.ndarray) -> np.ndarray:
    """
    Compute a REPET-style beat spectrum from a power spectrogram.

    power_spec shape: (freq_bins, time_frames)
    Returns:
        beat_spectrum shape: (time_frames,)
    """
    n_bins, n_frames = power_spec.shape
    ac_rows = np.zeros((n_bins, n_frames), dtype=np.float64)

    for k in range(n_bins):
        row = power_spec[k]
        ac_full = np.correlate(row, row, mode="full")
        ac = ac_full[n_frames - 1:]  # keep nonnegative lags
        ac_rows[k, :] = ac

    beat = np.mean(ac_rows, axis=0)

    if beat[0] > 0:
        beat = beat / beat[0]

    return beat


def repet_estimate_period(
    beat_spectrum: np.ndarray,
    min_period: int = 1,
    max_period: int | None = None,
    deviation: int = 2
) -> tuple[int, np.ndarray]:
    """
    Estimate repeating period from REPET beat spectrum using a simplified
    version of the paper's integer-multiple scoring idea.

    Returns:
        best_period, scores
    """
    b = np.asarray(beat_spectrum, dtype=np.float64)
    n = len(b)

    # discard lag 0
    if n < 8:
        raise ValueError("Beat spectrum too short to estimate a stable period.")

    usable_end = int(np.floor(0.75 * n))  # ignore last quarter of lags
    if usable_end <= 3:
        raise ValueError("Beat spectrum usable region too short.")

    if max_period is None:
        # keep at least 3 full cycles
        max_period = usable_end // 3

    max_period = max(min(max_period, usable_end // 3), min_period)

    scores = np.zeros(n, dtype=np.float64)

    for p in range(min_period, max_period + 1):
        neighborhood_half = max(1, p // 2)
        vals = []

        m = 1
        while True:
            center = m * p
            if center >= usable_end:
                break

            left = max(1, center - neighborhood_half - deviation)
            right = min(usable_end, center + neighborhood_half + deviation + 1)

            local = b[left:right]
            if local.size == 0:
                m += 1
                continue

            peak_val = np.max(local)
            local_mean = np.mean(local)
            vals.append(peak_val - local_mean)
            m += 1

        if len(vals) >= 3:
            scores[p] = np.mean(vals)

    best_period = int(np.argmax(scores))
    if best_period < min_period or scores[best_period] <= 0:
        # fallback: strongest lag in valid range
        search = b[min_period:max_period + 1]
        best_period = int(np.argmax(search)) + min_period

    return best_period, scores


def repet_segment_spectrogram(mag: np.ndarray, period_frames: int) -> tuple[np.ndarray, int]:
    """
    Segment magnitude spectrogram into equal-length chunks.

    Returns:
        segments: shape (n_segments, freq_bins, period_frames)
        usable_frames: total frames used
    """
    if period_frames <= 0:
        raise ValueError("period_frames must be positive")

    n_bins, n_frames = mag.shape
    n_segments = n_frames // period_frames

    if n_segments < 3:
        raise ValueError(
            f"Need at least 3 segments for REPET median model; got {n_segments}. "
            f"Try smaller n_fft/hop_length or shorter estimated period."
        )

    usable_frames = n_segments * period_frames
    trimmed = mag[:, :usable_frames]
    segments = trimmed.reshape(n_bins, n_segments, period_frames).transpose(1, 0, 2)
    return segments, usable_frames


def repet_repeating_segment_model(segments: np.ndarray) -> np.ndarray:
    """
    Median model across segments.

    segments shape: (n_segments, freq_bins, period_frames)
    returns shape: (freq_bins, period_frames)
    """
    return np.median(segments, axis=0)


def repet_repeating_spectrogram(segments: np.ndarray, repeating_model: np.ndarray) -> np.ndarray:
    """
    Build repeating spectrogram estimate by elementwise min with repeating model.

    returns shape: (freq_bins, usable_frames)
    """
    repeated = np.minimum(segments, repeating_model[None, :, :])
    out = repeated.transpose(1, 0, 2).reshape(repeating_model.shape[0], -1)
    return out


def repet_masks_from_magnitude(
    mag: np.ndarray,
    min_period: int = 1,
    max_period: int | None = None,
    mask_eps: float = 1e-8
) -> dict:
    """
    Full REPET pipeline on a mixture magnitude spectrogram.

    Returns:
        {
            "beat_spectrum",
            "period_frames",
            "period_scores",
            "segments",
            "usable_frames",
            "repeating_segment_model",
            "repeating_mag",
            "background_mask",
            "foreground_mask",
        }
    """
    power_spec = mag ** 2
    beat = repet_beat_spectrum(power_spec)
    period_frames, period_scores = repet_estimate_period(
        beat,
        min_period=min_period,
        max_period=max_period
    )

    segments, usable_frames = repet_segment_spectrogram(mag, period_frames)
    repeating_model = repet_repeating_segment_model(segments)
    repeating_mag = repet_repeating_spectrogram(segments, repeating_model)

    mag_used = mag[:, :usable_frames]
    background_mask = _safe_divide(repeating_mag, mag_used, eps=mask_eps)
    background_mask = np.clip(background_mask, 0.0, 1.0)

    foreground_mask = 1.0 - background_mask
    foreground_mask = np.clip(foreground_mask, 0.0, 1.0)

    return {
        "beat_spectrum": beat,
        "period_frames": period_frames,
        "period_scores": period_scores,
        "segments": segments,
        "usable_frames": usable_frames,
        "repeating_segment_model": repeating_model,
        "repeating_mag": repeating_mag,
        "background_mask": background_mask,
        "foreground_mask": foreground_mask,
    }


def repet_vocal_mask_from_audio(
    y_mix: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    min_period_sec: float = 0.25,
    max_period_sec: float | None = None,
    mask_eps: float = 1e-8
) -> dict:
    """
    Run REPET from audio and return a vocal-oriented foreground mask.

    The vocal mask is the complement of the repeating-background mask.
    """
    mix_feat = compute_stft_features(
        y_mix,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )

    mag = mix_feat["mag"]
    total_frames = mag.shape[1]

    min_period_frames = max(1, int(round(min_period_sec * sr / hop_length)))

    if max_period_sec is None:
        max_period_frames = max(3, int(np.floor(0.25 * total_frames)))
    else:
        max_period_frames = max(1, int(round(max_period_sec * sr / hop_length)))

    repet = repet_masks_from_magnitude(
        mag,
        min_period=min_period_frames,
        max_period=max_period_frames,
        mask_eps=mask_eps
    )

    usable_frames = repet["usable_frames"]
    full_vocal_mask = np.zeros_like(mag)
    full_bg_mask = np.zeros_like(mag)

    full_vocal_mask[:, :usable_frames] = repet["foreground_mask"]
    full_bg_mask[:, :usable_frames] = repet["background_mask"]

    if usable_frames < total_frames:
        tail = total_frames - usable_frames
        # simple fill for leftover frames: use last available column
        if usable_frames > 0:
            full_vocal_mask[:, usable_frames:] = repet["foreground_mask"][:, [-1]]
            full_bg_mask[:, usable_frames:] = repet["background_mask"][:, [-1]]

    repet["mix_feat"] = mix_feat
    repet["vocal_mask"] = full_vocal_mask
    repet["background_mask_full"] = full_bg_mask
    return repet


def repet_mask_metrics(
    estimated_vocal_mask: np.ndarray,
    ideal_vocal_mask: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Lightweight mask-quality metrics for debugging/prototyping.
    """
    est = np.asarray(estimated_vocal_mask, dtype=np.float64)
    ref = np.asarray(ideal_vocal_mask, dtype=np.float64)

    if est.shape != ref.shape:
        raise ValueError(f"Shape mismatch: est {est.shape}, ref {ref.shape}")

    mae = float(np.mean(np.abs(est - ref)))
    mse = float(np.mean((est - ref) ** 2))

    est_bin = est >= threshold
    ref_bin = ref >= threshold

    tp = np.sum(est_bin & ref_bin)
    fp = np.sum(est_bin & ~ref_bin)
    fn = np.sum(~est_bin & ref_bin)

    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))

    corr = float(np.corrcoef(est.ravel(), ref.ravel())[0, 1]) if np.std(est) > 0 and np.std(ref) > 0 else 0.0

    return {
        "mae": mae,
        "mse": mse,
        "precision@thr": precision,
        "recall@thr": recall,
        "f1@thr": f1,
        "corr": corr,
    }


def save_repet_outputs(
    outdir: str,
    track_name: str,
    mix_mag_db: np.ndarray,
    vocal_mask: np.ndarray,
    ideal_mask: np.ndarray,
    beat_spectrum: np.ndarray,
    period_scores: np.ndarray,
    sr: int,
    hop_length: int
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_name = track_name.replace("/", "_").replace("\\", "_")

    plot_spectrogram(
        mix_mag_db,
        sr,
        hop_length,
        title=f"Mixture Spectrogram: {track_name}",
        save_path=str(outdir / f"{safe_name}_repet_mix_spec.png"),
        db_scale=True
    )

    plot_spectrogram(
        vocal_mask,
        sr,
        hop_length,
        title=f"REPET Vocal Mask: {track_name}",
        save_path=str(outdir / f"{safe_name}_repet_vocal_mask.png"),
        db_scale=False
    )

    plot_spectrogram(
        ideal_mask,
        sr,
        hop_length,
        title=f"Ideal Vocal Mask: {track_name}",
        save_path=str(outdir / f"{safe_name}_ideal_vocal_mask.png"),
        db_scale=False
    )

    plt.figure(figsize=(10, 4))
    plt.plot(beat_spectrum)
    plt.title(f"REPET Beat Spectrum: {track_name}")
    plt.xlabel("Lag (frames)")
    plt.ylabel("Normalized autocorrelation")
    plt.tight_layout()
    plt.savefig(outdir / f"{safe_name}_repet_beat_spectrum.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(period_scores)
    plt.title(f"REPET Period Scores: {track_name}")
    plt.xlabel("Candidate period (frames)")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(outdir / f"{safe_name}_repet_period_scores.png", dpi=200, bbox_inches="tight")
    plt.close()
"""
-------------------------------------------------------------------------------
RPCA utility functions:
-------------------------------------------------------------------------------
"""
def rpca_ialm(
    M: np.ndarray,
    lam: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Inexact Augmented Lagrange Multiplier (IALM) solver for Robust PCA.
 
    Decomposes M ≈ L + S  where L is low-rank (accompaniment) and S is sparse
    (vocals / foreground events).  Follows Lin, Chen & Ma (2010).
 
    Parameters
    ----------
    M        : 2-D array to decompose, shape (freq_bins, time_frames)
    lam      : Sparsity regularization weight.  Defaults to
               1 / sqrt(max(m, n)).
    max_iter : Maximum number of outer iterations.
    tol      : Convergence threshold on ||M - L - S||_F / ||M||_F.
 
    Returns
    -------
    L        : Low-rank component (shape same as M)
    S        : Sparse component   (shape same as M)
    n_iter   : Number of iterations executed
    converged: True if the relative residual dropped below tol
    """
    m, n = M.shape
    norm_M = np.linalg.norm(M, "fro")
 
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
 
    # --- initialisation ---
    mu = 1.25 / np.linalg.norm(M, 2)   # operator / spectral norm
    mu_bar = mu * 1e7
    rho = 1.5
 
    Y = M / max(np.linalg.norm(M, 2), np.linalg.norm(M, np.inf) / lam)
    S = np.zeros_like(M)
    L = np.zeros_like(M)
 
    converged = False
    n_iter = 0
 
    for i in range(max_iter):
        n_iter = i + 1
 
        # --- update L via singular value thresholding ---
        U, sigma, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        threshold = 1.0 / mu
        sigma_thresh = np.maximum(sigma - threshold, 0.0)
        L = (U * sigma_thresh) @ Vt
 
        # --- update S via soft thresholding ---
        residual_S = M - L + Y / mu
        threshold_S = lam / mu
        S = np.sign(residual_S) * np.maximum(np.abs(residual_S) - threshold_S, 0.0)
 
        # --- update dual variable Y ---
        delta = M - L - S
        Y = Y + mu * delta
        mu = min(rho * mu, mu_bar)
 
        # --- convergence check ---
        rel_err = np.linalg.norm(delta, "fro") / (norm_M + 1e-12)
        if rel_err < tol:
            converged = True
            break
 
    return L, S, n_iter, converged
 
 
def rpca_vocal_mask_from_audio(
    y_mix: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    lam: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    mask_eps: float = 1e-8
) -> dict:
    """
    Run RPCA on the mixture log-magnitude spectrogram and return a soft
    vocal-oriented mask.
 
    The accompaniment is modelled by the low-rank component L; the sparse
    component S captures foreground events (vocals, transients).  The vocal
    mask is derived from the relative energy of S at each time-frequency bin.
 
    Parameters
    ----------
    y_mix      : Mono mixture waveform.
    sr         : Sample rate.
    n_fft / hop_length / win_length : STFT parameters.
    lam        : RPCA regularization weight (None → automatic).
    max_iter   : Maximum IALM iterations.
    tol        : IALM convergence tolerance.
    mask_eps   : Small constant for numerical stability in mask computation.
 
    Returns
    -------
    dict with keys:
        mix_feat         – output of compute_stft_features
        low_rank_mag     – magnitude approximation from L (accompaniment model)
        sparse_mag       – magnitude approximation from S (vocal / foreground)
        vocal_mask       – soft vocal mask in [0, 1], shape (freq, time)
        background_mask  – complement of vocal_mask
        lam_used         – lambda value that was used
        n_iter           – number of IALM iterations
        converged        – whether IALM converged
    """
    mix_feat = compute_stft_features(
        y_mix, sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
 
    mag = mix_feat["mag"]
 
    # RPCA is applied to the log-magnitude spectrogram; this compresses the
    # dynamic range and makes the low-rank assumption more appropriate.
    log_mag = np.log1p(mag)
 
    L, S, n_iter, converged = rpca_ialm(
        log_mag,
        lam=lam,
        max_iter=max_iter,
        tol=tol
    )
 
    lam_used = (lam if lam is not None
                else 1.0 / np.sqrt(max(log_mag.shape)))
 
    # Map back to linear magnitude space for mask construction.
    low_rank_mag = np.expm1(np.clip(L, 0.0, None))
    sparse_mag = np.expm1(np.clip(S, 0.0, None))
 
    # Soft vocal mask: proportion of sparse energy at each bin.
    total = low_rank_mag + sparse_mag
    vocal_mask = _safe_divide(sparse_mag, total, eps=mask_eps)
    vocal_mask = np.clip(vocal_mask, 0.0, 1.0)
 
    background_mask = 1.0 - vocal_mask
 
    return {
        "mix_feat": mix_feat,
        "low_rank_mag": low_rank_mag,
        "sparse_mag": sparse_mag,
        "vocal_mask": vocal_mask,
        "background_mask": background_mask,
        "lam_used": lam_used,
        "n_iter": n_iter,
        "converged": converged,
    }
 
 
def rpca_mask_metrics(
    estimated_vocal_mask: np.ndarray,
    ideal_vocal_mask: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Lightweight mask-quality metrics for RPCA vocal mask evaluation.
 
    Identical in structure to repet_mask_metrics so results are directly
    comparable across all three unsupervised methods.
    """
    est = np.asarray(estimated_vocal_mask, dtype=np.float64)
    ref = np.asarray(ideal_vocal_mask, dtype=np.float64)
 
    if est.shape != ref.shape:
        raise ValueError(f"Shape mismatch: est {est.shape}, ref {ref.shape}")
 
    mae = float(np.mean(np.abs(est - ref)))
    mse = float(np.mean((est - ref) ** 2))
 
    est_bin = est >= threshold
    ref_bin = ref >= threshold
 
    tp = np.sum(est_bin & ref_bin)
    fp = np.sum(est_bin & ~ref_bin)
    fn = np.sum(~est_bin & ref_bin)
 
    precision = float(tp / (tp + fp + 1e-8))
    recall    = float(tp / (tp + fn + 1e-8))
    f1        = float(2 * precision * recall / (precision + recall + 1e-8))
 
    corr = (
        float(np.corrcoef(est.ravel(), ref.ravel())[0, 1])
        if np.std(est) > 0 and np.std(ref) > 0
        else 0.0
    )
 
    return {
        "mae": mae,
        "mse": mse,
        "precision@thr": precision,
        "recall@thr": recall,
        "f1@thr": f1,
        "corr": corr,
    }
 
 
def save_rpca_outputs(
    outdir: str,
    track_name: str,
    mix_mag_db: np.ndarray,
    vocal_mask: np.ndarray,
    ideal_mask: np.ndarray,
    low_rank_mag: np.ndarray,
    sparse_mag: np.ndarray,
    sr: int,
    hop_length: int
) -> None:
    """Save RPCA spectrograms and masks as PNG files."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_name = track_name.replace("/", "_").replace("\\", "_")
 
    plot_spectrogram(
        mix_mag_db, sr, hop_length,
        title=f"Mixture Spectrogram: {track_name}",
        save_path=str(outdir / f"{safe_name}_rpca_mix_spec.png"),
        db_scale=True
    )
 
    plot_spectrogram(
        vocal_mask, sr, hop_length,
        title=f"RPCA Vocal Mask: {track_name}",
        save_path=str(outdir / f"{safe_name}_rpca_vocal_mask.png"),
        db_scale=False
    )
 
    plot_spectrogram(
        ideal_mask, sr, hop_length,
        title=f"Ideal Vocal Mask: {track_name}",
        save_path=str(outdir / f"{safe_name}_rpca_ideal_vocal_mask.png"),
        db_scale=False
    )
 
    low_rank_db = librosa.amplitude_to_db(low_rank_mag, ref=np.max)
    plot_spectrogram(
        low_rank_db, sr, hop_length,
        title=f"RPCA Low-Rank Component (Accompaniment): {track_name}",
        save_path=str(outdir / f"{safe_name}_rpca_low_rank.png"),
        db_scale=True
    )
 
    sparse_db = librosa.amplitude_to_db(sparse_mag + 1e-8, ref=np.max)
    plot_spectrogram(
        sparse_db, sr, hop_length,
        title=f"RPCA Sparse Component (Vocal/Foreground): {track_name}",
        save_path=str(outdir / f"{safe_name}_rpca_sparse.png"),
        db_scale=True
    )

"""
-------------------------------------------------------------------------------
HPSS utility functions:
-------------------------------------------------------------------------------
"""
def _ensure_odd(k: int, name: str) -> int:
    """Return k if it is a positive odd integer, otherwise raise a clear error."""
    if k < 1 or k % 2 == 0:
        raise ValueError(
            f"{name} must be a positive odd integer; got {k}. "
            "Try the next odd value, e.g. {k + 1 if k % 2 == 0 else k + 2}."
        )
    return k
 
 
def hpss_masks_from_magnitude(
    mag: np.ndarray,
    harmonic_kernel: int = 31,
    percussive_kernel: int = 31,
    margin: float = 1.0,
    mask_eps: float = 1e-8
) -> dict:
    """
    Compute HPSS-based harmonic, percussive, and residual masks from a
    magnitude spectrogram.
 
    Harmonic components are locally smooth along the time axis; percussive
    components are locally smooth along the frequency axis.  Median filtering
    in the respective dimension isolates each type.
 
    Parameters
    ----------
    mag               : Magnitude spectrogram, shape (freq_bins, time_frames).
    harmonic_kernel   : Median filter length in time (must be odd).
    percussive_kernel : Median filter length in frequency (must be odd).
    margin            : Soft-mask margin.  Values > 1 require a component to
                        exceed the other by this factor before claiming a bin.
    mask_eps          : Numerical stability constant.
 
    Returns
    -------
    dict with keys:
        harmonic_mag    – median-filtered harmonic estimate
        percussive_mag  – median-filtered percussive estimate
        harmonic_mask   – soft mask emphasising harmonic energy
        percussive_mask – soft mask emphasising percussive energy
        residual_mask   – energy claimed by neither component
        vocal_mask      – harmonic_mask (singing voice is predominantly harmonic)
    """
 
    harmonic_kernel   = _ensure_odd(harmonic_kernel,   "harmonic_kernel")
    percussive_kernel = _ensure_odd(percussive_kernel, "percussive_kernel")
 
    # Median filter along time axis  → harmonic model (horizontal continuity)
    harmonic_mag = median_filter(mag, size=(1, harmonic_kernel))
 
    # Median filter along frequency axis → percussive model (vertical continuity)
    percussive_mag = median_filter(mag, size=(percussive_kernel, 1))
 
    # Wiener-like soft masks with optional margin
    # A bin is assigned to the harmonic component when:
    #   harmonic_mag  >= margin * percussive_mag
    # and vice versa.  The margin parameter controls aggressiveness.
    h = harmonic_mag    ** 2
    p = percussive_mag  ** 2
    total = h + p + mask_eps
 
    harmonic_mask   = np.clip(h / total, 0.0, 1.0)
    percussive_mask = np.clip(p / total, 0.0, 1.0)
 
    if margin > 1.0:
        # Hard-threshold version: only assign bin if dominant by the margin.
        h_dominant = harmonic_mag   >= margin * percussive_mag
        p_dominant = percussive_mag >= margin * harmonic_mag
 
        harmonic_mask_hard   = np.where(h_dominant, harmonic_mask, 0.0)
        percussive_mask_hard = np.where(p_dominant, percussive_mask, 0.0)
        residual_mask        = np.clip(1.0 - harmonic_mask_hard - percussive_mask_hard, 0.0, 1.0)
 
        harmonic_mask   = harmonic_mask_hard
        percussive_mask = percussive_mask_hard
    else:
        residual_mask = np.zeros_like(harmonic_mask)
 
    # Vocal mask: singing voice is predominantly harmonic, so we expose the
    # harmonic mask as the vocal-oriented estimate.  The residual (if any) is
    # split equally between the two components as a conservative heuristic.
    vocal_mask = harmonic_mask + 0.5 * residual_mask
    vocal_mask = np.clip(vocal_mask, 0.0, 1.0)
 
    return {
        "harmonic_mag":    harmonic_mag,
        "percussive_mag":  percussive_mag,
        "harmonic_mask":   harmonic_mask,
        "percussive_mask": percussive_mask,
        "residual_mask":   residual_mask,
        "vocal_mask":      vocal_mask,
    }
 
 
def hpss_vocal_mask_from_audio(
    y_mix: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    harmonic_kernel: int = 31,
    percussive_kernel: int = 31,
    margin: float = 1.0,
    mask_eps: float = 1e-8
) -> dict:
    """
    Run HPSS from audio and return a vocal-oriented harmonic mask.
 
    Parameters
    ----------
    y_mix             : Mono mixture waveform.
    sr                : Sample rate.
    n_fft / hop_length / win_length : STFT parameters.
    harmonic_kernel   : Median filter length in time.
    percussive_kernel : Median filter length in frequency.
    margin            : Soft-mask margin (see hpss_masks_from_magnitude).
    mask_eps          : Numerical stability constant.
 
    Returns
    -------
    dict with keys:
        mix_feat        – output of compute_stft_features
        harmonic_mag    – harmonic magnitude estimate
        percussive_mag  – percussive magnitude estimate
        harmonic_mask   – soft harmonic mask
        percussive_mask – soft percussive mask
        residual_mask   – residual mask
        vocal_mask      – soft vocal mask (harmonic-oriented)
        background_mask – complement of vocal_mask
    """
    mix_feat = compute_stft_features(
        y_mix, sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
 
    hpss = hpss_masks_from_magnitude(
        mix_feat["mag"],
        harmonic_kernel=harmonic_kernel,
        percussive_kernel=percussive_kernel,
        margin=margin,
        mask_eps=mask_eps
    )
 
    hpss["mix_feat"]        = mix_feat
    hpss["background_mask"] = 1.0 - hpss["vocal_mask"]
 
    return hpss
 
 
def hpss_mask_metrics(
    estimated_vocal_mask: np.ndarray,
    ideal_vocal_mask: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Lightweight mask-quality metrics for HPSS vocal mask evaluation.
 
    Identical in structure to repet_mask_metrics and rpca_mask_metrics so
    results are directly comparable across all three unsupervised methods.
    """
    est = np.asarray(estimated_vocal_mask, dtype=np.float64)
    ref = np.asarray(ideal_vocal_mask, dtype=np.float64)
 
    if est.shape != ref.shape:
        raise ValueError(f"Shape mismatch: est {est.shape}, ref {ref.shape}")
 
    mae = float(np.mean(np.abs(est - ref)))
    mse = float(np.mean((est - ref) ** 2))
 
    est_bin = est >= threshold
    ref_bin = ref >= threshold
 
    tp = np.sum(est_bin & ref_bin)
    fp = np.sum(est_bin & ~ref_bin)
    fn = np.sum(~est_bin & ref_bin)
 
    precision = float(tp / (tp + fp + 1e-8))
    recall    = float(tp / (tp + fn + 1e-8))
    f1        = float(2 * precision * recall / (precision + recall + 1e-8))
 
    corr = (
        float(np.corrcoef(est.ravel(), ref.ravel())[0, 1])
        if np.std(est) > 0 and np.std(ref) > 0
        else 0.0
    )
 
    return {
        "mae": mae,
        "mse": mse,
        "precision@thr": precision,
        "recall@thr": recall,
        "f1@thr": f1,
        "corr": corr,
    }
 
 
def save_hpss_outputs(
    outdir: str,
    track_name: str,
    mix_mag_db: np.ndarray,
    vocal_mask: np.ndarray,
    ideal_mask: np.ndarray,
    harmonic_mag: np.ndarray,
    percussive_mag: np.ndarray,
    sr: int,
    hop_length: int
) -> None:
    """Save HPSS spectrograms and masks as PNG files."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_name = track_name.replace("/", "_").replace("\\", "_")
 
    plot_spectrogram(
        mix_mag_db, sr, hop_length,
        title=f"Mixture Spectrogram: {track_name}",
        save_path=str(outdir / f"{safe_name}_hpss_mix_spec.png"),
        db_scale=True
    )
 
    plot_spectrogram(
        vocal_mask, sr, hop_length,
        title=f"HPSS Vocal Mask (Harmonic): {track_name}",
        save_path=str(outdir / f"{safe_name}_hpss_vocal_mask.png"),
        db_scale=False
    )
 
    plot_spectrogram(
        ideal_mask, sr, hop_length,
        title=f"Ideal Vocal Mask: {track_name}",
        save_path=str(outdir / f"{safe_name}_hpss_ideal_vocal_mask.png"),
        db_scale=False
    )
 
    harmonic_db = librosa.amplitude_to_db(harmonic_mag, ref=np.max)
    plot_spectrogram(
        harmonic_db, sr, hop_length,
        title=f"HPSS Harmonic Component: {track_name}",
        save_path=str(outdir / f"{safe_name}_hpss_harmonic.png"),
        db_scale=True
    )
 
    percussive_db = librosa.amplitude_to_db(percussive_mag + 1e-8, ref=np.max)
    plot_spectrogram(
        percussive_db, sr, hop_length,
        title=f"HPSS Percussive Component: {track_name}",
        save_path=str(outdir / f"{safe_name}_hpss_percussive.png"),
        db_scale=True
    )

"""
-------------------------------------------------------------------------------
GBDT utility functions:
-------------------------------------------------------------------------------
"""
def build_gbdt_feature_matrix(
    mix_feat: dict,
    hpss_mask: np.ndarray,
    repet_mask: np.ndarray,
    rpca_mask: np.ndarray,
    sr: int,
    eps: float = 1e-8
) -> tuple[np.ndarray, list[str], tuple[int, int]]:
    """
    Build a per-time-frequency-bin feature matrix for GBDT learning.

    Returns
    -------
    X : np.ndarray, shape (n_bins * n_frames, n_features)
    feature_names : list[str]
    original_shape : tuple[int, int]
        (n_bins, n_frames), useful for reshaping predictions back to a mask
    """
    mag = np.asarray(mix_feat["mag"], dtype=np.float32)
    log_mag = np.asarray(mix_feat["log_mag"], dtype=np.float32)
    harmonic = np.asarray(mix_feat["harmonic"], dtype=np.float32)
    percussive = np.asarray(mix_feat["percussive"], dtype=np.float32)

    hpss_mask = np.asarray(hpss_mask, dtype=np.float32)
    repet_mask = np.asarray(repet_mask, dtype=np.float32)
    rpca_mask = np.asarray(rpca_mask, dtype=np.float32)

    if not (mag.shape == hpss_mask.shape == repet_mask.shape == rpca_mask.shape):
        raise ValueError(
            f"Shape mismatch: mag={mag.shape}, hpss={hpss_mask.shape}, "
            f"repet={repet_mask.shape}, rpca={rpca_mask.shape}"
        )

    n_bins, n_frames = mag.shape

    freq_idx = np.arange(n_bins, dtype=np.float32)[:, None]
    time_idx = np.arange(n_frames, dtype=np.float32)[None, :]

    freq_norm = np.broadcast_to(freq_idx / max(n_bins - 1, 1), (n_bins, n_frames))
    time_norm = np.broadcast_to(time_idx / max(n_frames - 1, 1), (n_bins, n_frames))

    freq_hz = librosa.fft_frequencies(sr=sr, n_fft=mix_feat["n_fft"]).astype(np.float32)
    freq_hz = np.broadcast_to(freq_hz[:, None], (n_bins, n_frames))
    freq_hz_norm = freq_hz / max(sr / 2.0, eps)

    harmonic_bin_ratio = harmonic / (harmonic + percussive + eps)

    def frame_to_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1 or x.shape[0] != n_frames:
            raise ValueError(f"Expected frame feature of length {n_frames}, got {x.shape}")
        return np.broadcast_to(x[None, :], (n_bins, n_frames))

    rms_2d = frame_to_2d(mix_feat["rms"])
    centroid_2d = frame_to_2d(mix_feat["centroid"]) / max(sr / 2.0, eps)
    bandwidth_2d = frame_to_2d(mix_feat["bandwidth"]) / max(sr / 2.0, eps)
    rolloff_2d = frame_to_2d(mix_feat["rolloff"]) / max(sr / 2.0, eps)
    flatness_2d = frame_to_2d(mix_feat["flatness"])
    hp_ratio_2d = frame_to_2d(mix_feat["hp_ratio"])

    candidate_mean = (hpss_mask + repet_mask + rpca_mask) / 3.0
    candidate_std = np.std(
        np.stack([hpss_mask, repet_mask, rpca_mask], axis=0),
        axis=0
    ).astype(np.float32)

    feature_maps = [
        freq_norm,
        time_norm,
        freq_hz_norm,
        log_mag,
        harmonic_bin_ratio,
        rms_2d,
        centroid_2d,
        bandwidth_2d,
        rolloff_2d,
        flatness_2d,
        hp_ratio_2d,
        hpss_mask,
        repet_mask,
        rpca_mask,
        candidate_mean,
        candidate_std,
    ]

    feature_names = [
        "freq_norm",
        "time_norm",
        "freq_hz_norm",
        "log_mag",
        "harmonic_bin_ratio",
        "frame_rms",
        "frame_centroid_norm",
        "frame_bandwidth_norm",
        "frame_rolloff_norm",
        "frame_flatness",
        "frame_hp_ratio",
        "hpss_mask",
        "repet_mask",
        "rpca_mask",
        "candidate_mean",
        "candidate_std",
    ]

    X = np.column_stack([f.reshape(-1) for f in feature_maps]).astype(np.float32)
    return X, feature_names, (n_bins, n_frames)


def sample_training_bins(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = 200_000,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample training bins so memory stays manageable.

    Uses a simple balanced strategy so the model does not get dominated by
    near-zero IRM bins.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    n = y.shape[0]
    if n <= max_samples:
        return X, y

    rng = np.random.default_rng(random_state)

    low_idx = np.where(y < 0.05)[0]
    mid_idx = np.where((y >= 0.05) & (y < 0.95))[0]
    high_idx = np.where(y >= 0.95)[0]

    groups = [low_idx, mid_idx, high_idx]
    nonempty_groups = [g for g in groups if len(g) > 0]

    per_group = max_samples // max(len(nonempty_groups), 1)
    chosen = []

    for g in nonempty_groups:
        take = min(len(g), per_group)
        chosen.append(rng.choice(g, size=take, replace=False))

    chosen = np.concatenate(chosen) if len(chosen) > 0 else np.array([], dtype=int)

    remaining = max_samples - len(chosen)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n), chosen, assume_unique=False)
        if len(pool) > 0:
            extra_take = min(len(pool), remaining)
            extra = rng.choice(pool, size=extra_take, replace=False)
            chosen = np.concatenate([chosen, extra])

    rng.shuffle(chosen)
    return X[chosen], y[chosen]


def train_gbdt_mask_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.05,
    max_depth: int = 8,
    max_iter: int = 300,
    min_samples_leaf: int = 50,
    validation_fraction: float = 0.1,
    random_state: int = 42
):
    """
    Train a gradient-boosted decision tree regressor to predict the ideal
    vocal mask.
    """
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        validation_fraction=validation_fraction,
        early_stopping=True,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_gbdt_mask(
    model,
    mix_feat: dict,
    hpss_mask: np.ndarray,
    repet_mask: np.ndarray,
    rpca_mask: np.ndarray,
    sr: int
) -> np.ndarray:
    """
    Predict a vocal mask from mixture features and unsupervised candidate masks.
    """
    X, _, original_shape = build_gbdt_feature_matrix(
        mix_feat=mix_feat,
        hpss_mask=hpss_mask,
        repet_mask=repet_mask,
        rpca_mask=rpca_mask,
        sr=sr
    )

    y_pred = model.predict(X)
    mask = y_pred.reshape(original_shape)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return mask

def gbdt_mask_metrics(
    estimated_vocal_mask: np.ndarray,
    ideal_vocal_mask: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Lightweight mask-quality metrics for GBDT vocal mask evaluation.

    Kept identical in structure to the HPSS / REPET / RPCA mask metrics
    so results are directly comparable across all methods.
    """
    est = np.asarray(estimated_vocal_mask, dtype=np.float64)
    ref = np.asarray(ideal_vocal_mask, dtype=np.float64)

    if est.shape != ref.shape:
        raise ValueError(f"Shape mismatch: est {est.shape}, ref {ref.shape}")

    mae = float(np.mean(np.abs(est - ref)))
    mse = float(np.mean((est - ref) ** 2))

    est_bin = est >= threshold
    ref_bin = ref >= threshold

    tp = np.sum(est_bin & ref_bin)
    fp = np.sum(est_bin & ~ref_bin)
    fn = np.sum(~est_bin & ref_bin)

    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))

    corr = (
        float(np.corrcoef(est.ravel(), ref.ravel())[0, 1])
        if np.std(est) > 0 and np.std(ref) > 0
        else 0.0
    )

    return {
        "mae": mae,
        "mse": mse,
        "precision@thr": precision,
        "recall@thr": recall,
        "f1@thr": f1,
        "corr": corr,
    }