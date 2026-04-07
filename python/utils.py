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
RPCA utility functions:
-------------------------------------------------------------------------------
HPSS utility functions:
-------------------------------------------------------------------------------
GBDT utility functions:
-------------------------------------------------------------------------------
Verification utility functions:
-------------------------------------------------------------------------------
"""
import os

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import musdb
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

    import matplotlib.pyplot as plt

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