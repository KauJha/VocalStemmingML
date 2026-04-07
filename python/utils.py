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
...
-------------------------------------------------------------------------------
etc
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