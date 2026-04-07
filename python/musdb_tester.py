import argparse
from pathlib import Path

import utils as utils

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_OUTDIR = SCRIPT_DIR.parent / "outputs"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "local"],
        default="sample",
        help="Use official 7-second sample snippets or local MUSDB18 dataset"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to local MUSDB18 root (required if mode=local)"
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Exact track name. If omitted, the first track is used."
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Subset for local mode"
    )
    parser.add_argument(
        "--is_wav",
        action="store_true",
        help="Use this if your local MUSDB18 is stored as WAV files"
    )
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=2048)
    parser.add_argument("--irm_p", type=int, default=1, choices=[1, 2])
    parser.add_argument("--vocal_threshold", type=float, default=0.6)
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument(
        "--list_tracks",
        action="store_true",
        help="List available track names and exit"
    )

    args = parser.parse_args()

    db = utils.get_database(
        mode=args.mode,
        root=args.root,
        subset=args.subset,
        is_wav=args.is_wav
    )

    if args.list_tracks:
        print("Available tracks:")
        for name in utils.list_track_names(db):
            print(name)
        return

    track = utils.select_track(db, args.track)
    data = utils.load_track_audio(track)

    sr = data["sample_rate"]
    accomp_mix = data["accompaniment_mono"]
    y_voc = data["vocals_mono"]
    y_mix = data["mixture_mono"]

    mix_feat = utils.compute_stft_features(
        y_mix,
        sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )

    accomp_feat = utils.compute_stft_features(
        accomp_mix,
        sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )

    voc_feat = utils.compute_stft_features(
        y_voc,
        sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )

    irm = utils.ideal_ratio_mask(
        vocals_mag=voc_feat["mag"],
        accomp_mag=accomp_feat["mag"],
        p=args.irm_p
    )
    print("IRM min/max/mean:", irm.min(), irm.max(), irm.mean())

    feature_df = utils.build_frame_feature_table(mix_feat)

    vocal_regions = utils.summarize_high_vocal_regions(
        irm,
        sr=sr,
        hop_length=args.hop_length,
        threshold=args.vocal_threshold
    )

    utils.save_outputs(
        outdir=args.outdir,
        track_name=data["track_name"],
        mix_feat=mix_feat,
        voc_feat=voc_feat,
        irm=irm,
        feature_df=feature_df,
        vocal_regions=vocal_regions,
        sr=sr,
        hop_length=args.hop_length
    )

    print(f"\nTrack loaded: {data['track_name']}")
    print(f"Mode: {args.mode}")
    print(f"Sample rate: {sr}")
    print(f"Mixture mono shape: {y_mix.shape}")
    print(f"Vocals mono shape: {y_voc.shape}")

    print("\nTop vocal-heavy frames:")
    if len(vocal_regions) == 0:
        print("No frames exceeded the chosen vocal threshold.")
    else:
        print(vocal_regions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()