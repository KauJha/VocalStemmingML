import argparse
from pathlib import Path
import joblib
import numpy as np
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--subset", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--is_wav", action="store_true")
    parser.add_argument("--max_tracks", type=int, default=None)
    parser.add_argument("--max_samples_per_track", type=int, default=100_000)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=2048)
    parser.add_argument("--irm_p", type=int, choices=[1, 2], default=1)
    parser.add_argument("--model_out", type=str, default="gbdt_mask_model.joblib")
    args = parser.parse_args()

    db = utils.get_database(
        mode="local",
        root=args.root,
        subset=args.subset,
        is_wav=args.is_wav
    )

    X_parts = []
    y_parts = []
    feature_names = None

    for track_idx, track in enumerate(db):
        if args.max_tracks is not None and track_idx >= args.max_tracks:
            break

        data = utils.load_track_audio(track)

        sr = data["sample_rate"]
        y_mix = data["mixture_mono"]
        y_voc = data["vocals_mono"]
        y_acc = data["accompaniment_mono"]

        mix_feat = utils.compute_stft_features(
            y_mix, sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length
        )

        hpss = utils.hpss_vocal_mask_from_audio(
            y_mix=y_mix, sr=sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length
        )

        repet = utils.repet_vocal_mask_from_audio(
            y_mix=y_mix, sr=sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length
        )

        rpca = utils.rpca_vocal_mask_from_audio(
            y_mix=y_mix, sr=sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length
        )

        voc_feat = utils.compute_stft_features(
            y_voc, sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length
        )

        acc_feat = utils.compute_stft_features(
            y_acc, sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length
        )

        ideal_mask = utils.ideal_ratio_mask(
            vocals_mag=voc_feat["mag"],
            accomp_mag=acc_feat["mag"],
            p=args.irm_p
        )

        X_track, feature_names, _ = utils.build_gbdt_feature_matrix(
            mix_feat=mix_feat,
            hpss_mask=hpss["vocal_mask"],
            repet_mask=repet["vocal_mask"],
            rpca_mask=rpca["vocal_mask"],
            sr=sr
        )

        y_track = ideal_mask.reshape(-1).astype(np.float32)

        X_track, y_track = utils.sample_training_bins(
            X_track,
            y_track,
            max_samples=args.max_samples_per_track,
            random_state=42 + track_idx
        )

        X_parts.append(X_track)
        y_parts.append(y_track)

        print(
            f"[{track_idx+1}] {data['track_name']}  "
            f"sampled_bins={len(y_track)}"
        )

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)

    print(f"Training matrix shape: {X_train.shape}")
    model = utils.train_gbdt_mask_regressor(X_train, y_train)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
        },
        args.model_out
    )
    print(f"Saved model to: {args.model_out}")


if __name__ == "__main__":
    main()