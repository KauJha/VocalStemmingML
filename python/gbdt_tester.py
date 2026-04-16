import argparse
from pathlib import Path

import joblib
import pandas as pd

import utils
def main():
    script_dir = Path(__file__).resolve().parent
    default_outdir = script_dir.parent / "outputs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["sample", "local"], default="sample")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--track", type=str, default=None)
    parser.add_argument("--subset", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--is_wav", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=2048)
    parser.add_argument("--irm_p", type=int, choices=[1, 2], default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--outdir", type=str, default=str(default_outdir))
    parser.add_argument("--list_tracks", action="store_true")
    args = parser.parse_args()

    db = utils.get_database(
        mode=args.mode,
        root=args.root,
        subset=args.subset,
        is_wav=args.is_wav,
    )

    if args.list_tracks:
        for name in utils.list_track_names(db):
            print(name)
        return

    bundle = joblib.load(args.model_path)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

    track = utils.select_track(db, args.track)
    data = utils.load_track_audio(track)

    sr = data["sample_rate"]
    y_mix = data["mixture_mono"]
    y_voc = data["vocals_mono"]
    y_acc = data["accompaniment_mono"]

    mix_feat = utils.compute_stft_features(
        y_mix,
        sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    hpss = utils.hpss_vocal_mask_from_audio(
        y_mix=y_mix,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    repet = utils.repet_vocal_mask_from_audio(
        y_mix=y_mix,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    rpca = utils.rpca_vocal_mask_from_audio(
        y_mix=y_mix,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    voc_feat = utils.compute_stft_features(
        y_voc,
        sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    acc_feat = utils.compute_stft_features(
        y_acc,
        sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    ideal_mask = utils.ideal_ratio_mask(
        vocals_mag=voc_feat["mag"],
        accomp_mag=acc_feat["mag"],
        p=args.irm_p,
    )

    gbdt_mask = utils.predict_gbdt_mask(
        model=model,
        mix_feat=mix_feat,
        hpss_mask=hpss["vocal_mask"],
        repet_mask=repet["vocal_mask"],
        rpca_mask=rpca["vocal_mask"],
        sr=sr,
    )

    stem_estimates = utils.estimate_stems_from_mask(
        mix_feat=mix_feat,
        vocal_mask=gbdt_mask,
        length=len(y_mix),
    )

    stem_paths = utils.save_estimated_stems(
        outdir=args.outdir,
        track_name=data["track_name"],
        stems=stem_estimates,
        sr=sr,
        prefix="gbdt",
    )

    mask_scores = utils.gbdt_mask_metrics(
        estimated_vocal_mask=gbdt_mask,
        ideal_vocal_mask=ideal_mask,
        threshold=args.threshold,
    )

    vocal_stem_scores = utils.stem_metrics(
        estimated_stem=stem_estimates["vocals_est"],
        reference_stem=y_voc,
    )

    utils.save_gbdt_outputs(
        outdir=args.outdir,
        track_name=data["track_name"],
        mix_mag_db=mix_feat["mag_db"],
        vocal_mask=gbdt_mask,
        ideal_mask=ideal_mask,
        sr=sr,
        hop_length=args.hop_length,
    )

    metrics_df = pd.DataFrame([{
        "track": data["track_name"],
        "mode": args.mode,
        "model_path": str(args.model_path),
        **{f"mask_{k}": v for k, v in mask_scores.items()},
        **{f"vocals_{k}": v for k, v in vocal_stem_scores.items()}
    }])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / f"{data['track_name'].replace('/', '_')}_gbdt_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Track: {data['track_name']}")
    print(f"Sample rate: {sr}")
    print(f"Model path: {args.model_path}")
    print("Mask metrics:")
    for k, v in mask_scores.items():
        print(f"  {k}: {v:.6f}")

    print("Vocal stem metrics:")
    for k, v in vocal_stem_scores.items():
        print(f"  {k}: {v:.6f}")

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved estimated stems to: {stem_paths['vocals_path']}")


if __name__ == "__main__":
    main()
