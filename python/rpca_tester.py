import argparse
from pathlib import Path
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
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=2048)
    parser.add_argument("--irm_p", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--lam",
        type=float,
        default=None,
        help=(
            "RPCA regularization parameter lambda. "
            "Controls the balance between low-rank (accompaniment) and sparse (vocal) "
            "components. Defaults to 1/sqrt(max(freq_bins, time_frames)) if not set."
        )
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum RPCA IALM iterations."
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="RPCA convergence tolerance on relative Frobenius residual."
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--outdir", type=str, default=str(default_outdir))
    parser.add_argument("--list_tracks", action="store_true")
    args = parser.parse_args()

    db = utils.get_database(
        mode=args.mode,
        root=args.root,
        subset=args.subset,
        is_wav=args.is_wav
    )

    if args.list_tracks:
        for name in utils.list_track_names(db):
            print(name)
        return

    track = utils.select_track(db, args.track)
    data = utils.load_track_audio(track)

    sr = data["sample_rate"]
    y_mix = data["mixture_mono"]
    y_voc = data["vocals_mono"]
    y_acc = data["accompaniment_mono"]

    rpca = utils.rpca_vocal_mask_from_audio(
        y_mix=y_mix,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        lam=args.lam,
        max_iter=args.max_iter,
        tol=args.tol
    )

    mix_feat = rpca["mix_feat"]

    voc_feat = utils.compute_stft_features(
        y_voc,
        sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )

    acc_feat = utils.compute_stft_features(
        y_acc,
        sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )

    ideal_mask = utils.ideal_ratio_mask(
        vocals_mag=voc_feat["mag"],
        accomp_mag=acc_feat["mag"],
        p=args.irm_p
    )

    est_mask = rpca["vocal_mask"]

    metrics = utils.rpca_mask_metrics(
        estimated_vocal_mask=est_mask,
        ideal_vocal_mask=ideal_mask,
        threshold=args.threshold
    )

    utils.save_rpca_outputs(
        outdir=args.outdir,
        track_name=data["track_name"],
        mix_mag_db=mix_feat["mag_db"],
        vocal_mask=est_mask,
        ideal_mask=ideal_mask,
        low_rank_mag=rpca["low_rank_mag"],
        sparse_mag=rpca["sparse_mag"],
        sr=sr,
        hop_length=args.hop_length
    )

    metrics_df = pd.DataFrame([{
        "track": data["track_name"],
        "mode": args.mode,
        "lam": rpca["lam_used"],
        "n_iter": rpca["n_iter"],
        "converged": rpca["converged"],
        **metrics
    }])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / f"{data['track_name'].replace('/', '_')}_rpca_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Track: {data['track_name']}")
    print(f"Sample rate: {sr}")
    print(f"Lambda used: {rpca['lam_used']:.6f}")
    print(f"RPCA iterations: {rpca['n_iter']}  converged: {rpca['converged']}")
    print("Mask metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()