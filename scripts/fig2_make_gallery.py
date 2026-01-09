import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse, os, sys, subprocess
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_prefix", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1)
                           
    ap.add_argument("--mask_seed", type=int, default=0)
    ap.add_argument("--pcr_smooth_ksize", type=int, default=7)
    ap.add_argument("--pcr_clip_percentile", type=float, default=99.5)
    ap.add_argument("--pcr_topk", type=float, default=0.02)
    ap.add_argument("--style", default="overlay")
    ap.add_argument("--overlay_topq", type=float, default=0.90)
    ap.add_argument("--overlay_alpha", type=float, default=0.75)
    ap.add_argument("--anat_q", type=float, default=0.75)
    ap.add_argument("--no_colorbar", action="store_true")
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    n = len(df)
    stop = n if args.stop < 0 else min(args.stop, n)

    tmp_dir = os.path.join(args.out_dir, "_tmp_manifests", args.run_prefix)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.start, stop):
        tmp_csv = os.path.join(tmp_dir, f"one_{i:04d}.csv")
        df.iloc[[i]].to_csv(tmp_csv, index=False)

        run_id = f"{args.run_prefix}_{i:04d}"
        cmd = [
            sys.executable, "scripts/fig2_make.py",
            "--manifest", tmp_csv,
            "--data_root", args.data_root,
            "--out_dir", args.out_dir,
            "--run_id", run_id,
            "--mask_seed", str(args.mask_seed),
            "--pcr_smooth_ksize", str(args.pcr_smooth_ksize),
            "--pcr_clip_percentile", str(args.pcr_clip_percentile),
            "--pcr_topk", str(args.pcr_topk),
            "--style", args.style,
            "--overlay_topq", str(args.overlay_topq),
            "--overlay_alpha", str(args.overlay_alpha),
            "--anat_q", str(args.anat_q),
            "--dpi", str(args.dpi),
        ]
        if args.no_colorbar:
            cmd.append("--no_colorbar")

        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"[OK] Rendered cases {args.start}..{stop-1} into {args.out_dir}")

if __name__ == "__main__":
    main()