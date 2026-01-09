from __future__ import annotations

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import os
import sys
                                                                      

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

try:
    import h5py
except Exception as e:
    raise RuntimeError("h5py is required for reading fastMRI .h5 files. Please `pip install h5py`.") from e


                                                                        
REF_KEYS: List[str] = [
    "reconstruction_rss",
    "reconstruction_esc",
    "reconstruction",
    "target",
    "reconstruction_ref",
]


def _find_ref_key(f: h5py.File) -> str:
    for k in REF_KEYS:
        if k in f:
            return k
    keys = list(f.keys())
    raise KeyError(f"No reference image key found. Tried {REF_KEYS}. Available keys: {keys}")


def _load_ref_slice(h5_path: Path, slice_idx: int) -> Tuple[np.ndarray, int]:
    with h5py.File(h5_path, "r") as f:
        key = _find_ref_key(f)
        vol = f[key]           
        S = int(vol.shape[0])
        if slice_idx < 0 or slice_idx >= S:
            raise IndexError(f"slice index {slice_idx} out of range for {h5_path} with S={S}")
        return np.asarray(vol[slice_idx]), S


def _ref_energy(ref: np.ndarray, metric: str = "mean_abs") -> float:
    a = np.abs(ref).astype(np.float64)
    if metric == "mean_abs":
        return float(a.mean())
    if metric == "p95_abs":
        return float(np.quantile(a, 0.95))
    raise ValueError(f"Unknown energy metric: {metric}")


def _quantile_thr(y: np.ndarray, tail: str, q: float) -> float:
    y = np.asarray(y, dtype=np.float64)
    if tail == "high":
        return float(np.quantile(y, 1.0 - q))
    return float(np.quantile(y, q))


def _pick_unique_files(df_sorted: pd.DataFrame, n: int, used_files: set) -> pd.DataFrame:
    out_rows = []
    for _, r in df_sorted.iterrows():
        if r["file"] in used_files:
            continue
        out_rows.append(r)
        used_files.add(r["file"])
        if len(out_rows) >= n:
            break
    if len(out_rows) == 0:
        return df_sorted.head(0).copy()
    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)

           
                                                            
                                                                                     
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="If --out_csv is a directory, write fig2_manifest_{run_id}.csv inside it. If omitted, defaults to results_csv stem.",
    )

    ap.add_argument("--score_col", type=str, default="pcr_fail_mean")
    ap.add_argument("--y_col", type=str, default="nrmse")

                               
    ap.add_argument("--fail_quantile", type=float, default=0.10)
    ap.add_argument("--fail_tail", type=str, default="high", choices=["high", "low"])

                                                                         
    ap.add_argument("--energy_metric", type=str, default="mean_abs", choices=["mean_abs", "p95_abs"])
    ap.add_argument("--energy_keep_top", type=float, default=0.70)
    ap.add_argument("--energy_keep_top_fail", type=float, default=None)
    ap.add_argument("--energy_keep_top_ok", type=float, default=None)

                                                             
    ap.add_argument("--disable_slice_frac_filter", action="store_true")
    ap.add_argument("--slice_frac_min", type=float, default=0.20)
    ap.add_argument("--slice_frac_max", type=float, default=0.80)

                                                   
    ap.add_argument("--ok_y_keep_bottom", type=float, default=0.30)
                                                                                   
    ap.add_argument("--fail_y_keep_top", type=float, default=1.00)

    ap.add_argument("--n_per_group", type=int, default=1)
    ap.add_argument("--unique_files", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.results_csv)

    required = {"file", "slice", "accel", "center_fraction", args.score_col, args.y_col}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")

    data_root = Path(args.data_root)

                                                                       
    ref_energy = []
    slice_frac = []
    num_slices = []
    for f, s in zip(df["file"].astype(str), df["slice"].astype(int)):
        h5_path = (Path(f) if Path(f).exists() else (data_root / f))
        ref, S = _load_ref_slice(h5_path, int(s))
        e = _ref_energy(ref, metric=args.energy_metric)
        ref_energy.append(e)
        num_slices.append(S)
        sf = 0.0 if S <= 1 else float(s) / float(S - 1)
        slice_frac.append(sf)

    df = df.copy()
    df["ref_energy"] = np.asarray(ref_energy, dtype=np.float64)
    df["slice_frac"] = np.asarray(slice_frac, dtype=np.float64)
    df["num_slices"] = np.asarray(num_slices, dtype=np.int32)

                                             
    if not args.disable_slice_frac_filter:
        df = df[(df["slice_frac"] >= args.slice_frac_min) & (df["slice_frac"] <= args.slice_frac_max)].copy()

                                              
    y = df[args.y_col].to_numpy(dtype=np.float64)
    thr_global = _quantile_thr(y, args.fail_tail, args.fail_quantile)
    if args.fail_tail == "high":
        is_fail = y >= thr_global
        fail_desc = f"{args.y_col} top {args.fail_quantile:.2f}"
    else:
        is_fail = y <= thr_global
        fail_desc = f"{args.y_col} bottom {args.fail_quantile:.2f}"
    df["is_fail"] = is_fail

                                           
    keep_fail = args.energy_keep_top_fail if args.energy_keep_top_fail is not None else args.energy_keep_top
    keep_ok = args.energy_keep_top_ok if args.energy_keep_top_ok is not None else args.energy_keep_top

    def energy_filter(sub: pd.DataFrame, keep_top: float) -> Tuple[pd.DataFrame, float]:
        if len(sub) == 0:
            return sub, float("nan")
        thr = float(np.quantile(sub["ref_energy"].to_numpy(dtype=np.float64), 1.0 - keep_top))
        return sub[sub["ref_energy"] >= thr].copy(), thr

    df_fail = df[df["is_fail"]].copy()
    df_ok = df[~df["is_fail"]].copy()

    df_fail, e_thr_fail = energy_filter(df_fail, keep_fail)
    df_ok, e_thr_ok = energy_filter(df_ok, keep_ok)

                                                                             
    if len(df_ok) > 0:
        y_ok = df_ok[args.y_col].to_numpy(dtype=np.float64)
        thr_ok_keep = float(np.quantile(y_ok, args.ok_y_keep_bottom))
        if args.fail_tail == "high":
            df_ok = df_ok[df_ok[args.y_col] <= thr_ok_keep].copy()
        else:
            df_ok = df_ok[df_ok[args.y_col] >= thr_ok_keep].copy()

                                                                              
    if args.fail_y_keep_top < 1.0 and len(df_fail) > 0:
        y_fail = df_fail[args.y_col].to_numpy(dtype=np.float64)
        thr_fail_keep = float(np.quantile(y_fail, 1.0 - args.fail_y_keep_top)) if args.fail_tail == "high" else float(np.quantile(y_fail, args.fail_y_keep_top))
        if args.fail_tail == "high":
            df_fail = df_fail[df_fail[args.y_col] >= thr_fail_keep].copy()
        else:
            df_fail = df_fail[df_fail[args.y_col] <= thr_fail_keep].copy()

                                                                                                   
    score = df[args.score_col].to_numpy(dtype=np.float64)

                                                                            
                                                     
    score_thr = float(np.median(score[np.isfinite(score)])) if np.isfinite(score).any() else 0.0

                                          
    def with_score(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["is_high_score"] = sub[args.score_col].to_numpy(dtype=np.float64) >= score_thr
        return sub

    df_fail = with_score(df_fail)
    df_ok = with_score(df_ok)

    tp = df_fail[df_fail["is_high_score"]].copy()
    fn = df_fail[~df_fail["is_high_score"]].copy()
    fp = df_ok[df_ok["is_high_score"]].copy()
    tn = df_ok[~df_ok["is_high_score"]].copy()

                                                       
                                                     
                                                     
    tp = tp.sort_values(by=[args.y_col, args.score_col], ascending=[False, False])
    fn = fn.sort_values(by=[args.y_col, args.score_col], ascending=[False, True])
    fp = fp.sort_values(by=[args.y_col, args.score_col], ascending=[True, False])
    tn = tn.sort_values(by=[args.y_col, args.score_col], ascending=[True, True])

    used_files: set = set()

    def pick(df_sorted: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0:
            return df_sorted.head(0).copy()
        if len(df_sorted) == 0:
            return df_sorted.head(0).copy()
        if args.unique_files:
            return _pick_unique_files(df_sorted, n, used_files)
        return df_sorted.head(n).copy()

    tp_sel = pick(tp, args.n_per_group); tp_sel["group"] = "TP (fail, high score)"
    fn_sel = pick(fn, args.n_per_group); fn_sel["group"] = "FN (fail, low score)"
    fp_sel = pick(fp, args.n_per_group); fp_sel["group"] = "FP (non-fail, high score)"
    tn_sel = pick(tn, args.n_per_group); tn_sel["group"] = "TN (non-fail, low score)"

    out = pd.concat([tp_sel, fn_sel, fp_sel, tn_sel], axis=0, ignore_index=True)

    cols_first = [
        "group", "file", "slice", "accel", "center_fraction",
        args.y_col, args.score_col, "ref_energy", "slice_frac", "is_fail",
    ]
    cols_rest = [c for c in out.columns if c not in cols_first]
    out = out[cols_first + cols_rest]

    out_base = Path(args.out_csv)
    if out_base.suffix.lower() == ".csv":
        out_path = out_base
    else:
        run_id = args.run_id or Path(args.results_csv).stem
        out_path = out_base / f"fig2_manifest_{run_id}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[OK] Saved manifest: {out_path}")
    print(f"[INFO] fail definition: {fail_desc}, thr_global({args.y_col})={thr_global:.6f}")
    print(f"[INFO] energy metric: {args.energy_metric}")
    print(f"[INFO] energy keep top: fail={keep_fail:.2f} (thr={e_thr_fail:.6e}), non-fail={keep_ok:.2f} (thr={e_thr_ok:.6e})")
    if not args.disable_slice_frac_filter:
        print(f"[INFO] slice_frac keep: [{args.slice_frac_min:.2f}, {args.slice_frac_max:.2f}]")
    print(f"[INFO] y-keep: ok bottom={args.ok_y_keep_bottom:.2f}, fail keep frac={args.fail_y_keep_top:.2f}")
    if args.unique_files:
        print("[INFO] unique files in manifest:", out["file"].nunique())


if __name__ == "__main__":
    main()