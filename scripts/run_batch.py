from __future__ import annotations

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse
import glob
import os
import random
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

                                                      

from pcrmap.data_io_fastmri import load_multicoil_h5_slice, build_zf_coil_images
from pcrmap.mri_ops import make_vd_cartesian_mask, estimate_sens_maps_from_acs, sense_combine
from pcrmap.pcrmap import compute_pcr_map, PCRConfig
from pcrmap.metrics import nrmse, psnr, ssim


def _fft2c(img: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(img, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


def _ifft2c(ksp: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


def _infer_dataset_name(data_root: str) -> str:
    p = data_root.lower()
    if "brain" in p:
        return "brain_mc"
    if "knee" in p:
        return "knee_mc"
                           
    return os.path.basename(os.path.normpath(data_root)) or "unknown"


def iter_slices(
    data_root: str,
    total_slices: int,
    seed: int,
    num_cases: Optional[int] = None,
    slices_per_case: Optional[int] = None,
) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(data_root, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {data_root}")

    if (num_cases is None) ^ (slices_per_case is None):
        raise ValueError("Use --num_cases and --slices_per_case together, or use neither.")

    try:
        import h5py                
    except Exception as e:
        raise RuntimeError("h5py is required for reading fastMRI .h5 files. Please `pip install h5py`.") from e

    def _num_slices(fp: str) -> int:
        with h5py.File(fp, "r") as f:
            return int(f["kspace"].shape[0])

    def _sample_slices(fp: str, k: int) -> List[int]:
        ns = _num_slices(fp)
        if ns <= 0:
            return []
        cand = list(range(ns))
        rng.shuffle(cand)
        return cand[:k]

                                       
    if num_cases is not None and slices_per_case is not None:
        rng.shuffle(files)
        files = files[: int(num_cases)]
        per_file = {fp: _sample_slices(fp, int(slices_per_case)) for fp in files}

        pairs: List[Tuple[str, int]] = []
        for i in range(int(slices_per_case)):
            for fp in files:
                if i < len(per_file[fp]):
                    pairs.append((fp, int(per_file[fp][i])))
                                                                                
        target = int(num_cases) * int(slices_per_case)
        if len(pairs) < target:
            pairs = pairs + iter_slices(data_root, target - len(pairs), seed=seed + 12345)
        return pairs[:target]

                     
    total_slices = int(total_slices)
    pairs = []
    seen: set[Tuple[str, int]] = set()
    max_tries = max(10 * total_slices, 1000)
    tries = 0
    while len(pairs) < total_slices and tries < max_tries:
        tries += 1
        fp = rng.choice(files)
        ns = _num_slices(fp)
        if ns <= 0:
            continue
        sidx = int(rng.randrange(ns))
        key = (fp, sidx)
                                        
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

                                                                                                                
    while len(pairs) < total_slices:
        fp = rng.choice(files)
        ns = _num_slices(fp)
        if ns <= 0:
            continue
        sidx = int(rng.randrange(ns))
        pairs.append((fp, sidx))

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Folder containing fastMRI multicoil .h5 files")
    ap.add_argument("--out_csv", type=str, required=True)

    ap.add_argument("--accel", type=int, default=8)
    ap.add_argument("--center_fraction", type=float, default=0.08)

                                                                                          
    ap.add_argument("--seed", type=int, default=0, help="Seed for case/slice sampling only.")
    ap.add_argument("--mask_seed", type=int, default=0, help="Seed for undersampling mask generation (fixed mask).")

    ap.add_argument("--device", type=str, default="cpu")

                       
    ap.add_argument("--num_slices", type=int, default=2000, help="Used only when num_cases/slices_per_case not set")
    ap.add_argument("--num_cases", type=int, default=None)
    ap.add_argument("--slices_per_case", type=int, default=None)

                                                                                      
    ap.add_argument("--pcr_smooth_ksize", type=int, default=7)
    ap.add_argument("--pcr_clip_percentile", type=float, default=99.5)
    ap.add_argument("--pcr_topk", type=float, default=0.02)

                                                                          
    ap.add_argument("--out_manifest", type=str, default="", help="Optional CSV to save sampled (file,slice) list.")

                                                                            
    ap.add_argument("--dataset", type=str, default="", help="Optional dataset name to store in CSV (e.g., brain_mc).")

    args = ap.parse_args()
    device = torch.device(args.device)

    if args.num_cases is not None and args.slices_per_case is not None:
        total = int(args.num_cases) * int(args.slices_per_case)
    else:
        total = int(args.num_slices)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    dataset = args.dataset.strip() or _infer_dataset_name(args.data_root)

                                                                   
    pairs = iter_slices(
        args.data_root,
        total_slices=total,
        seed=args.seed,
        num_cases=args.num_cases,
        slices_per_case=args.slices_per_case,
    )
    if args.out_manifest.strip():
        outm = args.out_manifest
        os.makedirs(os.path.dirname(outm) or ".", exist_ok=True)
        man = pd.DataFrame(
            {
                "dataset": dataset,
                "file": [os.path.basename(fp) for fp, _ in pairs],
                "slice": [int(s) for _, s in pairs],
                "accel": int(args.accel),
                "center_fraction": float(args.center_fraction),
                "seed": int(args.seed),
                "mask_seed": int(args.mask_seed),
            }
        )
        man.to_csv(outm, index=False)
        print(f"[OK] Wrote manifest: {outm}  (n={len(man)})")

    cfg = PCRConfig(
        smooth_ksize=int(args.pcr_smooth_ksize),
        clip_percentile=float(args.pcr_clip_percentile),
        topk=float(args.pcr_topk),
    )

                                                             
    mask_cache: dict[Tuple[int, int], torch.Tensor] = {}

    rows = []
    with torch.no_grad():
        for fp, sidx in tqdm(pairs, total=len(pairs), desc="Slices"):
            data = load_multicoil_h5_slice(fp, sidx)

                                                                          
            k_full = data["kspace_full"].to(device)
            ref_rss = data.get("ref_rss", None)
            if ref_rss is not None:
                ref_rss = ref_rss.to(device)

            C, H, W = k_full.shape

                                               
            if (H, W) not in mask_cache:
                mask_cache[(H, W)] = make_vd_cartesian_mask(
                    (H, W), accel=args.accel, center_fraction=args.center_fraction, seed=args.mask_seed
                ).to(device)
            mask = mask_cache[(H, W)]
            mask_f = mask.to(dtype=torch.float32)

                              
            y = k_full * mask_f[None, ...]

                                                    
            ref_shape = tuple(ref_rss.shape) if ref_rss is not None else None
            coil_zf = build_zf_coil_images(y, ref_shape=ref_shape).to(device)           

            h, w = int(coil_zf.shape[-2]), int(coil_zf.shape[-1])
            if (h, w) != (H, W):
                                                                                        
                if (h, w) not in mask_cache:
                    mask_cache[(h, w)] = make_vd_cartesian_mask(
                        (h, w), accel=args.accel, center_fraction=args.center_fraction, seed=args.mask_seed
                    ).to(device)
                mask = mask_cache[(h, w)]
                mask_f = mask.to(dtype=torch.float32)
                y = _fft2c(coil_zf) * mask_f[None, ...]

                                          
            sens = estimate_sens_maps_from_acs(y, acs_fraction=args.center_fraction).to(device)
            xhat = sense_combine(coil_zf, sens)                 

                                                                               
            eps = 1e-8

                                       
            coil_pred = xhat[None, ...] * sens                                 
            y_pred = _fft2c(coil_pred) * mask_f[None, ...]                     
            r = (y - y_pred)                                                                   

            resid_norm = (torch.linalg.vector_norm(r) / (torch.linalg.vector_norm(y) + eps)).item()

                                              
            b_coil = _ifft2c(r)                                                
            b = torch.sum(torch.conj(sens) * b_coil, dim=0)                  
            bp_abs_mean = torch.mean(torch.abs(b)).item()

                                                             
            zf_rss = torch.sqrt(torch.sum(torch.abs(coil_zf) ** 2, dim=0) + eps)         
            bp_rel_mean = (torch.mean(torch.abs(b)) / (torch.mean(zf_rss) + eps)).item()

                                                                      
            pcr_map, scores, _ = compute_pcr_map(y=y, xhat=xhat, sens=sens, mask=mask, cfg=cfg)

                            
            pcr_mean = float(scores.get("pcr_mean", np.nan))
            pcr_topk_ratio = float(scores.get("pcr_topk_ratio", np.nan))
            pcr_core_periphery = float(scores.get("pcr_core_periphery", np.nan))
            pcr_peak_density = float(scores.get("pcr_peak_density", np.nan))

                                                                                           
                                                                                      
            pcr_fail_mean = -pcr_mean
            bp_fail = -bp_rel_mean
            pcr_fail_topk_ratio = pcr_topk_ratio
            pcr_fail_core_periphery = pcr_core_periphery

                                                                   
            xhat_mag = torch.abs(xhat).detach().cpu().numpy()
            has_ref = ref_rss is not None

            row = {
                                            
                "dataset": dataset,
                "file": os.path.basename(fp),
                "slice": int(sidx),
                "accel": int(args.accel),
                "center_fraction": float(args.center_fraction),
                "seed": int(args.seed),
                "mask_seed": int(args.mask_seed),
                "device": str(device),
                "pcr_smooth_ksize": int(args.pcr_smooth_ksize),
                "pcr_clip_percentile": float(args.pcr_clip_percentile),
                "pcr_topk": float(args.pcr_topk),
                "has_ref": bool(has_ref),

                                  
                "resid_norm": float(resid_norm),
                "bp_abs_mean": float(bp_abs_mean),
                "bp_rel_mean": float(bp_rel_mean),

                                      
                "pcr_mean": float(pcr_mean),
                "pcr_topk_ratio": float(pcr_topk_ratio),
                "pcr_core_periphery": float(pcr_core_periphery),
                "pcr_peak_density": float(pcr_peak_density),

                                  
                "pcr_fail_mean": float(pcr_fail_mean),
                "bp_fail": float(bp_fail),
                "pcr_fail_topk_ratio": float(pcr_fail_topk_ratio),
                "pcr_fail_core_periphery": float(pcr_fail_core_periphery),
            }

            if ref_rss is not None:
                ref = ref_rss.detach().cpu().numpy()
                             
                if ref.shape != xhat_mag.shape:
                    mh = min(ref.shape[0], xhat_mag.shape[0])
                    mw = min(ref.shape[1], xhat_mag.shape[1])
                    ref = ref[:mh, :mw]
                    xhat_mag = xhat_mag[:mh, :mw]

                row["nrmse"] = float(nrmse(xhat_mag, ref))
                row["psnr"] = float(psnr(xhat_mag, ref))
                row["ssim"] = float(ssim(xhat_mag, ref))
            else:
                row["nrmse"] = np.nan
                row["psnr"] = np.nan
                row["ssim"] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote: {args.out_csv}  (n={len(df)})")


if __name__ == "__main__":
    main()