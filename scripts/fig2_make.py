#!/usr/bin/env python

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

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pcrmap.data_io_fastmri import load_multicoil_h5_slice, build_zf_coil_images
from pcrmap.mri_ops import make_vd_cartesian_mask, estimate_sens_maps_from_acs, sense_combine
from pcrmap.pcrmap import compute_pcr_map, PCRConfig


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


def _ensure_on(x, device: torch.device):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)


def _resolve_h5_path(data_root: Path, fname: str) -> Path:
    p = Path(fname)
    if p.exists():
        return p
    p2 = data_root / fname
    if p2.exists():
        return p2
    if p.suffix.lower() != ".h5":
        p3 = data_root / (fname + ".h5")
        if p3.exists():
            return p3
    raise FileNotFoundError(f"Cannot resolve H5 path for '{fname}' under '{data_root}'")


def _compute_one(
    h5_path: Path,
    sl: int,
    accel: int,
    cf: float,
    mask_seed: int,
    cfg: PCRConfig,
    device: torch.device,
):
    data = load_multicoil_h5_slice(str(h5_path), sl)
    k_full = _ensure_on(data["kspace_full"], device)                           
    ref_rss = _ensure_on(data.get("ref_rss", None), device)                        

    C, H, W = k_full.shape
    mask = make_vd_cartesian_mask((H, W), accel=accel, center_fraction=cf, seed=mask_seed).to(device)
    mask_f = mask.to(dtype=torch.float32)

                      
    y = k_full * mask_f[None, ...]

                                         
    ref_shape = tuple(ref_rss.shape) if ref_rss is not None else None
    coil_zf = build_zf_coil_images(y, ref_shape=ref_shape).to(device)           
    h, w = int(coil_zf.shape[-2]), int(coil_zf.shape[-1])

                                                                                        
    if (h, w) != (H, W):
        mask = make_vd_cartesian_mask((h, w), accel=accel, center_fraction=cf, seed=mask_seed).to(device)
        mask_f = mask.to(dtype=torch.float32)
        y = _fft2c(coil_zf) * mask_f[None, ...]

                  
    sens = estimate_sens_maps_from_acs(y, acs_fraction=cf).to(device)
    xhat = sense_combine(coil_zf, sens)                 

             
    pcr_map, scores, extras = compute_pcr_map(y=y, xhat=xhat, sens=sens, mask=mask, cfg=cfg)

    recon = torch.abs(xhat).detach().cpu().numpy()
    if ref_rss is not None:
        ref = ref_rss.detach().cpu().numpy()
        if ref.shape != recon.shape:
            mh = min(ref.shape[0], recon.shape[0])
            mw = min(ref.shape[1], recon.shape[1])
            ref = ref[:mh, :mw]
            recon = recon[:mh, :mw]
        err = np.abs(recon - ref)
    else:
        err = np.zeros_like(recon)

    pcr_raw = pcr_map.detach().cpu().numpy()
    return recon, err, pcr_raw, scores


def _quantile_finite(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _anatomy_mask(recon: np.ndarray, q: float = 0.80) -> np.ndarray:
    a = np.asarray(recon, dtype=np.float64)
    a = np.abs(a)
    thr = _quantile_finite(a, q)
    if not np.isfinite(thr):
        return np.ones_like(a, dtype=bool)
    return a >= thr


def _normalize01(x: np.ndarray, vmax: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(x)) if np.isfinite(np.nanmax(x)) else 1.0
    return np.clip(x / vmax, 0.0, 1.0)


def _overlay_top_tail(
    base_gray: np.ndarray,
    pcr_norm01: np.ndarray,
    anat: np.ndarray,
    topq: float,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    base = np.asarray(base_gray, dtype=np.float64)
    base = _normalize01(base, vmax=_quantile_finite(base, 0.99) or 1.0)

    p = np.asarray(pcr_norm01, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)

    mask = anat.astype(bool)
    thr = _quantile_finite(p[mask], topq) if mask.any() else _quantile_finite(p, topq)
    if not np.isfinite(thr):
        thr = 1.0

    hot = (p >= thr) & mask
    strength = np.zeros_like(p)
    strength[hot] = np.clip(((p[hot] - thr) / (1e-8 + (1.0 - thr))) ** gamma, 0.0, 1.0)

    rgb = np.stack([base, base, base], axis=-1)
    rgb[..., 0] = np.clip(rgb[..., 0] + alpha * strength, 0.0, 1.0)               
    return rgb


def _save_grid(
    cases,
    out_path_png: Path,
    out_path_pdf: Path,
    style: str,
    vmax_recon: float,
    vmax_err: float,
    vmax_pcr: float,
    dpi: int,
    overlay_alpha: float,
    overlay_topq: float,
    overlay_gamma: float,
    anat_q: float,
    show_colorbar: bool,
    err_mode: str,
):
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.2 * n), squeeze=False)

    for i, c in enumerate(cases):
        recon = c["recon"]
        err = c["err"]
        pcr = c["pcr"]

        recon_n = _normalize01(recon, vmax_recon)
        err_n = _normalize01(err, vmax_err)
        pcr_n = _normalize01(pcr, vmax_pcr)

        label = f'{c["group"]} | {c["stem"]} | sl={c["slice"]}'
        if np.isfinite(c.get("nrmse", np.nan)):
            label += f' | NRMSE={c["nrmse"]:.4f}'
        if np.isfinite(c.get("score", np.nan)):
            label += f' | score={c["score"]:.4f}'
        axes[i, 0].set_ylabel(label, fontsize=9)

        axes[i, 0].imshow(recon_n, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("|x̂|", fontsize=10)
        axes[i, 0].axis("off")

        if err_mode == "abs":
            err_show = err_n
            title_err = "| |x̂| - |x| |"
        else:
            err_show = err_n
            title_err = "abs error"
        axes[i, 1].imshow(err_show, cmap="gray" if style == "gray" else "inferno", vmin=0, vmax=1)
        axes[i, 1].set_title(title_err, fontsize=10)
        axes[i, 1].axis("off")

        if style == "overlay":
            anat = _anatomy_mask(recon, q=anat_q)
            rgb = _overlay_top_tail(
                base_gray=recon,
                pcr_norm01=pcr_n,
                anat=anat,
                topq=overlay_topq,
                alpha=overlay_alpha,
                gamma=overlay_gamma,
            )
            axes[i, 2].imshow(rgb, vmin=0, vmax=1)
        else:
            axes[i, 2].imshow(pcr_n, cmap="gray" if style == "gray" else "magma", vmin=0, vmax=1)
        axes[i, 2].set_title("PCR-Map", fontsize=10)
        axes[i, 2].axis("off")

    if show_colorbar:
        mappable = plt.cm.ScalarMappable(cmap="gray")
        mappable.set_array([0, 1])
        cb = fig.colorbar(mappable, ax=axes[:, 0], fraction=0.03, pad=0.02)
        cb.set_label("Recon (normalized)", fontsize=10)

        mappable2 = plt.cm.ScalarMappable(cmap="inferno")
        mappable2.set_array([0, 1])
        cb2 = fig.colorbar(mappable2, ax=axes[:, 1], fraction=0.03, pad=0.02)
        cb2.set_label("Error (normalized)", fontsize=10)

        mappable3 = plt.cm.ScalarMappable(cmap="magma")
        mappable3.set_array([0, 1])
        cb3 = fig.colorbar(mappable3, ax=axes[:, 2], fraction=0.03, pad=0.02)
        cb3.set_label("PCR (normalized)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path_png}")
    print(f"[OK] Saved: {out_path_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs/fig2")
    ap.add_argument("--run_id", type=str, default="", help="Optional run id; images are saved under out_dir/run_id/.")
    ap.add_argument("--accel", type=int, default=8, help="Fallback accel if manifest missing accel column.")
    ap.add_argument("--center_fraction", type=float, default=0.08, help="Fallback center fraction if manifest missing center_fraction column.")

                                                                      
    ap.add_argument("--pcr_smooth_ksize", type=int, default=7)
    ap.add_argument("--pcr_clip_percentile", type=float, default=99.5)
    ap.add_argument("--pcr_topk", type=float, default=0.02)
    ap.add_argument("--mask_seed", type=int, default=0, help="Seed for mask generation (should match batch).")
    ap.add_argument("--dpi", type=int, default=300)

    ap.add_argument("--device", type=str, default="", help="cuda:0 / cpu. Default: cuda if available else cpu.")
    ap.add_argument("--score_col", type=str, default="pcr_fail_mean", help="Column to display as score in row label.")

    ap.add_argument("--img_q", type=float, default=0.99, help="Global quantile for recon/error vmax.")
    ap.add_argument("--pcr_q", type=float, default=0.99, help="Global quantile for PCR vmax before 0-1 normalization.")

    ap.add_argument("--style", type=str, default="all", choices=["all", "gray", "heat", "overlay"])
    ap.add_argument("--overlay_alpha", type=float, default=0.55, help="Base overlay strength for PCR.")
    ap.add_argument("--overlay_topq", type=float, default=0.95, help="Overlay PCR above this quantile within anatomy (0.90~0.98 recommended).")
    ap.add_argument("--overlay_gamma", type=float, default=1.0, help="Gamma for hotspot strength.")
    ap.add_argument("--anat_q", type=float, default=0.80, help="Anatomy mask quantile from recon (0.70~0.90).")

    ap.add_argument("--no_colorbar", action="store_true")
    ap.add_argument("--err_mode", type=str, default="abs", choices=["abs", "signed"])

    args = ap.parse_args()

    if args.device.strip():
        device = torch.device(args.device.strip())
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    out_dir = Path(args.out_dir)
    if args.run_id.strip():
        out_dir = out_dir / args.run_id.strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    ex = pd.read_csv(manifest_path)

    cfg = PCRConfig(
        smooth_ksize=int(args.pcr_smooth_ksize),
        clip_percentile=float(args.pcr_clip_percentile),
        topk=float(args.pcr_topk),
    )

    data_root = Path(args.data_root)

    cases = []
    recon_vals = []
    err_vals = []
    pcr_vals = []

    with torch.no_grad():
        for _, row in ex.iterrows():
            fname = str(row["file"])
            sl = int(row["slice"])
            accel = int(row.get("accel", args.accel))
            cf = float(row.get("center_fraction", args.center_fraction))
            h5_path = _resolve_h5_path(data_root, fname)

            nrmse_val = float(row["nrmse"]) if ("nrmse" in ex.columns and pd.notna(row.get("nrmse", np.nan))) else float("nan")
            group_val = str(row["group"]) if ("group" in ex.columns and pd.notna(row.get("group", np.nan))) else ""
            score_val = float(row[args.score_col]) if (args.score_col in ex.columns and pd.notna(row.get(args.score_col, np.nan))) else float("nan")

            recon, err, pcr_raw, _scores = _compute_one(h5_path, sl, accel, cf, args.mask_seed, cfg, device=device)

            stem = Path(str(fname)).stem
            cases.append({
                "stem": stem,
                "file": fname,
                "slice": sl,
                "group": group_val,
                "nrmse": nrmse_val,
                "score": score_val,
                "recon": recon,
                "err": err,
                "pcr": pcr_raw,
            })

            recon_vals.append(_quantile_finite(recon, args.img_q))
            err_vals.append(_quantile_finite(err, args.img_q))
            pcr_vals.append(_quantile_finite(pcr_raw, args.pcr_q))

    vmax_recon = float(np.nanmax(recon_vals)) if len(recon_vals) else 1.0
    vmax_err = float(np.nanmax(err_vals)) if len(err_vals) else 1.0
    vmax_pcr = float(np.nanmax(pcr_vals)) if len(pcr_vals) else 1.0

    styles = ["gray", "heat", "overlay"] if args.style == "all" else [args.style]
    for st in styles:
        out_png = out_dir / f"Fig2_{st}.png"
        out_pdf = out_dir / f"Fig2_{st}.pdf"
        _save_grid(
            cases=cases,
            out_path_png=out_png,
            out_path_pdf=out_pdf,
            style=st,
            vmax_recon=vmax_recon,
            vmax_err=vmax_err,
            vmax_pcr=vmax_pcr,
            dpi=args.dpi,
            overlay_alpha=args.overlay_alpha,
            overlay_topq=args.overlay_topq,
            overlay_gamma=args.overlay_gamma,
            anat_q=args.anat_q,
            show_colorbar=(not args.no_colorbar),
            err_mode=args.err_mode,
        )


if __name__ == "__main__":
    main()