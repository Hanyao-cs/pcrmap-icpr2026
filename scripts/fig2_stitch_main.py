                             
from __future__ import annotations

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


                              
           
                              
def _read_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def _get_font(size: int):
    try:
        import matplotlib.font_manager as fm

        font_path = fm.findfont("DejaVu Sans")
        return ImageFont.truetype(font_path, size=size)
    except Exception:
        return ImageFont.load_default()


def _auto_title_crop_y0(im: Image.Image, thr: int = 245, frac: float = 0.20) -> int:
    arr = np.asarray(im.convert("RGB"))
    gray = arr.mean(axis=2)
    nonwhite = gray < thr
    frac_row = nonwhite.mean(axis=1)

    for y in range(len(frac_row)):
        if frac_row[y] > frac:
            if y + 6 < len(frac_row) and float(frac_row[y : y + 6].mean()) > frac:
                return int(y)
    return 0


def _crop_outer_white_margins(im: Image.Image, thr: int = 245, frac: float = 0.01) -> Image.Image:
    arr = np.asarray(im.convert("RGB"))
    gray = arr.mean(axis=2)
    nonwhite = gray < thr

    frac_col = nonwhite.mean(axis=0)
    xs = np.where(frac_col > frac)[0]
    if len(xs) == 0:
        return im
    x0, x1 = int(xs[0]), int(xs[-1] + 1)

    frac_row = nonwhite.mean(axis=1)
    ys = np.where(frac_row > frac)[0]
    if len(ys) == 0:
        return im
    y0, y1 = int(ys[0]), int(ys[-1] + 1)

    return im.crop((x0, y0, x1, y1))


def _split_triplet(im: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image]:
    w, h = im.size
    w1 = w // 3
    w2 = w // 3
    w3 = w - w1 - w2
    a = im.crop((0, 0, w1, h))
    b = im.crop((w1, 0, w1 + w2, h))
    c = im.crop((w1 + w2, 0, w1 + w2 + w3, h))
    return a, b, c


def _crop_lr_whitespace(im: Image.Image, thr: int = 250, frac: float = 0.05, pad: int = 1) -> Image.Image:
    arr = np.asarray(im.convert("RGB"))
    gray = arr.mean(axis=2)
    nonwhite = gray < thr

    frac_col = nonwhite.mean(axis=0)
    xs = np.where(frac_col > frac)[0]
    if len(xs) == 0:
        return im

    x0, x1 = int(xs[0]), int(xs[-1] + 1)
    x0 = max(0, x0 - pad)
    x1 = min(im.size[0], x1 + pad)
    return im.crop((x0, 0, x1, im.size[1]))


def _hcat(imgs: List[Image.Image], gap: int, bg=(255, 255, 255)) -> Image.Image:
    h = max(im.size[1] for im in imgs)
    w = sum(im.size[0] for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.size[0] + gap
    return canvas


def preprocess_triplet_to_row(
    im: Image.Image,
    crop_top: int,
    panel_gap: int,
) -> Tuple[Image.Image, Tuple[int, int, int], int]:
                        
    if crop_top >= 0:
        y0 = crop_top
    else:
        y0 = _auto_title_crop_y0(im, thr=245, frac=0.20)
    im = im.crop((0, y0, im.size[0], im.size[1]))

                        
    im = _crop_outer_white_margins(im, thr=245, frac=0.01)

                                                
    a, b, c = _split_triplet(im)
    a = _crop_lr_whitespace(a, thr=250, frac=0.05, pad=1)
    b = _crop_lr_whitespace(b, thr=250, frac=0.05, pad=1)
    c = _crop_lr_whitespace(c, thr=250, frac=0.05, pad=1)

    row = _hcat([a, b, c], gap=panel_gap, bg=(255, 255, 255))
    return row, (a.size[0], b.size[0], c.size[0]), a.size[1]


def _path_from_gallery(gallery_dir: Path, run_prefix: str, idx: int) -> Path:
    key = f"{run_prefix}_{idx:04d}"
    subdir = gallery_dir / key

                        
    cand = subdir / "Fig2_overlay.png"
    if cand.exists():
        return cand

                                        
    for name in ["fig2_overlay.png", "Fig2_overlay.PNG", "fig2_overlay.PNG"]:
        p = subdir / name
        if p.exists():
            return p

                                                     
    if subdir.is_dir():
        pngs = list(subdir.glob("*.png")) + list(subdir.glob("*.PNG"))
        if pngs:
            pngs.sort(key=lambda p: p.stat().st_size, reverse=True)
            return pngs[0]

    raise FileNotFoundError(f"Cannot find overlay png for idx={idx} under {subdir}")


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


                              
      
                              
def main():
    ap = argparse.ArgumentParser()

                                
    ap.add_argument("--gallery_dir", default="", help=r"e.g. E:\pcr-map\outputs\fig2_gallery_cand200_mid")
    ap.add_argument("--run_prefix", default="", help="e.g. brain_fig2_cand200_mid")
    ap.add_argument("--indices", default="", help="comma-separated indices, e.g. 25,131,6,53,1,50")

                                                                              
    ap.add_argument("--paths", default="", help="comma-separated full paths to Fig2_overlay.png in desired order")

            
    ap.add_argument("--row_labels", default="", help='comma-separated labels, e.g. "TN,TN,TP,TP,TP(severe),FP"')

            
    ap.add_argument("--out", required=True, help=r"output path, e.g. E:\pcr-map\figures\Fig2_main.png")
    ap.add_argument("--dpi", type=int, default=300)

            
    ap.add_argument("--panel_gap", type=int, default=10, help="gap between 3 panels (pixels)")
    ap.add_argument("--row_gap", type=int, default=14, help="gap between rows (pixels)")
    ap.add_argument("--outer_pad", type=int, default=18, help="outer padding (pixels)")

                                
    ap.add_argument("--add_header", action="store_true", help="add a single header row for 3 columns")
    ap.add_argument("--header_h", type=int, default=70, help="header height (pixels)")
    ap.add_argument("--header_font", type=int, default=34, help="header font size")

                 
    ap.add_argument("--label_font", type=int, default=34, help="row label font size")
    ap.add_argument("--label_w", type=int, default=170, help="left label margin width (pixels)")

              
    ap.add_argument("--crop_top", type=int, default=-1, help="-1 auto; or set fixed pixels to crop from top")

                                                                      
                                                            
                                                        
    ap.add_argument("--group_break_after", default="", help="comma-separated row indices after which to add extra gap")
    ap.add_argument("--group_extra_gap", type=int, default=28, help="extra gap added at group breaks (pixels)")

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

                                                       
    inputs: List[Path] = []
    if args.paths.strip():
        inputs = [Path(x.strip()) for x in args.paths.split(",") if x.strip()]
        for p in inputs:
            if not p.exists():
                raise FileNotFoundError(f"Missing image: {p}")
    else:
        if not (args.gallery_dir and args.run_prefix and args.indices):
            raise ValueError("Either provide --paths OR provide --gallery_dir --run_prefix --indices")
        gallery_dir = Path(args.gallery_dir)
        indices = _parse_int_list(args.indices)
        for idx in indices:
            inputs.append(_path_from_gallery(gallery_dir, args.run_prefix, idx))

    row_labels = _parse_str_list(args.row_labels) if args.row_labels.strip() else []
    if row_labels and len(row_labels) != len(inputs):
        raise ValueError(f"row_labels count ({len(row_labels)}) != number of rows ({len(inputs)})")

    group_break_after = set(_parse_int_list(args.group_break_after)) if args.group_break_after.strip() else set()

                                              
    rows: List[Image.Image] = []
    panel_ws: Optional[Tuple[int, int, int]] = None
    panel_h: Optional[int] = None

    for p in inputs:
        print("[IN]", p)
        im = _read_rgb(p)
        row, ws, h = preprocess_triplet_to_row(im, crop_top=args.crop_top, panel_gap=args.panel_gap)
        rows.append(row)
        if panel_ws is None:
            panel_ws = ws
            panel_h = h
        else:
                                                                                      
            if h != panel_h:
                raise RuntimeError("Panel heights differ after preprocessing; check input images.")

    assert panel_ws is not None and panel_h is not None

                   
    nrows = len(rows)
    row_h = rows[0].size[1]
    label_w = args.label_w if row_labels else 0
    header_h = args.header_h if args.add_header else 0

                                                
    extra_gaps = sum(args.group_extra_gap for i in range(nrows) if i in group_break_after)
    W = 2 * args.outer_pad + label_w + rows[0].size[0]
    H = 2 * args.outer_pad + header_h + nrows * row_h + (nrows - 1) * args.row_gap + extra_gaps

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

                                   
    if args.add_header:
        font = _get_font(args.header_font)
        x0 = args.outer_pad + label_w
        w1, w2, w3 = panel_ws

        c1 = x0 + w1 // 2
        c2 = x0 + w1 + args.panel_gap + w2 // 2
        c3 = x0 + w1 + args.panel_gap + w2 + args.panel_gap + w3 // 2
        y_text = args.outer_pad + header_h // 2

                                                                                                    
        draw.text((c1, y_text), "|x_hat|", anchor="mm", fill=(0, 0, 0), font=font)
        draw.text((c2, y_text), "abs error", anchor="mm", fill=(0, 0, 0), font=font)
        draw.text((c3, y_text), "PCR-Map", anchor="mm", fill=(0, 0, 0), font=font)

                              
    font_lab = _get_font(args.label_font)
    y = args.outer_pad + header_h
    for i, row in enumerate(rows):
        x = args.outer_pad + label_w
        canvas.paste(row, (x, y))

        if row_labels:
            draw.text(
                (args.outer_pad + label_w - 10, y + row_h // 2),
                row_labels[i],
                anchor="rm",
                fill=(0, 0, 0),
                font=font_lab,
            )

        y += row_h + args.row_gap
        if i in group_break_after:
            y += args.group_extra_gap

                    
    out_png = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    canvas.save(out_png)
    print("[OK] Saved PNG:", out_png)

    out_pdf = out_png.with_suffix(".pdf")
    canvas.save(out_pdf, "PDF", resolution=args.dpi)
    print("[OK] Saved PDF:", out_pdf)


if __name__ == "__main__":
    main()