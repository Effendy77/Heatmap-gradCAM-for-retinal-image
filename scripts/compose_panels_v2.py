#!/usr/bin/env python
from __future__ import annotations
import argparse, pathlib, re
from PIL import Image, ImageDraw, ImageFont
import yaml

def _load_thr(config_path: str | None) -> str:
    if not config_path:
        return ""
    try:
        cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        thr = cfg.get("threshold", None)
        return f"{thr}" if thr is not None else ""
    except Exception:
        return ""

def label_bar(text: str, width: int, height: int = 40) -> Image.Image:
    img = Image.new("RGB", (width, height), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.text((10, 10), text, fill=(0, 0, 0))
    d.line((0, height-1, width, height-1), fill=(200, 200, 200), width=1)
    return img

def tile(img_paths: list[pathlib.Path], cols: int) -> Image.Image | None:
    if not img_paths:
        return None
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        grid.paste(im, (c * w, r * h))
    return grid

def main(root: str, out: str, cols: int, outcomes: list[str], cohorts: list[str] | None,
         stratum_regex: str | None, config: str | None):

    rootp = pathlib.Path(root)
    outp = pathlib.Path(out)
    outp.mkdir(parents=True, exist_ok=True)

    thr = _load_thr(config)
    thr_suffix = f", thr={thr}" if thr else ""

    cohort_dirs = [p for p in rootp.iterdir() if p.is_dir()]
    cohort_dirs = sorted(cohort_dirs)

    if cohorts:
        keep = set([c.lower() for c in cohorts])
        cohort_dirs = [p for p in cohort_dirs if p.name.lower() in keep]

    stratum_re = re.compile(stratum_regex) if stratum_regex else None

    for cohort_dir in cohort_dirs:
        strata = [p for p in cohort_dir.iterdir() if p.is_dir()]
        strata = sorted(strata)

        for stratum_dir in strata:
            if stratum_re and not stratum_re.search(stratum_dir.name):
                continue

            rows = []
            labels = []
            max_width = None

            for oc in outcomes:
                d = stratum_dir / oc
                if not d.exists():
                    continue
                imgs = sorted(d.glob("*_cam.png"))
                grid = tile(list(imgs), cols=cols)
                if grid is None:
                    continue

                if max_width is None:
                    max_width = grid.size[0]
                labels.append(f"{cohort_dir.name} | {stratum_dir.name} — {oc} (n={len(imgs)}{thr_suffix})")
                rows.append(grid)

            if not rows:
                continue

            width = max_width
            height = sum(r.size[1] for r in rows) + 40 * len(rows)
            panel = Image.new("RGB", (width, height), (255, 255, 255))

            y = 0
            for lbl, row in zip(labels, rows):
                bar = label_bar(lbl, width)
                panel.paste(bar, (0, y)); y += bar.size[1]
                panel.paste(row, (0, y)); y += row.size[1]

            out_path = outp / f"{cohort_dir.name}__{stratum_dir.name}__{'-'.join(outcomes)}__panel.png"
            panel.save(out_path)
            print("wrote", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--outcomes", nargs="+", default=["TP","FP","FN","TN"])
    ap.add_argument("--cohorts", nargs="*", default=None, help="e.g., secondary primary")
    ap.add_argument("--stratum_regex", default=None, help="regex filter for stratum folder names")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    main(a.root, a.out, a.cols, a.outcomes, a.cohorts, a.stratum_regex, a.config)
