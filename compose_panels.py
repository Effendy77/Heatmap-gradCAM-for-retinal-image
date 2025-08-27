#!/usr/bin/env python
from __future__ import annotations
import argparse, os, pathlib
from PIL import Image, ImageDraw

def tile(img_paths, cols=4, pad=8):
    imgs = [Image.open(p).convert('RGB') for p in img_paths]
    if not imgs: return None
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new('RGB', (cols*w + (cols+1)*pad, rows*h + (rows+1)*pad), (255,255,255))
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i>=len(imgs): break
            x = pad + c*(w+pad); y = pad + r*(h+pad)
            canvas.paste(imgs[i], (x,y)); i += 1
    return canvas

def label_bar(text, width, height=40):
    bar = Image.new('RGB', (width, height), (255,255,255))
    d = ImageDraw.Draw(bar)
    d.text((10,10), text, fill=(0,0,0))
    return bar

def main(root_dir: str, out_dir: str, cols: int = 4):
    root = pathlib.Path(root_dir)
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for cohort_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for stratum_dir in sorted([p for p in cohort_dir.iterdir() if p.is_dir()]):
            rows = []; labels = []
            for oc in ['TP','FP','FN','TN']:
                d = stratum_dir/oc
                if not d.exists(): continue
                imgs = sorted([str(x) for x in d.glob('*_cam.png')])
                grid = tile(imgs, cols=cols)
                if grid is not None:
                    rows.append(grid); labels.append(oc)
            if not rows: continue
            width = max(r.size[0] for r in rows)
            height = sum(r.size[1] for r in rows) + 40*len(rows)
            panel = Image.new('RGB', (width, height), (255,255,255))
            y = 0
            for lbl, row in zip(labels, rows):
                panel.paste(label_bar(f"{cohort_dir.name} — {stratum_dir.name} — {lbl}", width), (0,y)); y += 40
                panel.paste(row, (0,y)); y += row.size[1]
            out_path = out / f"{cohort_dir.name}__{stratum_dir.name}__panel.png"
            panel.save(out_path); print('wrote', out_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cols', type=int, default=4)
    a = ap.parse_args()
    main(a.root, a.out, a.cols)
    print(f'Panels generated in {a.out}')
    print('You can now run `compose_panels.py` to create a final overview panel.')  