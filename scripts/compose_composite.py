#!/usr/bin/env python
import argparse, pathlib
from PIL import Image, ImageDraw

def label_bar(text, width, height=40):
    img = Image.new("RGB", (width, height), (245,245,245))
    d = ImageDraw.Draw(img)
    d.text((10,10), text, fill=(0,0,0))
    d.line((0,height-1,width,height-1), fill=(200,200,200), width=1)
    return img

def main(in_dir, out_png, title, cols=2, pad=20):
    in_dir = pathlib.Path(in_dir)
    paths = sorted(in_dir.glob("*__panel.png"))
    if not paths:
        raise SystemExit(f"No panels found in {in_dir}")

    imgs = [Image.open(p).convert("RGB") for p in paths]
    w = max(im.size[0] for im in imgs)
    h = max(im.size[1] for im in imgs)

    rows = (len(imgs) + cols - 1) // cols
    canvas_w = cols*w + (cols+1)*pad
    canvas_h = rows*h + (rows+1)*pad + 40  # title bar

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255,255,255))
    y = 0
    canvas.paste(label_bar(title, canvas_w, 40), (0,0))
    y = 40

    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        x0 = pad + c*(w+pad)
        y0 = y + pad + r*(h+pad)
        canvas.paste(im, (x0, y0))

    out_png = pathlib.Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    print("wrote", out_png)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="folder containing *__panel.png")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--cols", type=int, default=2)
    a = ap.parse_args()
    main(a.in_dir, a.out_png, a.title, cols=a.cols)
