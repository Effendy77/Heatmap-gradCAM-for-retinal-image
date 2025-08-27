
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

def _make_colorbar(height: int, width: int = 24) -> Image.Image:
    grad = np.linspace(1.0, 0.0, height, dtype=np.float32).reshape(height, 1)
    grad_u8 = (grad * 255).astype(np.uint8)
    color = cv2.applyColorMap(grad_u8, cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    rgb = np.repeat(rgb, width, axis=1)
    return Image.fromarray(rgb)

def overlay_cam_on_image(rgb: np.ndarray, cam: np.ndarray, add_colorbar: bool = True) -> Image.Image:
    overlay = show_cam_on_image(rgb, cam, use_rgb=True)
    img = Image.fromarray(overlay)
    if not add_colorbar:
        return img
    bar = _make_colorbar(img.height, 24)
    pad = 16
    bar_canvas = Image.new('RGB', (bar.width + pad*2, img.height), (255, 255, 255))
    bar_canvas.paste(bar, (pad, 0))
    d = ImageDraw.Draw(bar_canvas)
    d.text((2, 2), "1", fill=(0, 0, 0))
    d.text((2, img.height - 14), "0", fill=(0, 0, 0))
    out = Image.new('RGB', (img.width + bar_canvas.width, img.height), (255, 255, 255))
    out.paste(img, (0, 0))
    out.paste(bar_canvas, (img.width, 0))
    return out
