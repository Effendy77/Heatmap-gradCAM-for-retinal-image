#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml, os, pathlib, pandas as pd, numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

# --- at the top imports ---
import re

def _safe_dir(s: str) -> str:
    """Make a Windows-safe folder name from a stratum label."""
    s = str(s)
    # map disallowed characters to readable alternatives
    repl = {'<':'lt', '>':'gt', ':':'-', '"':"'", '/':'-', '\\':'-', '|':'-', '?':'', '*':''}
    return re.sub(r'[<>:"/\\|?*]', lambda m: repl[m.group(0)], s)


# --- make the internal package importable when running from scripts/ ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "src")))

# --- grad-cam imports (current API) ---
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform

# method registry so you can pick in YAML
_CAMS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
}

# local helpers
from hybrid_gradcam.model_loader import load_model, resolve_target_layer
from hybrid_gradcam.gradcam_utils import overlay_cam_on_image


def preprocess_if_enabled(pil: Image.Image, enable: bool, size: int) -> Image.Image:
    """Optionally circular-crop with white background; else just resize."""
    if not enable:
        return pil.resize((size, size))
    img = np.array(pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x, y, w, h = cv2.boundingRect(mask)
    crop = img[y:y + h, x:x + w]
    circ = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    c = (crop.shape[1] // 2, crop.shape[0] // 2)
    r = min(crop.shape[0], crop.shape[1]) // 2
    cv2.circle(circ, c, r, 255, -1)
    masked = cv2.bitwise_and(crop, crop, mask=circ)
    masked[circ == 0] = [255, 255, 255]
    return Image.fromarray(cv2.resize(masked, (size, size)))


def _pick_targets(cam_cfg: dict, num_logits: int) -> list | None:
    """
    Decide CAM targets from YAML:
      cam.target:
        - "auto"  -> None (argmax)
        - "pos"   -> index 0 if num_logits==1 else 1
        - "<int>" -> that index
    """
    t = str(cam_cfg.get("target", "auto")).lower()
    if t == "auto":
        return None
    if t == "pos":
        idx = 0 if num_logits == 1 else 1
        return [ClassifierOutputTarget(idx)]
    # integer index fallback
    try:
        idx = int(t)
    except Exception:
        idx = 0
    return [ClassifierOutputTarget(idx)]


def main(cfg_path: str, selections_dir: str, out_root: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    device = torch.device(cfg.get("device", "cpu"))
    size = int(cfg.get("image_size", 224))
    pre_flag = bool(cfg.get("preprocess", {}).get("enable", False))
    add_bar = bool(cfg.get("visuals", {}).get("add_colorbar", True))

    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=cfg.get("normalize_std", [0.229, 0.224, 0.225]),
        )
    ])

    os.makedirs(out_root, exist_ok=True)
    img_root_cfg = cfg["images_dir"]  # dict or str
    target = cfg.get("target_layer", "blocks.-1.norm1")
    backbone = str(cfg.get("backbone", "vit")).lower()

    # CAM config (optional in YAML)
    cam_cfg = cfg.get("cam", {})  # method, target, batch_size
    method = str(cam_cfg.get("method", "gradcam")).lower()
    CamCls = _CAMS.get(method, GradCAM)

    # reshape transform for ViT-like models (tokens -> spatial)
    reshape = vit_reshape_transform if backbone.startswith("vit") or "retfound" in backbone else None

    for cohort in ["primary", "secondary"]:
        ckpt = cfg["checkpoints"][cohort]
        # NOTE: model_loader is set to num_classes=1 for your binary head by default.
        model = load_model(backbone, ckpt, num_classes=1).to(device)
        model.eval()

        target_layer = resolve_target_layer(model, target)
        cam = CamCls(model=model, target_layers=[target_layer], reshape_transform=reshape)
        # optional batch size (some CAMs expose this)
        if hasattr(cam, "batch_size"):
            try:
                cam.batch_size = int(cam_cfg.get("batch_size", 16))
            except Exception:
                pass

        # pick cohort-specific image root
        img_root = img_root_cfg[cohort] if isinstance(img_root_cfg, dict) else img_root_cfg

        sel_csv = os.path.join(selections_dir, f"{cohort}_selections.csv")
        if not os.path.exists(sel_csv):
            print(f"[WARN] missing selections: {sel_csv}")
            continue
        df = pd.read_csv(sel_csv)

        # decide class targets based on head size (1 logit -> idx 0; else from YAML/index)
        num_logits = 1  # because we instantiated with num_classes=1
        targets = _pick_targets(cam_cfg, num_logits)

        for _, r in df.iterrows():
            img_path = os.path.join(img_root, str(r["filename"]))
            if not os.path.exists(img_path):
                continue
            pil = Image.open(img_path).convert("RGB")
            pil = preprocess_if_enabled(pil, pre_flag, size)
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            tensor = tfm(pil).unsqueeze(0).to(device)

            # CAM compute
            grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

            # overlay & save
            heat = overlay_cam_on_image(arr, grayscale_cam)

            # ✅ sanitize stratum for Windows-safe folder names
            stratum_dir = _safe_dir(r["stratum"])          # e.g., "<50" -> "lt50"
            out_dir = os.path.join(out_root, cohort, stratum_dir, str(r["outcome"]))
            os.makedirs(out_dir, exist_ok=True)

            stem = os.path.splitext(os.path.basename(img_path))[0]
            heat.save(os.path.join(out_dir, f"{stem}_cam.png"))


    print(f"Grad-CAM images saved -> {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--selections", required=True, help="dir containing *_selections.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.config, a.selections, a.out)
