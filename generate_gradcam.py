#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml, os, pathlib, pandas as pd, numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
import cv2

from hybrid_gradcam.model_loader import load_model, resolve_target_layer
from hybrid_gradcam.gradcam_utils import overlay_cam_on_image

def preprocess_if_enabled(pil: Image.Image, enable: bool, size: int) -> Image.Image:
    if not enable:
        return pil.resize((size,size))
    # circular mask + white bg
    img = np.array(pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x, y, w, h = cv2.boundingRect(mask)
    crop = img[y:y+h, x:x+w]
    circ = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    c = (crop.shape[1]//2, crop.shape[0]//2); r = min(crop.shape[0], crop.shape[1])//2
    cv2.circle(circ, c, r, 255, -1)
    masked = cv2.bitwise_and(crop, crop, mask=circ)
    masked[circ==0] = [255,255,255]
    return Image.fromarray(cv2.resize(masked, (size,size)))

def main(cfg_path: str, selections_dir: str, out_root: str):
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device(cfg.get('device','cpu'))
    size = int(cfg.get('image_size',224))
    pre_flag = bool(cfg.get('preprocess',{}).get('enable', False))
    add_bar = bool(cfg.get('visuals',{}).get('add_colorbar', True))

    tfm = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.get('normalize_mean',[0.485,0.456,0.406]),
                             std=cfg.get('normalize_std',[0.229,0.224,0.225]))
    ])

    os.makedirs(out_root, exist_ok=True)
    img_root_cfg = cfg['images_dir']  # dict or str
    target = cfg.get('target_layer','blocks.-1.norm1')
    backbone = cfg.get('backbone','vit')

    for cohort in ['primary','secondary']:
        ckpt = cfg['checkpoints'][cohort]
        model = load_model(backbone, ckpt, num_classes=1).to(device)
        target_layer = resolve_target_layer(model, target)
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type=='cuda'))

        # pick cohort-specific image root
        img_root = img_root_cfg[cohort] if isinstance(img_root_cfg, dict) else img_root_cfg

        sel_csv = os.path.join(selections_dir, f'{cohort}_selections.csv')
        if not os.path.exists(sel_csv):
            print(f'[WARN] missing selections: {sel_csv}')
            continue
        df = pd.read_csv(sel_csv)

        for _, r in df.iterrows():
            img_path = os.path.join(img_root, str(r['filename']))
            if not os.path.exists(img_path):
                continue
            pil = Image.open(img_path).convert('RGB')
            pil = preprocess_if_enabled(pil, pre_flag, size)
            arr = np.asarray(pil, dtype=np.float32)/255.0
            tensor = tfm(pil).unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=tensor, targets=[BinaryClassifierOutputTarget(1)])[0]
            heat = overlay_cam_on_image(arr, grayscale_cam, add_colorbar=add_bar)

            out_dir = os.path.join(out_root, cohort, str(r['stratum']), r['outcome'])
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            heat.save(os.path.join(out_dir, f'{stem}_cam.png'))

    print(f'Grad-CAM images saved -> {out_root}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--selections', required=True, help='dir containing *_selections.csv')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.config, a.selections, a.out)
