import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import timm
from functools import partial
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -------------------- CONFIG --------------------
CHECKPOINT_PATH = r"D:/PREVIOUS_TRAINING/RETFound_MAE41ver4-CLAHE_T1diabetes/output_fold_3/checkpoint-best.pth"
IMG_DIR = r"D:/DATA/main_data/bilateralclean"
OUTPUT_DIR = r"D:/EXPERIMENT/gradCAM/T1D"

TARGETS = {
    "TP": "1715151_21015_0_0.png",
    "TN": "3861308_21015_0_0.png",
    "FP": "1774990_21015_0_0.png",
    "FN": "4821733_21015_0_0.png"
}

# -------------------- MODEL WRAPPER --------------------
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
        return x

def vit_large_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

# -------------------- GRAD-CAM RUN --------------------
model = vit_large_patch16(global_pool=True, num_classes=2)
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

target_layers = [model.blocks[-1].norm1]
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
targets = [ClassifierOutputTarget(1)]  # MACE class

# -------------------- LOOP THROUGH TARGETS --------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
for label, fname in TARGETS.items():
    img_path = os.path.join(IMG_DIR, fname)
    output_path = os.path.join(OUTPUT_DIR, f"GradCAM_{label}_T1D.png")

    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        continue

    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    rgb_img = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(device)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved: {output_path}")
