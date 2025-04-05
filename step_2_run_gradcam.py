import argparse
import cv2
import numpy as np
import torch
import os
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from functools import partial
import torch.nn as nn
import timm
import models_vit  # Assuming this module exists for loading models

# Define the VisionTransformer class and vit_large_patch16 function
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

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
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# Load the model architecture
def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    # Build model
    model = vit_large_patch16(global_pool=True, num_classes=2)  # Set to 2 classes for binary classification
    # Load model checkpoint
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    return model

# Hardcoded arguments
args = argparse.Namespace(
    device='cpu',
    image_path='/home/fendy77/RETFound_MAE27_Kfold/Project/images/1NoMACE_image_afterstep1.png',
    output_dir='./cam_output',
    aug_smooth=False,
    eigen_smooth=False,
    method='gradcam',  # You can change this to any method like 'gradcam++', 'scorecam', etc.
    model_path='/home/fendy77/RETFound_MAE27_Kfold/Project/checkpoint/checkpoint-best.pth'  # Replace with the actual model path
)
def parse_args():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM for an image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output image')
    parser.add_argument('--method', type=str, required=True, help='Method name for generating CAM')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply augmentation smoothing')
    parser.add_argument('--eigen_smooth', action='store_true', help='Apply eigenvalue smoothing')

    return parser.parse_args()
# Initialize the model
model = prepare_model(args.model_path, 'vit_large_patch16')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)  # Bring channels to first dimension
    return result

# Main code
if __name__ == '__main__':
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }

    if args.method not in methods:
        raise ValueError(f"Method should be one of {list(methods.keys())}")

    # Set the target layer for Grad-CAM
    target_layers = [model.blocks[-1].norm1]  # Adjust this according to your model architecture

    # Initialize the CAM method
    cam = methods[args.method](model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # Read and preprocess the input image
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]  # Convert BGR to RGB
    rgb_img = cv2.resize(rgb_img, (224, 224))  # Resize image to model input size
    rgb_img = np.float32(rgb_img) / 255  # Normalize to [0, 1]
    
    # Preprocess the image according to your model's requirement
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(device)

    # Use None for the highest scoring category
    targets = None

    # Optional: adjust batch size for computation if necessary
    cam.batch_size = 32

    # Compute the Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets,
                        eigen_smooth=args.eigen_smooth, aug_smooth=args.aug_smooth)

    grayscale_cam = grayscale_cam[0, :]  # Get the first image's CAM

    # Overlay the heatmap on the original image
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    # Ensure the output directory exists
    try:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    except OSError as e:
        print(f"Error creating directory {args.output_dir}: {e}")
        raise

    # Save the resulting image with CAM applied
    output_image_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
    try:
        cv2.imwrite(output_image_path, cam_image)
        print(f"CAM image saved at: {output_image_path}")
    except cv2.error as e:
        print(f"Error saving image {output_image_path}: {e}")
        raise
