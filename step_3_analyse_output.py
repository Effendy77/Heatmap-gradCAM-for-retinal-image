import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Paths for the input image and Grad-CAM output
original_image_path = '/home/fendy77/RETFound_MAE27_Kfold/Project/images/1NoMACE_image_afterstep1.png'
cam_image_path = './cam_output/gradcam_cam.jpg'  # Update based on Step 2 output file name
output_dir = './analysis_output'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the original and CAM images
original_image = cv2.imread(original_image_path, 1)[:, :, ::-1]  # Convert BGR to RGB
cam_image = cv2.imread(cam_image_path, 1)[:, :, ::-1]  # Convert BGR to RGB

# Resize the images for consistency
original_image_resized = cv2.resize(original_image, (224, 224))  # Assuming input size is 224x224
cam_image_resized = cv2.resize(cam_image, (224, 224))

# Plot and save overlay comparison
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(original_image_resized)
plt.title('Original Image')
plt.axis('off')

# Grad-CAM Image
plt.subplot(1, 3, 2)
plt.imshow(cam_image_resized)
plt.title('Grad-CAM Image')
plt.axis('off')

# Overlay Grad-CAM Heatmap on Original
overlay = cv2.addWeighted(original_image_resized, 0.6, cam_image_resized, 0.4, 0)
plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title('Overlay: Original + Grad-CAM')
plt.axis('off')

# Save the overlay comparison plot
output_comparison_path = os.path.join(output_dir, 'gradcam_comparison.jpg')
plt.savefig(output_comparison_path, bbox_inches='tight')
print(f"Comparison image saved at: {output_comparison_path}")

# Save individual overlay image
overlay_image_path = os.path.join(output_dir, 'overlay_gradcam.jpg')
cv2.imwrite(overlay_image_path, overlay[:, :, ::-1])  # Convert back to BGR for saving
print(f"Overlay image saved at: {overlay_image_path}")
