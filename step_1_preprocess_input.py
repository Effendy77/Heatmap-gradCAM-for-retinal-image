import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
path = r'/home/fendy77/RETFound_MAE27_Kfold/Project/images/1NoMACE_image.png'
img_rgb = Image.open(path)

# Convert to numpy array
img_array = np.array(img_rgb)

# Convert to grayscale
gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

# Threshold the image to create a binary mask, using Otsu's thresholding method
_, binary_mask = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find the bounding box of the non-black area
x, y, w, h = cv2.boundingRect(binary_mask)

# Crop the image to the bounding box region (without the black border)
cropped_img = img_array[y:y+h, x:x+w]

# Create a circular mask
mask = np.zeros((cropped_img.shape[0], cropped_img.shape[1]), dtype=np.uint8)

# Get the center and radius of the circle
center = (cropped_img.shape[1] // 2, cropped_img.shape[0] // 2)
radius = min(cropped_img.shape[0], cropped_img.shape[1]) // 2

# Draw the circle on the mask
cv2.circle(mask, center, radius, (255), thickness=-1)

# Apply the circular mask to the image
masked_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

# Replace the black background (outside the circle) with white
masked_img_with_white_bg = np.copy(masked_img)
masked_img_with_white_bg[mask == 0] = [255, 255, 255]  # Set background to white

# Resize the image to 224x224 for model input
masked_img_resized = cv2.resize(masked_img_with_white_bg, (224, 224))

# Convert resized image to PIL format
masked_img_resized_pil = Image.fromarray(masked_img_resized)

# Save the preprocessed image
output_path = '/home/fendy77/RETFound_MAE27_Kfold/Project/images/1NoMACE_image_afterstep1.png'
masked_img_resized_pil.save(output_path)

# Display the preprocessed image
plt.imshow(masked_img_resized_pil)
plt.axis('off')
plt.show()

print(f"Preprocessed image saved to {output_path}")

