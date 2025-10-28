import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from abc import ABC
from typing import List
import numpy as np
import torch
from PIL import Image
import sys, os
# In test.py, change:
from .bg_decomposer import ObjectClearDecomposer  # Update this line

# And update the initialization:
decomposer = ObjectClearDecomposer(device="cuda" if torch.cuda.is_available() else "cpu")

# Load the input image and mask
image_path = "../animating_image/src/concept_decomposer/bg_decomposer/input/image.jpg"
mask_path = "../animating_image/src/concept_decomposer/bg_decomposer/input/mask.jpg"

print("Absolute image path:", os.path.abspath(image_path))

image = np.array(Image.open(image_path).convert('RGB'))
mask = np.array(Image.open(mask_path).convert('L'))  # Ensure the mask is grayscale (single channel)

# If you have multiple masks, ensure it's a list of masks
masks = [mask]  # Add more masks if needed

# Initialize IBgDecomposer
# decomposer = IBgDecomposer(device="cuda" if torch.cuda.is_available() else "cpu")

# Decompose the image by inpainting the object mask
result_image = decomposer.decompose(image, masks)
image = Image.fromarray(result_image)
image.save("output.png")

# Visualize the result
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Result after inpainting
axes[1].imshow(result_image)
axes[1].set_title("Decomposed Image")
axes[1].axis('off')

plt.tight_layout()
plt.show()