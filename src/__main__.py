from .concept_decomposer.object_decomposer import GroundedSAMDecomposer
import torch
import numpy as np
from PIL import Image

sam2_config = "./configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_checkpoint = "./external/checkpoints/sam2.1_hiera_large.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

object_decomposer = GroundedSAMDecomposer(
    sam2_checkpoint=sam2_checkpoint,
    sam2_config=sam2_config,
    device=device
)

prompt = [
    "man", 
    "head", 
    "hair", 
    "face", 
    "eyes", 
    "mouth", 
    "shirt", 
    "pants", 
    "shoes", 
    "arms", 
    "hands", 
    "legs"
]

image = Image.open("/home/anhndt/animating_image/images/test/handdrawing_child_with_hat.png").convert("RGB")
image_np = np.array(image)

output = "/home/anhndt/animating_image/images/output_object_decompose"

res = object_decomposer.decompose(prompt_i=prompt, image=image_np, visualize=True, output_dir=output)
print(res)


