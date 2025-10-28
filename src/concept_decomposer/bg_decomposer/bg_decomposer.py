import numpy as np
import torch
from PIL import Image
from typing import List

from .objectclear.pipelines import ObjectClearPipeline
from .objectclear.utils import resize_by_short_side
from .interface import IBgDecomposer

class ObjectClearDecomposer(IBgDecomposer):
    def __init__(self, device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            "jixin0101/ObjectClear",
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            apply_attention_guided_fusion=True,
            variant="fp16" if "cuda" in self.device else None
        ).to(self.device)

    def decompose(self, image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """
        Inpaint the given object masks in the image.

        Parameters
        ----------
        image : np.ndarray
            Input image (H, W, C).
        masks : List[np.ndarray]
            List of masks (H, W), each mask corresponds to an object to inpaint.

        Returns
        -------
        np.ndarray
            Image after inpainting.
        """
        combined_mask = np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8) * 255

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(combined_mask)

        image_resized = resize_by_short_side(image_pil, 512, Image.BICUBIC)
        mask_resized = resize_by_short_side(mask_pil, 512, Image.NEAREST)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=("cuda" in self.device)):
            result = self.pipe(
                prompt="",
                image=image_resized,
                mask_image=mask_resized,
                generator=torch.Generator(self.device).manual_seed(42),
                num_inference_steps=15,
                strength=0.99,
                guidance_scale=2.5,
                height=image_resized.size[1],
                width=image_resized.size[0],
            )

        # Resize the result back to the original size
        cleaned_image = result.images[0].resize(image_pil.size)

        return np.array(cleaned_image)