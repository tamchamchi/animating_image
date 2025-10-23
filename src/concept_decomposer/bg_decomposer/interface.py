from abc import ABC, abstractmethod
from typing import List
import numpy as np

class IBgDecomposer(ABC):
    @abstractmethod
    def __init__(self, device=None):
        pass

    @abstractmethod
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
<<<<<<< HEAD
        pass
=======
        # Combine all object masks into a single binary mask
        combined_mask = np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8) * 255

        # Convert to PIL images
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(combined_mask)

        # Resize both for model input
        image_resized = resize_by_short_side(image_pil, 512, Image.BICUBIC)
        mask_resized = resize_by_short_side(mask_pil, 512, Image.NEAREST)

        # Run inpainting
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
>>>>>>> f17307da0b485d7c7ee91da9e765ad7edba5b76d
