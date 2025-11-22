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
        pass
