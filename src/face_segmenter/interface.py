from abc import ABC, abstractmethod

import numpy as np


class IFaceSegmenter(ABC):

    @abstractmethod
    def segment(self, image: np.ndarray):
        """
        Input:
            image: numpy array (H, W, 3) RGB
        Output:
            face_mask: np.ndarray (H, W) 0/255
            face_region: np.ndarray (H, W, 3)
        """
        pass
