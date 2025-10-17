from abc import ABC, abstractmethod
import numpy as np

class IObjectDecomposer(ABC):
    @abstractmethod
    def decompose(prompt_i: list[str], image: np.ndarray, ) -> dict:
        """
        Decompose the input image into separate objects, excluding the background.

        Args:
            prompt_i (list[str]): A list of object descriptions or prompts.
            image (np.ndarray): The input image containing multiple objects.

        Returns:
            dict: A dictionary containing information for each object, including:
                  - image: the cropped image of the object
                  - mask: the segmentation mask of the object
                  - bounding_box: the bounding box coordinates of the object
        """
        pass
