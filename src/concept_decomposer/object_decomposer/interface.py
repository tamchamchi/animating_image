from abc import ABC, abstractmethod
import numpy as np
from PIL import Image # <-- CẦN THIẾT: Import thư viện PIL

class IObjectDecomposer(ABC):
    @abstractmethod
    def decompose(self, object_list: list[str], input_image: Image.Image) -> dict:
        """
        Decompose the input image into separate objects, excluding the background.

        Args:
            self: Tham chiếu đến thể hiện của lớp (ĐÃ THÊM).
            object_list (list[str]): A list of object descriptions or prompts.
            input_image (PIL.Image.Image): The input image (ĐÃ THAY THẾ np.ndarray).

        Returns:
            dict: A dictionary containing information for each object, including:
                  - image: the cropped image of the object (numpy.ndarray)
                  - mask: the segmentation mask of the object (numpy.ndarray)
                  - bounding_box: the bounding box coordinates of the object (tuple)
        """
        pass