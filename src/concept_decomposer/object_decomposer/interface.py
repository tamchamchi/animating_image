from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np

class IObjectDecomposer(ABC):
    """
    Giao diện trừu tượng cho các lớp phân tách đối tượng.
    """
    @abstractmethod
    def decompose(self, image: np.ndarray) -> Optional[Dict]:
        """
        Tự động phân tách ảnh để tìm đối tượng chính.
        
        Args:
            image: Ảnh đầu vào dưới dạng numpy array (BGR).
            
        Returns:
            Optional[Dict]: Một dictionary chứa thông tin của đối tượng chính
            (bao gồm "image" (ảnh RGBA), "mask", "mask_image_viz"),
            hoặc None nếu không tìm thấy.
        """
        pass