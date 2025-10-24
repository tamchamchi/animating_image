# interface.py
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

class IObjectDecomposer(ABC):
    """
    Giao diện trừu tượng cho các lớp phân tách đối tượng.
    """
    @abstractmethod
    def decompose(self, prompt_i: List[str], image: np.ndarray) -> Dict[str, Dict]:
        """
        Phân tách ảnh thành các đối tượng/mask dựa trên danh sách các prompt.
        
        Args:
            prompt_i: Danh sách các chuỗi mô tả đối tượng cần tìm (ví dụ: ["dog", "cat", "sky"]).
            image: Ảnh đầu vào dưới dạng numpy array (BGR).
            
        Returns:
            Dict: Một dictionary, với key là prompt, value là dictionary chứa 
                  "is_background", "image" (ảnh đã cắt), "mask", "bounding_box".
        """
        pass