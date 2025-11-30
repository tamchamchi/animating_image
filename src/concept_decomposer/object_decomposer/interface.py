# interface.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import numpy as np

class IObjectDecomposer(ABC):
    """
    Giao diện trừu tượng cho các lớp phân tách đối tượng.
    """
    @abstractmethod
    def decompose(self, image: np.ndarray) -> Optional[Dict]:
        """Tự động phân tách ảnh để tìm đối tượng chính (Output Mask + Texture)."""
        pass

    @abstractmethod
    def detect_objects(self, image: np.ndarray, prompts: List[str]) -> List[Dict]:
        """
        Phát hiện đối tượng dựa trên danh sách prompt.
        
        Args:
            image: Ảnh đầu vào (BGR).
            prompts: Danh sách tên các vật thể cần tìm (VD: ["ổ điện", "bảng nội quy"]).
            
        Returns:
            List[Dict]: Danh sách kết quả, mỗi phần tử có dạng:
            { "name": "tên object", "bbox": [x_min, y_min, x_max, y_max], "score": float }
        """
        pass