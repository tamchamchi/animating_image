# utils.py
import numpy as np
import cv2
from typing import Tuple, Optional

def compute_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tính Bounding Box (x_min, y_min, x_max, y_max) từ mask nhị phân (HxW)."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    # Trả về định dạng (x_min, y_min, x_max, y_max)
    return (x_min, y_min, x_max, y_max)

def create_mask_visualization(mask: np.ndarray) -> np.ndarray:
    """Tạo ảnh 3 kênh màu trắng (255) cho mask và đen (0) cho nền."""
    mask_3ch = np.stack([mask] * 3, axis=-1)
    mask_3ch = (mask_3ch > 0).astype(np.uint8) * 255
    return mask_3ch