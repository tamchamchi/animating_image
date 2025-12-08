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
    # Đảm bảo mask đầu vào là 1 kênh
    if mask.ndim == 3:
        mask = mask[..., 0]
        
    mask_3ch = np.stack([mask] * 3, axis=-1)
    mask_3ch = (mask_3ch > 0).astype(np.uint8) * 255
    return mask_3ch

def draw_results(image, detections):
    vis_img = image.copy()
    for det in detections:
        bbox = det['bbox']
        name = det['name']
        score = det['score']
        
        x_min, y_min, x_max, y_max = bbox
        
        # Vẽ box
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        label = f"{name} ({score:.2f})"
        cv2.putText(vis_img, label, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return vis_img