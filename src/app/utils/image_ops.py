import cv2
import numpy as np
from fastapi import UploadFile


async def read_image_as_numpy(file: UploadFile) -> np.ndarray:
    content = await file.read()
    # Convert bytes to numpy (opencv format BGR)
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


async def save_upload_file(file: UploadFile, save_path: str):
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    # Reset cursor
    await file.seek(0)

def resize_by_scale(img, scale: float, interpolation=None):
    """
    Resize ảnh theo hệ số scale, giữ nguyên tỉ lệ.

    Args:
        img (np.ndarray): Ảnh đầu vào (BGR - OpenCV)
        scale (float): Hệ số scale (vd: 0.8, 1.2)
        interpolation (int, optional): cv2 interpolation method

    Returns:
        np.ndarray: Ảnh đã resize
    """
    if img is None:
        raise ValueError("Input image is None")

    if scale <= 0:
        raise ValueError("Scale must be > 0")

    # Chọn interpolation mặc định hợp lý
    if interpolation is None:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

    resized = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=interpolation
    )
    return resized
