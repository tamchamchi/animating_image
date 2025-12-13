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
