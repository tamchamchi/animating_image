import os
import uuid

import aiofiles
from fastapi import UploadFile


async def save_upload_file(file: UploadFile, directory: str) -> str:
    file_path = os.path.join(directory, file.filename)
    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    return file_path


def generate_id():
    return str(uuid.uuid4())


def get_file_url(folder_type: str, item_id: str, filename: str):
    return f"/static/{folder_type}/{item_id}/{filename}"
