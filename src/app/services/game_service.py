import os
import json
import aiofiles
from fastapi import HTTPException
from src.app.core.config import settings

class GameService:
    def __init__(self):
        self.base_url = "/static/animations"

    def _get_path(self, anim_id):
        return os.path.join(settings.STATIC_DIR, "animations", anim_id)

    async def get_resources(self, game_id: str):
        """
        Lấy tài nguyên của game: GIF, Background, JSON Objects
        """
        work_dir = self._get_path(game_id)
        
        if not os.path.exists(work_dir):
            raise HTTPException(status_code=404, detail="Game ID not found")

        # 1. Tìm action.gif
        # Tìm file gif bất kỳ trong folder, ưu tiên file tên "action.gif"
        gif_file = "action.gif"
        if not os.path.exists(os.path.join(work_dir, gif_file)):
             gifs = [f for f in os.listdir(work_dir) if f.endswith(".gif")]
             if gifs:
                 gif_file = gifs[0]
             else:
                 gif_file = None # Hoặc raise lỗi nếu bắt buộc phải có

        # 2. Tìm background.png
        bg_file = "background.png"
        if not os.path.exists(os.path.join(work_dir, bg_file)):
            bg_file = None

        # 3. Đọc detected_object.json
        json_file = "detected_object.json"
        json_path = os.path.join(work_dir, json_file)
        detected_objects = []
        
        if os.path.exists(json_path):
            async with aiofiles.open(json_path, mode='r') as f:
                content = await f.read()
                try:
                    detected_objects = json.loads(content)
                except json.JSONDecodeError:
                    print("Error decoding JSON")
        
        # Construct response
        return {
            "game_id": game_id,
            "action_gif_url": f"{self.base_url}/{game_id}/{gif_file}" if gif_file else None,
            "background_url": f"{self.base_url}/{game_id}/{bg_file}" if bg_file else None,
            "detected_objects": detected_objects
        }