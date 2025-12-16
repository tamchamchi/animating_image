import json
import os

import aiofiles
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageSequence

from src.app.core.config import settings


class GameService:
    def __init__(self):
        self.base_url = "/static/animations"

    def _get_path(self, anim_id):
        return os.path.join(settings.STATIC_DIR, "animations", anim_id)

    def _trim_gif_bottom(self, file_path: str) -> str:
        """
        Cắt khoảng trắng thừa ở dưới chân nhân vật trong GIF
        nhưng VẪN GIỮ NGUYÊN BACKGROUND TRONG SUỐT.
        """
        file_dir = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)

        trimmed_filename = f"{name}_trimmed{ext}"
        trimmed_path = os.path.join(file_dir, trimmed_filename)

        if os.path.exists(trimmed_path):
            return trimmed_filename

        try:
            with Image.open(file_path) as im:
                # 1. Lấy thông tin Transparency gốc (QUAN TRỌNG)
                # GIF dùng Palette mode ('P'), transparency là index của màu trong suốt trong bảng màu.
                transparency_index = im.info.get("transparency")

                # 2. Tính toán điểm thấp nhất (Max Bottom) dựa trên RGBA
                # Chuyển sang RGBA tạm thời để hàm getbbox() nhận diện chính xác độ trong suốt (alpha channel)
                max_bottom = 0
                frames = []

                for frame in ImageSequence.Iterator(im):
                    # Copy frame gốc (Mode P) để dành cho việc cắt sau này
                    original_frame = frame.copy()
                    frames.append(original_frame)

                    # Convert sang RGBA chỉ để tính toán bbox (không dùng để save vì sẽ làm tăng dung lượng GIF)
                    rgba_frame = frame.convert("RGBA")
                    bbox = rgba_frame.getbbox()

                    if bbox:
                        # bbox = (left, top, right, bottom)
                        if bbox[3] > max_bottom:
                            max_bottom = bbox[3]

                # Nếu ảnh rỗng hoặc không tìm thấy điểm cắt, trả về file gốc
                if max_bottom == 0:
                    return filename

                # 3. Thực hiện Crop trên các frame gốc (Mode P)
                width = im.size[0]
                cropped_frames = []

                for frame in frames:
                    # Crop trực tiếp trên Mode P để giữ nguyên bảng màu
                    cropped = frame.crop((0, 0, width, max_bottom))
                    cropped_frames.append(cropped)

                # 4. Lưu file mới
                if cropped_frames:
                    # Các tham số save() bắt buộc để giữ animation và transparency mượt mà
                    save_kwargs = {
                        "save_all": True,
                        "append_images": cropped_frames[1:],
                        "loop": 0,  # Lặp vô tận
                        "duration": im.info.get("duration", 100),
                        # 2 = Restore to background color (Xóa frame cũ đi -> Tránh bị chồng hình)
                        "disposal": 2,
                        "optimize": False,  # Tắt optimize đôi khi giúp giữ bảng màu ổn định hơn
                    }

                    # Nếu file gốc có transparency, truyền lại đúng index đó
                    if transparency_index is not None:
                        save_kwargs["transparency"] = transparency_index

                    cropped_frames[0].save(trimmed_path, **save_kwargs)

                    return trimmed_filename

        except Exception as e:
            print(f"Error trimming GIF {filename}: {e}")
            return filename

        return filename

    async def get_resources(self, game_id: str):
        """
        Lấy tài nguyên game.
        """
        work_dir = self._get_path(game_id)

        if not os.path.exists(work_dir):
            raise HTTPException(status_code=404, detail="Game ID not found")

        # 1. Xử lý GIF
        files = os.listdir(work_dir)
        # Chỉ lấy file gốc, không lấy file đã trim
        raw_gifs = [
            f for f in files if f.lower().endswith(".gif") and "_trimmed" not in f
        ]

        final_gif_urls = []

        # Chạy tác vụ xử lý ảnh trong threadpool
        for gif_file in raw_gifs:
            full_path = os.path.join(work_dir, gif_file)

            processed_filename = await run_in_threadpool(
                self._trim_gif_bottom, full_path
            )

            final_gif_urls.append(f"{self.base_url}/{game_id}/{processed_filename}")

        final_gif_urls.sort()

        # 2. Background
        bg_file = None
        for ext in ["png", "jpg", "jpeg"]:
            possible_bg = f"background.{ext}"
            if os.path.exists(os.path.join(work_dir, possible_bg)):
                bg_file = possible_bg
                break

        bg_url = f"{self.base_url}/{game_id}/{bg_file}" if bg_file else None

        # 3. JSON Objects
        json_filename = "detected_objects.json"
        json_path = os.path.join(work_dir, json_filename)
        if not os.path.exists(json_path):
            json_path = os.path.join(work_dir, "detected_object.json")

        detected_objects = []
        if os.path.exists(json_path):
            async with aiofiles.open(json_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
                try:
                    detected_objects = json.loads(content)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {json_path}")

        return {
            "game_id": game_id,
            "action_gif_urls": final_gif_urls,
            "background_url": bg_url,
            "detected_objects": detected_objects,
        }
