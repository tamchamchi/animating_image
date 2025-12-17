import json
import os

import aiofiles
import numpy as np
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
        Cắt khoảng trắng thừa dưới chân GIF,
        dịch nhân vật xuống sát đáy,
        thêm bóng đen lệch trái 5px,
        vẫn giữ background trong suốt & animation.
        """

        file_dir = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)

        trimmed_filename = f"{name}_trimmed{ext}"
        trimmed_path = os.path.join(file_dir, trimmed_filename)

        # nếu đã xử lý rồi thì bỏ qua
        if os.path.exists(trimmed_path):
            return trimmed_filename

        try:
            with Image.open(file_path) as im:
                transparency_index = im.info.get("transparency")

                max_bottom = 0
                frames = []

                # ========= 1. detect bottom chung ==========
                for frame in ImageSequence.Iterator(im):
                    original = frame.copy()
                    frames.append(original)

                    rgba = frame.convert("RGBA")
                    bbox = rgba.getbbox()

                    if bbox:
                        max_bottom = max(max_bottom, bbox[3])

                if max_bottom == 0:
                    return filename

                width = im.size[0]
                cropped_frames = []

                # ========= 2. xử lý từng frame ==========
                for frame in frames:
                    # crop theo bottom chung
                    cropped = frame.crop((0, 0, width, max_bottom))
                    rgba = cropped.convert("RGBA")

                    # detect alpha để dịch xuống đáy
                    arr = np.array(rgba)
                    alpha = arr[:, :, 3]
                    ys, xs = np.where(alpha > 10)

                    if len(ys) > 0:
                        h = arr.shape[0]
                        bottom = ys.max()
                        shift_y = (h - 1) - bottom

                        if shift_y > 0:
                            shifted = Image.new("RGBA", (width, max_bottom), (0, 0, 0, 0))
                            shifted.paste(rgba, (0, shift_y))
                            rgba = shifted

                    # ====== build shadow ======
                    arr2 = np.array(rgba)
                    alpha2 = arr2[:, :, 3]

                    shadow = np.zeros_like(arr2)
                    shadow[:, :, 3] = (alpha2 * 0.8).astype(np.uint8)
                    shadow_img = Image.fromarray(shadow, mode="RGBA")

                    # canvas rộng hơn để tránh crop trái
                    canvas_w = rgba.size[0] + 5
                    canvas_h = rgba.size[1]

                    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

                    # paste shadow lệch trái
                    canvas.paste(shadow_img, (0, 0))

                    # paste nhân vật chính
                    canvas.paste(rgba, (10, 0), rgba)

                    rgba = canvas

                    # convert về P để giữ transparency đúng
                    final_frame = rgba.convert("P", palette=Image.ADAPTIVE)
                    cropped_frames.append(final_frame)

                # ========= 3. Save GIF ==========
                if cropped_frames:
                    save_kwargs = {
                        "save_all": True,
                        "append_images": cropped_frames[1:],
                        "loop": 0,
                        "duration": im.info.get("duration", 100),
                        "disposal": 2,
                        "optimize": False,
                    }

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
