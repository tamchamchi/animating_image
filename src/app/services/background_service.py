import os
import cv2
import json
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

from fastapi.concurrency import run_in_threadpool
from fastapi import HTTPException, UploadFile

from src.app.core.config import settings
from src.app.core.container import ai_container
from src.app.utils.image_ops import read_image_as_numpy
from src.utils.svg_file import (
    extract_topk_polygons,
    get_svg_size,
    polygons_to_json_object,
    restore_polygon_to_image_coords,
)

load_dotenv()

class BackgroundService:
    def __init__(self):
        self.targets_file_path = Path(os.getenv("TARGET_OBJECT"))

    def _get_path(self, anim_id):
        return os.path.join(settings.STATIC_DIR, "animations", anim_id)

    def _load_targets_from_file(self) -> dict:
        if not self.targets_file_path.exists():
            print(f"Warning: Targets file not found at {self.targets_file_path}")
            return {}
        try:
            with open(self.targets_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading targets file: {e}")
            return {}

    async def get_polygon_by_model(
        self,
        anim_id: str,
        file: UploadFile,
        confidence_threshold: float
    ):
        work_dir = self._get_path(anim_id)
        if not os.path.exists(work_dir):
            raise HTTPException(status_code=404, detail=f"Animation ID '{anim_id}' not found")

        targets = self._load_targets_from_file()
        if not targets:
            raise HTTPException(status_code=500, detail="No detection targets found in configuration.")

        bg_img = await read_image_as_numpy(file)
        
        # Tạo thư mục nếu chưa tồn tại (đề phòng)
        os.makedirs(work_dir, exist_ok=True)
        cv2.imwrite(os.path.join(work_dir, "background.jpg"), bg_img)

        final_output = []

        async with ai_container.semaphore:
            for output_name, model_prompt in targets.items():
                detections = await run_in_threadpool(
                    ai_container.decomposer.detect_objects,
                    image=bg_img,
                    prompts=[model_prompt],
                    threshold=confidence_threshold,
                )

                for det in detections:
                    item = {
                        "name": output_name,
                        "bbox": det['bbox'],
                        "score": det['score'],
                        "polygon": det["polygon"]
                    }
                    final_output.append(item)

        json_path = os.path.join(work_dir, "detected_objects.json")
        with open(json_path, 'w') as f:
            json.dump(final_output, f, indent=2)

        return final_output

    async def get_polygon_by_svg(
        self,
        anim_id: str,
        file: UploadFile,
        top_k: int
    ):
        work_dir = Path(self._get_path(anim_id))
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)

        bg_img = await read_image_as_numpy(file)
        
        original_img_path = work_dir / "background.jpg"
        cv2.imwrite(str(original_img_path), bg_img)

        pil_bg = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))

        async with ai_container.semaphore:
            svg_content = await run_in_threadpool(
                ai_container.svg_converter.convert,
                images=[pil_bg],
                limit=10000
            )

            svg_path = work_dir / "background.svg"
            svg_path.write_text(svg_content)

            def process_svg_polygons():
                polys = extract_topk_polygons(svg_path, top_k=top_k)
                svg_w, svg_h = get_svg_size(svg_path)
                bg_h, bg_w = bg_img.shape[:2]

                restored = [
                    restore_polygon_to_image_coords(
                        poly, svg_w, svg_h, bg_w, bg_h)
                    for poly in polys
                ]
                return polygons_to_json_object(restored[1:])

            final_output = await run_in_threadpool(process_svg_polygons)

        json_path = work_dir / "detected_objects.json"
        with open(json_path, 'w') as f:
            json.dump(final_output, f, indent=2)

        return final_output