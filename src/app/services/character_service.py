import logging
import os
import time
import uuid

import cv2
from fastapi.concurrency import run_in_threadpool

from src.app.core.config import settings
from src.app.core.container import ai_container
from src.app.utils.image_ops import read_image_as_numpy
from src.utils.prompt import PROMPT_SUBJECT_GENERATION

logger = logging.getLogger(__name__)


class CharacterService:
    def __init__(self):
        self.base_url = "/static/characters"

    def _get_workspace(self):
        char_id = str(uuid.uuid4())
        path = os.path.join(settings.STATIC_DIR, "characters", char_id)
        os.makedirs(path, exist_ok=True)
        return char_id, path

    async def create_from_face(self, file, body_file):
        start_t = time.time()
        char_id, work_dir = self._get_workspace()
        logger.info(f"===> [CHAR FACE] Creating character from face. ID: {char_id}")
        # 1. Read input image
        input_img = await read_image_as_numpy(file)  # BGR

        # 2. Load body cartoon template (Cần chuẩn bị sẵn file này)
        body_img = await read_image_as_numpy(body_file)

        # 3. AI Processing (Threadpool + Semaphore)
        logger.info(f"     [CHAR FACE] {char_id} waiting for SEMAPHORE...")
        async with ai_container.semaphore:
            # SegFormer segment -> trả về mask, region, bbox...
            # Gọi hàm run_in_threadpool vì AI là code chặn (blocking)
            logger.info(
                f"     [CHAR FACE] {char_id} ACQUIRED SEMAPHORE. Segmenting and Merging..."
            )
            ai_start = time.time()

            face_mask, face_region, bbox, face_crop = await run_in_threadpool(
                ai_container.face_segmenter.segment, input_img
            )
            logger.info(
                f"     [CHAR FACE] AI Logic finished in {time.time() - ai_start:.2f}s"
            )

            # Apply to body
            # Lưu ý: Convert BGR -> RGB vì model apply_to_body dùng PIL logic bên trong
            final_img_rgb = await run_in_threadpool(
                ai_container.face_segmenter.apply_to_body,
                cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
            )

        # 4. Save results
        # Convert lại BGR để lưu bằng OpenCV
        final_img_bgr = cv2.cvtColor(final_img_rgb, cv2.COLOR_RGB2BGR)
        output_name = "character_result.png"
        cv2.imwrite(os.path.join(work_dir, output_name), final_img_bgr)

        face_filename = "face_region.png"
        face_url = None

        if face_region is not None and face_region.size > 0:
            face_rgba = cv2.cvtColor(face_region, cv2.COLOR_BGR2BGRA)
            face_rgba[:, :, 3] = face_mask
            cv2.imwrite(os.path.join(work_dir, face_filename), face_rgba)
            face_url = f"{self.base_url}/{char_id}/{face_filename}"
        logger.info(f"<=== [CHAR FACE] Completed in {time.time() - start_t:.2f}s")

        return {
            "id": char_id,
            "image_url": f"{self.base_url}/{char_id}/{output_name}",
            "face_url": face_url,
        }

    async def create_from_prompt(self, prompt: str):
        start_t = time.time()

        char_id, work_dir = self._get_workspace()
        logger.info(f"===> [CHAR PROMPT] Generating character. ID: {char_id}")

        prompt = PROMPT_SUBJECT_GENERATION.format(subject=prompt)

        async with ai_container.semaphore:
            # NanoBananaGenerator trả về PIL Image
            logger.info(
                f"     [CHAR PROMPT] {char_id} ACQUIRED SEMAPHORE. Calling Generator..."
            )
            ai_start = time.time()
            pil_image = await run_in_threadpool(ai_container.generator.generate, prompt)
            logger.info(
                f"     [CHAR PROMPT] AI finished in {time.time() - ai_start:.2f}s"
            )

        output_name = "generated_char.png"
        pil_image.save(os.path.join(work_dir, output_name))
        logger.info(f"<=== [CHAR PROMPT] Completed in {time.time() - start_t:.2f}s")

        return {"id": char_id, "image_url": f"{self.base_url}/{char_id}/{output_name}"}
