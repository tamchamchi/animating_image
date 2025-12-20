import os
import uuid
import cv2
import time
import logging
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from src.app.core.config import settings
from src.app.core.container import ai_container
from src.app.utils.image_ops import save_upload_file

# Khởi tạo logger
logger = logging.getLogger(__name__)


class AnimationService:
    def __init__(self):
        self.base_url = "/static/animations"

    def _get_path(self, anim_id):
        return os.path.join(settings.STATIC_DIR, "animations", anim_id)

    async def init_session(self, file):
        logger.info("===> [INIT SESSION] Starting new session")
        anim_id = str(uuid.uuid4())
        work_dir = self._get_path(anim_id)
        os.makedirs(work_dir, exist_ok=True)

        # Lưu file gốc
        file_path = os.path.join(work_dir, "original.png")
        await save_upload_file(file, file_path)

        logger.info(f"<=== [INIT SESSION] Done. ID: {anim_id}")
        return {"id": anim_id, "status": "initialized"}

    async def step1_decompose(self, anim_id: str):
        start_t = time.time()
        logger.info(f"===> [STEP 1] Decompose ID: {anim_id}")

        work_dir = self._get_path(anim_id)
        input_path = os.path.join(work_dir, "original.png")
        if not os.path.exists(input_path):
            logger.error(f"     [STEP 1] Session {anim_id} not found")
            raise HTTPException(404, "Session not found or image missing")

        img = cv2.imread(input_path)

        logger.info(f"     [STEP 1] {anim_id} is waiting for SEMAPHORE...")
        async with ai_container.semaphore:
            logger.info(
                f"     [STEP 1] {anim_id} ACQUIRED SEMAPHORE. Running AI...")
            ai_start = time.time()
            result = await run_in_threadpool(ai_container.decomposer.decompose, img)
            logger.info(
                f"     [STEP 1] {anim_id} AI Model finished in {time.time() - ai_start:.2f}s")

        if not result:
            logger.error(f"     [STEP 1] {anim_id} Decomposition failed")
            raise HTTPException(400, "Decomposition failed (No mask found)")

        # Lưu Mask và Texture
        cv2.imwrite(os.path.join(work_dir, "mask.png"),
                    cv2.cvtColor(result["mask"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(work_dir, "texture.png"),
                    cv2.cvtColor(result["texture"], cv2.COLOR_RGB2BGR))

        logger.info(
            f"<=== [STEP 1] Completed for {anim_id} in {time.time() - start_t:.2f}s")
        return {
            "mask_url": f"{self.base_url}/{anim_id}/mask.png",
            "texture_url": f"{self.base_url}/{anim_id}/texture.png",
        }

    async def step2_pose(self, anim_id: str):
        start_t = time.time()
        logger.info(f"===> [STEP 2] Pose ID: {anim_id}")

        work_dir = self._get_path(anim_id)
        texture_path = os.path.join(work_dir, "texture.png")
        if not os.path.exists(texture_path):
            texture_path = os.path.join(work_dir, "original.png")

        output_yaml = os.path.join(work_dir, "char_cfg.yaml")
        viz_output = os.path.join(work_dir, "pose_viz.png")

        logger.info(f"     [STEP 2] {anim_id} is waiting for SEMAPHORE...")
        async with ai_container.semaphore:
            logger.info(
                f"     [STEP 2] {anim_id} ACQUIRED SEMAPHORE. Running MMPose...")
            ai_start = time.time()
            await run_in_threadpool(
                ai_container.pose_estimator.predict,
                image=texture_path,
                output_file=viz_output,
                output_yaml=output_yaml,
            )
            logger.info(
                f"     [STEP 2] {anim_id} AI Model finished in {time.time() - ai_start:.2f}s")

        logger.info(
            f"<=== [STEP 2] Completed for {anim_id} in {time.time() - start_t:.2f}s")
        return {
            "joint_yaml_url": f"{self.base_url}/{anim_id}/char_cfg.yaml",
            "pose_viz_url": f"{self.base_url}/{anim_id}/pose_viz.png",
        }

    async def step3_animate(self, anim_id: str, action: str = "walk"):
        start_t = time.time()
        logger.info(f"===> [STEP 3] Animate ID: {anim_id} | Action: {action}")

        work_dir = self._get_path(anim_id)
        required_files = ["char_cfg.yaml", "mask.png", "texture.png"]
        for f in required_files:
            if not os.path.exists(os.path.join(work_dir, f)):
                logger.error(f"     [STEP 3] {anim_id} missing file: {f}")
                raise HTTPException(400, f"Missing prerequisite file: {f}")

        logger.info(f"     [STEP 3] {anim_id} is waiting for SEMAPHORE...")
        async with ai_container.semaphore:
            logger.info(
                f"     [STEP 3] {anim_id} ACQUIRED SEMAPHORE. Rendering GIF (Very Heavy)...")
            ai_start = time.time()
            try:
                await run_in_threadpool(
                    ai_container.animator.animate,
                    action=action,
                    char_path=work_dir,
                    char_name=anim_id,
                )
                logger.info(
                    f"     [STEP 3] {anim_id} Animation AI finished in {time.time() - ai_start:.2f}s")
            except Exception as e:
                logger.error(f"     [STEP 3] {anim_id} Error: {e}")
                raise HTTPException(
                    500, f"Animation generation failed: {str(e)}")

        # --- XỬ LÝ FILE OUTPUT ---
        output_filename = f"{action}.gif"
        output_path = os.path.join(work_dir, output_filename)

        if not os.path.exists(output_path):
            if os.path.exists(os.path.join(work_dir, "video.gif")):
                output_filename = "video.gif"
            else:
                gifs = [f for f in os.listdir(work_dir) if f.endswith(".gif")]
                if gifs:
                    output_filename = gifs[0]
                else:
                    logger.error(
                        f"     [STEP 3] {anim_id} GIF not found after processing")
                    raise HTTPException(500, "No GIF file found.")

        logger.info(
            f"<=== [STEP 3] Completed for {anim_id} in {time.time() - start_t:.2f}s")
        return {
            "status": "success",
            "action": action,
            "gif_url": f"{self.base_url}/{anim_id}/{output_filename}",
        }
