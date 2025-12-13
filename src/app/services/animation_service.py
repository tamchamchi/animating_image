import os
import uuid

import cv2
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from src.app.core.config import settings
from src.app.core.container import ai_container
from src.app.utils.image_ops import save_upload_file


class AnimationService:
    def __init__(self):
        self.base_url = "/static/animations"

    def _get_path(self, anim_id):
        return os.path.join(settings.STATIC_DIR, "animations", anim_id)

    async def init_session(self, file):
        anim_id = str(uuid.uuid4())
        work_dir = self._get_path(anim_id)
        os.makedirs(work_dir, exist_ok=True)

        # Lưu file gốc
        file_path = os.path.join(work_dir, "original.png")
        await save_upload_file(file, file_path)

        return {"id": anim_id, "status": "initialized"}

    async def step1_decompose(self, anim_id: str):
        work_dir = self._get_path(anim_id)
        input_path = os.path.join(work_dir, "original.png")
        if not os.path.exists(input_path):
            raise HTTPException(404, "Session not found or image missing")

        # Read image
        img = cv2.imread(input_path)

        async with ai_container.semaphore:
            # ConcreteObjectDecomposer trả về dict
            result = await run_in_threadpool(ai_container.decomposer.decompose, img)

        if not result:
            raise HTTPException(400, "Decomposition failed (No mask found)")

        # Lưu Mask và Texture
        cv2.imwrite(os.path.join(work_dir, "mask.png"), cv2.cvtColor(result["mask"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(work_dir, "texture.png"), cv2.cvtColor(result["texture"], cv2.COLOR_RGB2BGR))

        return {
            "mask_url": f"{self.base_url}/{anim_id}/mask.png",
            "texture_url": f"{self.base_url}/{anim_id}/texture.png",
        }

    async def step2_pose(self, anim_id: str):
        work_dir = self._get_path(anim_id)
        texture_path = os.path.join(
            work_dir, "texture.png"
        )  # Dùng texture hoặc original tùy logic
        if not os.path.exists(texture_path):
            texture_path = os.path.join(work_dir, "original.png")

        output_yaml = os.path.join(work_dir, "char_cfg.yaml")
        viz_output = os.path.join(work_dir, "pose_viz.png")

        async with ai_container.semaphore:
            # MMPoseEstimator
            await run_in_threadpool(
                ai_container.pose_estimator.predict,
                image=texture_path,
                output_file=viz_output,
                output_yaml=output_yaml,
            )

        return {
            "joint_yaml_url": f"{self.base_url}/{anim_id}/char_cfg.yaml",
            "pose_viz_url": f"{self.base_url}/{anim_id}/pose_viz.png",
        }

    async def step3_animate(self, anim_id: str, action: str = "walk"):
        """
        Bước 3: Tạo Animation GIF
        Args:
            anim_id: ID của session
            action: Tên hành động (ví dụ: 'walk', 'run', 'jump', 'wave')
        """
        work_dir = self._get_path(anim_id)

        # Validate: Cần đảm bảo các bước trước đã chạy xong
        required_files = ["char_cfg.yaml", "mask.png", "texture.png"]
        for f in required_files:
            if not os.path.exists(os.path.join(work_dir, f)):
                raise HTTPException(
                    400, f"Missing prerequisite file: {f}. Please run Step 1 & 2 first."
                )

        # Xử lý Logic AI
        async with ai_container.semaphore:
            try:
                # Gọi MetaAnimator
                # Lưu ý: output sẽ được sinh ra tại work_dir
                await run_in_threadpool(
                    ai_container.animator.animate,
                    action=action,
                    char_path=work_dir,
                    char_name=anim_id,
                )
            except Exception as e:
                print(f"Animation Error: {e}")
                raise HTTPException(500, f"Animation generation failed: {str(e)}")

        # --- XỬ LÝ FILE OUTPUT ---

        # Giả định file output có tên trùng với action, ví dụ: "walk.gif"
        output_filename = f"{action}.gif"
        output_path = os.path.join(work_dir, output_filename)

        # Kiểm tra xem file có thực sự tồn tại không (phòng trường hợp AI lỗi hoặc tên file khác)
        if not os.path.exists(output_path):
            # Fallback 1: Thử tìm file tên là video.gif (mặc định của một số thư viện)
            if os.path.exists(os.path.join(work_dir, "video.gif")):
                output_filename = "video.gif"
            # Fallback 2: Thử tìm file gif bất kỳ vừa được tạo
            else:
                gifs = [f for f in os.listdir(work_dir) if f.endswith(".gif")]
                if gifs:
                    output_filename = gifs[0]  # Lấy file gif đầu tiên tìm thấy
                else:
                    raise HTTPException(
                        500,
                        "Animation finished but no GIF file found in output directory.",
                    )

        return {
            "status": "success",
            "action": action,
            "gif_url": f"{self.base_url}/{anim_id}/{output_filename}",
        }
