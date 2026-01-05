import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

import cv2
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.app.core.config import settings
from src.app.core.container import ai_container
from src.app.utils.image_ops import (
    save_upload_file,
)  # Added read_image_as_numpy for safety

# Initialize logger for this module
logger = logging.getLogger(__name__)


class AnimationService:
    """
    Service class responsible for handling the multi-step animation generation process.
    This includes initializing a session, decomposing a character image, estimating pose,
    and finally generating an animation based on a specified action.
    """

    def __init__(self):
        """
        Initializes the AnimationService with the base URL for serving static animations.
        """
        self.base_url = "/static/animations"

    def _get_path(self, anim_id: str) -> str:
        """
        Constructs the absolute file system path for a given animation ID.

        Args:
            anim_id: The unique identifier for the animation session.

        Returns:
            The full file system path to the animation's workspace directory.
        """
        return os.path.join(settings.STATIC_DIR, "animations", anim_id)

    async def init_session(self, file: UploadFile) -> Dict[str, str]:
        """
        Initializes a new animation session by creating a unique workspace
        and saving the original character image.

        Args:
            file: An UploadFile object containing the original character image (e.g., PNG).

        Returns:
            A dictionary containing:
            - 'id' (str): The unique ID of the new animation session.
            - 'status' (str): The status of the initialization ("initialized").
        """
        start_t = time.time()
        logger.info("===> [INIT SESSION] Starting new animation session.")

        anim_id = str(uuid.uuid4())
        work_dir = self._get_path(anim_id)
        os.makedirs(work_dir, exist_ok=True)  # Ensure the workspace directory exists

        # Save the original input character image to the workspace
        original_image_filename = "original.png"
        file_path = os.path.join(work_dir, original_image_filename)
        await save_upload_file(file, file_path)
        logger.debug(f"Saved original image for session {anim_id} to {file_path}")

        logger.info(
            f"<=== [INIT SESSION] Session {anim_id} initialized in {time.time() - start_t:.2f}s."
        )
        return {
            "id": anim_id,
            "status": "initialized",
            "original_image_url": f"{self.base_url}/{anim_id}/{original_image_filename}",
        }

    async def step1_decompose(self, anim_id: str) -> Dict[str, str]:
        """
        Performs the decomposition step of the character animation pipeline.
        This involves using an AI model to decompose the original character image
        into a mask and a texture.

        Args:
            anim_id: The ID of the animation session.

        Returns:
            A dictionary containing URLs to the generated mask and texture images.

        Raises:
            HTTPException: If the session is not found, original image is missing (404),
                           or if decomposition fails (400).
        """
        start_t = time.time()
        logger.info(f"===> [STEP 1] Starting decomposition for session ID: {anim_id}")

        work_dir = self._get_path(anim_id)
        input_path = os.path.join(work_dir, "original.png")

        # Verify that the original image exists for the session
        if not os.path.exists(input_path):
            logger.error(
                f"     [STEP 1] Session {anim_id} not found or original image missing at {input_path}"
            )
            raise HTTPException(
                status_code=404, detail="Session not found or original image missing"
            )

        # Read the image using OpenCV (BGR format)
        # Using read_image_as_numpy which handles file I/O safely and returns numpy array
        # Assuming read_image_as_numpy can also take a file path if implemented for it,
        # otherwise cv2.imread is fine as it's typically faster for local files.
        img = cv2.imread(input_path)
        if img is None:
            logger.error(
                f"     [STEP 1] Could not read image at {input_path} for session {anim_id}."
            )
            raise HTTPException(
                status_code=500, detail="Failed to read original image."
            )

        logger.info(f"     [STEP 1] Session {anim_id} is waiting for AI SEMAPHORE...")
        async with (
            ai_container.semaphore
        ):  # Acquire semaphore for AI model concurrency control
            logger.info(
                f"     [STEP 1] Session {anim_id} ACQUIRED SEMAPHORE. Running AI Decomposition Model..."
            )
            ai_start = time.time()
            # Run the decomposer model in a thread pool as it's a synchronous operation
            result = await run_in_threadpool(ai_container.decomposer.decompose, img)
            logger.info(
                f"     [STEP 1] Session {anim_id} AI Decomposition Model finished in {time.time() - ai_start:.2f}s"
            )

        # Check if the decomposition yielded a valid result
        if (
            not result
            or "mask" not in result
            or "texture" not in result
            or result["mask"] is None
            or result["texture"] is None
        ):
            logger.error(
                f"     [STEP 1] Session {anim_id} Decomposition failed (No valid mask or texture found)."
            )
            raise HTTPException(
                status_code=400,
                detail="Decomposition failed: No valid mask or texture found.",
            )

        # Save the generated Mask and Texture images
        mask_path = os.path.join(work_dir, "mask.png")
        texture_path = os.path.join(work_dir, "texture.png")
        cv2.imwrite(
            mask_path, cv2.cvtColor(result["mask"], cv2.COLOR_RGB2BGR)
        )  # Convert back to BGR for saving
        cv2.imwrite(texture_path, cv2.cvtColor(result["texture"], cv2.COLOR_RGB2BGR))
        logger.debug(f"Saved mask to {mask_path} and texture to {texture_path}")

        logger.info(
            f"<=== [STEP 1] Decomposition for {anim_id} completed in {time.time() - start_t:.2f}s."
        )
        return {
            "mask_url": f"{self.base_url}/{anim_id}/mask.png",
            "texture_url": f"{self.base_url}/{anim_id}/texture.png",
        }

    async def step2_pose(self, anim_id: str) -> Dict[str, str]:
        """
        Estimates the pose of the character using an AI pose estimation model.
        This step requires the 'texture.png' generated from decomposition.

        Args:
            anim_id: The ID of the animation session.

        Returns:
            A dictionary containing URLs to the generated joint configuration (YAML)
            and a visualization of the estimated pose.

        Raises:
            HTTPException: If the texture image is missing (404).
        """
        start_t = time.time()
        logger.info(f"===> [STEP 2] Starting pose estimation for session ID: {anim_id}")

        work_dir = self._get_path(anim_id)
        texture_path = os.path.join(work_dir, "texture.png")

        # Fallback to original.png if texture.png is not found (though decomposition should produce texture.png)
        if not os.path.exists(texture_path):
            texture_path = os.path.join(work_dir, "original.png")
            logger.warning(
                f"     [STEP 2] Texture.png not found for {anim_id}, falling back to original.png."
            )
            if not os.path.exists(texture_path):
                logger.error(
                    f"     [STEP 2] No texture.png or original.png found for session {anim_id}."
                )
                raise HTTPException(
                    status_code=404,
                    detail="Required texture or original image not found.",
                )

        output_yaml = os.path.join(work_dir, "char_cfg.yaml")
        viz_output = os.path.join(work_dir, "pose_viz.png")

        logger.info(f"     [STEP 2] Session {anim_id} is waiting for AI SEMAPHORE...")
        async with (
            ai_container.semaphore
        ):  # Acquire semaphore for AI model concurrency control
            logger.info(
                f"     [STEP 2] Session {anim_id} ACQUIRED SEMAPHORE. Running AI Pose Estimator (MMPose)..."
            )
            ai_start = time.time()
            # Run the pose estimation model in a thread pool
            await run_in_threadpool(
                ai_container.pose_estimator.predict,
                image=texture_path,  # Input image for pose estimation
                output_file=viz_output,  # Path to save pose visualization
                output_yaml=output_yaml,  # Path to save joint configuration
            )
            logger.info(
                f"     [STEP 2] Session {anim_id} AI Pose Estimator finished in {time.time() - ai_start:.2f}s"
            )

        # Verify that output files were generated
        if not os.path.exists(output_yaml):
            logger.error(
                f"     [STEP 2] Pose estimation for {anim_id} failed: char_cfg.yaml not generated."
            )
            raise HTTPException(
                status_code=500,
                detail="Pose estimation failed: Joint configuration not generated.",
            )
        if not os.path.exists(viz_output):
            logger.error(
                f"     [STEP 2] Pose estimation for {anim_id} failed: pose_viz.png not generated."
            )
            # This might be less critical than YAML, so adjust status code if needed
            raise HTTPException(
                status_code=500,
                detail="Pose estimation failed: Visualization not generated.",
            )

        logger.info(
            f"<=== [STEP 2] Pose estimation for {anim_id} completed in {time.time() - start_t:.2f}s."
        )
        return {
            "joint_yaml_url": f"{self.base_url}/{anim_id}/char_cfg.yaml",
            "pose_viz_url": f"{self.base_url}/{anim_id}/pose_viz.png",
        }

    async def step3_animate(self, anim_id: str, action: str = "walk") -> Dict[str, Any]:
        """
        Generates an animation (GIF) for the character based on a specified action.
        This step requires the outputs from decomposition and pose estimation.

        Args:
            anim_id: The ID of the animation session.
            action: The desired animation action (e.g., "walk", "run", "idle").
                    Defaults to "walk".

        Returns:
            A dictionary containing:
            - 'status' (str): "success" if animation was generated.
            - 'action' (str): The action requested.
            - 'gif_url' (str): The URL to the generated animation GIF.

        Raises:
            HTTPException: If any prerequisite files are missing (400),
                           or if animation generation fails (500).
        """
        start_t = time.time()
        logger.info(
            f"===> [STEP 3] Starting animation for session ID: {anim_id} | Action: {action}"
        )

        work_dir = self._get_path(anim_id)
        # List of files required from previous steps
        required_files = ["char_cfg.yaml", "mask.png", "texture.png"]
        for f_name in required_files:
            file_path = os.path.join(work_dir, f_name)
            if not os.path.exists(file_path):
                logger.error(
                    f"     [STEP 3] Session {anim_id} missing prerequisite file: {f_name} at {file_path}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing prerequisite file for animation: {f_name}",
                )

        logger.info(f"     [STEP 3] Session {anim_id} is waiting for AI SEMAPHORE...")
        async with (
            ai_container.semaphore
        ):  # Acquire semaphore for AI model concurrency control
            logger.info(
                f"     [STEP 3] Session {anim_id} ACQUIRED SEMAPHORE. Running Animation Generator (Very Heavy)..."
            )
            ai_start = time.time()
            try:
                # Run the animator model in a thread pool
                await run_in_threadpool(
                    ai_container.animator.animate,
                    action=action,  # Desired animation action
                    char_path=work_dir,  # Path to character's workspace
                    char_name=anim_id,  # Character/session name
                )
                logger.info(
                    f"     [STEP 3] Session {anim_id} Animation AI finished in {time.time() - ai_start:.2f}s"
                )
            except Exception as e:
                logger.error(
                    f"     [STEP 3] Session {anim_id} Animation generation failed: {e}",
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=500, detail=f"Animation generation failed: {str(e)}"
                )

        # --- Handle Output GIF File ---
        # The expected output filename based on action.
        expected_output_filename = f"{action}.gif"
        output_path = os.path.join(work_dir, expected_output_filename)

        final_output_filename: Optional[str] = None

        # Check for the expected filename first
        if os.path.exists(output_path):
            final_output_filename = expected_output_filename
        else:
            # Fallback for generic "video.gif"
            if os.path.exists(os.path.join(work_dir, "video.gif")):
                final_output_filename = "video.gif"
                logger.warning(
                    f"     [STEP 3] Expected '{expected_output_filename}' not found. Using 'video.gif' for session {anim_id}."
                )
            else:
                # Fallback: find any GIF if named differently
                gifs = [f for f in os.listdir(work_dir) if f.lower().endswith(".gif")]
                if gifs:
                    final_output_filename = gifs[0]
                    logger.warning(
                        f"     [STEP 3] Using first found GIF '{final_output_filename}' for session {anim_id}."
                    )
                else:
                    logger.error(
                        f"     [STEP 3] No GIF file found after animation processing for session {anim_id}."
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="No GIF file found after animation generation.",
                    )

        logger.info(
            f"<=== [STEP 3] Animation for {anim_id} completed in {time.time() - start_t:.2f}s. Output: {final_output_filename}"
        )
        return {
            "status": "success",
            "action": action,
            "gif_url": f"{self.base_url}/{anim_id}/{final_output_filename}",
        }
