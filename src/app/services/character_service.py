import logging
import os
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import cv2
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from src.app.core.config import settings
from src.app.core.container import ai_container
from src.app.utils.image_ops import read_image_as_numpy
from src.utils.prompt import PROMPT_SUBJECT_GENERATION

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class CharacterService:
    """
    Service class responsible for creating character images using AI models.
    Supports character creation from a face image or from a text prompt.
    """

    def __init__(self):
        """
        Initializes the CharacterService with the base URL for serving character images.
        """
        self.base_url = "/static/characters"

    def _get_workspace(self) -> Tuple[str, str]:
        """
        Creates a unique workspace directory for a new character.

        A new UUID is generated as the character ID, and a corresponding
        directory is created under the STATIC_DIR/characters path.

        Returns:
            A tuple containing:
            - char_id (str): The unique ID generated for the character.
            - work_dir (str): The absolute path to the character's workspace directory.
        """
        char_id = str(uuid.uuid4())
        path = os.path.join(settings.STATIC_DIR, "characters", char_id)
        os.makedirs(path, exist_ok=True)  # Ensure the directory exists
        logger.debug(f"Created workspace for character ID: {char_id} at {path}")
        return char_id, path

    async def create_from_face(
        self, file: UploadFile, body_file: UploadFile
    ) -> Dict[str, Any]:
        """
        Creates a character image by segmenting a face from an input image
        and merging it onto a cartoon body template.

        The process involves:
        1. Reading the input face image and body template image.
        2. Using an AI face segmenter to extract the face mask, region, bbox, and cropped face.
        3. Applying the segmented face onto the cartoon body template.
        4. Saving the final merged character image and the extracted face region.

        Args:
            file: An UploadFile object containing the input image with the face.
            body_file: An UploadFile object containing the cartoon body template image.

        Returns:
            A dictionary containing:
            - 'id' (str): The unique ID of the new character.
            - 'image_url' (str): The URL to the final merged character image.
            - 'face_url' (Optional[str]): The URL to the extracted face region image,
                                          or None if face region extraction failed.
        """
        start_t = time.time()
        char_id, work_dir = self._get_workspace()
        logger.info(f"===> [CHAR FACE] Creating character from face. ID: {char_id}")

        # 1. Read input face image and body template image
        # read_image_as_numpy expects file-like objects, UploadFile is one.
        input_img = await read_image_as_numpy(file)  # Assumes BGR format
        body_img = await read_image_as_numpy(body_file)  # Assumes BGR format

        # Check if images were read successfully
        if input_img is None or body_img is None:
            logger.error(f"[CHAR FACE] {char_id}: Failed to read input image(s).")
            # Consider raising an HTTPException here
            raise HTTPException(
                status_code=400, detail="Failed to read input image(s)."
            )

        # 3. AI Processing (using Threadpool for synchronous AI calls + Semaphore for concurrency control)
        logger.info(f"     [CHAR FACE] {char_id} waiting for SEMAPHORE...")
        async with (
            ai_container.semaphore
        ):  # Acquire semaphore to limit concurrent AI tasks
            logger.info(
                f"     [CHAR FACE] {char_id} ACQUIRED SEMAPHORE. Segmenting and Merging..."
            )
            ai_start = time.time()

            # Perform face segmentation in a thread pool as it's a synchronous operation
            face_mask, face_region, bbox, face_crop = await run_in_threadpool(
                ai_container.face_segmenter.segment, input_img
            )
            logger.info(
                f"     [CHAR FACE] Face segmentation finished in {time.time() - ai_start:.2f}s"
            )

            # Apply segmented face to the body template
            # Convert images to RGB as required by ai_container.face_segmenter.apply_to_body
            final_img_rgb = await run_in_threadpool(
                ai_container.face_segmenter.apply_to_body,
                cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
            )
            logger.info(
                f"     [CHAR FACE] Face application to body finished in {time.time() - ai_start:.2f}s (total AI time)"
            )

        # 4. Save results
        # Convert the final image back to BGR for saving with OpenCV
        final_img_bgr = cv2.cvtColor(final_img_rgb, cv2.COLOR_RGB2BGR)
        output_name = "character_result.png"
        output_path = os.path.join(work_dir, output_name)
        cv2.imwrite(output_path, final_img_bgr)
        logger.debug(f"Saved final character image to {output_path}")

        face_filename = "face_region.png"
        face_url: Optional[str] = None  # Initialize as Optional[str]

        # Save the extracted face region with its mask as an RGBA image
        if (
            face_region is not None
            and face_region.size > 0
            and face_mask is not None
            and face_mask.size > 0
        ):
            face_rgba = cv2.cvtColor(face_region, cv2.COLOR_BGR2BGRA)
            # Apply the mask to the alpha channel
            face_rgba[:, :, 3] = face_mask
            face_path = os.path.join(work_dir, face_filename)
            cv2.imwrite(face_path, face_rgba)
            face_url = f"{self.base_url}/{char_id}/{face_filename}"
            logger.debug(f"Saved face region image to {face_path}")
        else:
            logger.warning(
                f"[CHAR FACE] {char_id}: Face region or mask was empty. Not saving face_region.png."
            )

        logger.info(
            f"<=== [CHAR FACE] Character creation completed in {time.time() - start_t:.2f}s. ID: {char_id}"
        )

        return {
            "id": char_id,
            "image_url": f"{self.base_url}/{char_id}/{output_name}",
            "face_url": face_url,
        }

    async def create_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Generates a character image from a given text prompt using an AI generator.

        The process involves:
        1. Constructing a detailed prompt using a predefined subject generation template.
        2. Calling an AI image generation model with the constructed prompt.
        3. Saving the generated image.

        Args:
            prompt: The text description of the character to be generated.

        Returns:
            A dictionary containing:
            - 'id' (str): The unique ID of the new character.
            - 'image_url' (str): The URL to the generated character image.
        """
        start_t = time.time()
        char_id, work_dir = self._get_workspace()
        logger.info(
            f"===> [CHAR PROMPT] Generating character from prompt. ID: {char_id}"
        )

        # Format the input prompt with a predefined template for better AI generation
        full_prompt = PROMPT_SUBJECT_GENERATION.format(subject=prompt)
        logger.debug(f"     [CHAR PROMPT] {char_id} using prompt: {full_prompt}")

        # AI Processing (using Semaphore for concurrency control)
        async with (
            ai_container.semaphore
        ):  # Acquire semaphore to limit concurrent AI tasks
            logger.info(
                f"     [CHAR PROMPT] {char_id} ACQUIRED SEMAPHORE. Calling Generator..."
            )
            ai_start = time.time()
            # Call the AI generator in a thread pool (assuming generator.generate is synchronous)
            pil_image: Image.Image = await run_in_threadpool(
                ai_container.generator.generate, full_prompt
            )
            logger.info(
                f"     [CHAR PROMPT] AI generation finished in {time.time() - ai_start:.2f}s"
            )

        # 4. Save results
        output_name = "generated_char.png"
        output_path = os.path.join(work_dir, output_name)
        pil_image.save(output_path)  # PIL Image object has a .save() method
        logger.debug(f"Saved generated character image to {output_path}")
        logger.info(
            f"<=== [CHAR PROMPT] Character generation completed in {time.time() - start_t:.2f}s. ID: {char_id}"
        )

        return {"id": char_id, "image_url": f"{self.base_url}/{char_id}/{output_name}"}
