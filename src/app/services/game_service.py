import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiofiles
import numpy as np
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageSequence

from src.app.core.config import settings

logger = logging.getLogger(__name__)


class GameService:
    """
    Service class to handle game-related resources, including processing
    and serving animated GIFs, background images, and detected object data.
    """

    def __init__(self):
        """
        Initializes the GameService with the base URL for static animations.
        """
        # Base URL for accessing static animation files.
        self.base_url = "/static/animations"

    def _get_path(self, anim_id: str) -> str:
        """
        Constructs the absolute file system path for a given animation ID.

        Args:
            anim_id: The unique identifier for the animation set (e.g., a game ID).

        Returns:
            The full file system path to the animation directory.
        """
        return os.path.join(settings.STATIC_DIR, "animations", anim_id)

    def _trim_gif_bottom(self, file_path: str) -> str:
        """
        Processes a GIF file to trim excess transparent space at the bottom,
        shift characters down to the base, and add a subtle left-shifted shadow,
        while preserving transparency and animation.

        The trimmed GIF is saved to the same directory with "_trimmed" suffix.
        If a trimmed version already exists, its filename is returned.

        Args:
            file_path: The full path to the input GIF file.

        Returns:
            The filename of the processed (trimmed) GIF if successful,
            or the original filename if an error occurred or no trimming was needed.
        """
        file_dir = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)

        trimmed_filename = f"{name}_trimmed{ext}"
        trimmed_path = os.path.join(file_dir, trimmed_filename)

        # Check if the trimmed version already exists to avoid re-processing.
        if os.path.exists(trimmed_path):
            logger.debug(f"Trimmed GIF already exists: {trimmed_filename}")
            return trimmed_filename

        try:
            with Image.open(file_path) as im:
                # Store original GIF transparency index if available.
                transparency_index = im.info.get("transparency")

                max_bottom = 0
                frames_raw = []  # Store original frames

                # --- 1. Detect the lowest non-transparent pixel across all frames ---
                for frame_idx, frame in enumerate(ImageSequence.Iterator(im)):
                    # Ensure frame is converted to RGBA for consistent transparency check
                    # and create a copy to avoid issues with PIL's lazy loading.
                    original_rgba = frame.convert("RGBA")
                    frames_raw.append(original_rgba)

                    # Get bounding box of non-transparent content
                    bbox = original_rgba.getbbox()

                    if bbox:
                        # Update max_bottom if the current frame's content goes lower
                        max_bottom = max(max_bottom, bbox[3])
                    else:
                        # If a frame is entirely transparent, its bbox is None.
                        # This scenario needs careful handling; if ALL frames are empty, max_bottom remains 0.
                        pass

                # If no content was found in any frame, return original filename.
                if max_bottom == 0:
                    logger.info(f"No visible content in GIF {filename}. Skipping trim.")
                    return filename

                width = im.size[0]
                # cropped_height = max_bottom # The height of the new canvas
                processed_frames: List[Image.Image] = []

                # --- 2. Process each frame: crop, shift, add shadow ---
                for frame_idx, rgba_frame in enumerate(frames_raw):
                    # Crop the frame to the detected max_bottom height
                    cropped_rgba = rgba_frame.crop((0, 0, width, max_bottom))

                    # Convert to NumPy array for alpha channel manipulation
                    arr = np.array(cropped_rgba)
                    alpha = arr[:, :, 3]  # Extract alpha channel

                    # Find the lowest non-transparent pixel in the current frame
                    ys, xs = np.where(alpha > 10)  # Alpha threshold for visibility

                    if len(ys) > 0:
                        # Calculate vertical shift needed to move character to bottom of 'max_bottom' canvas
                        current_bottom = ys.max()
                        shift_y = (
                            (max_bottom - 1) - current_bottom
                        )  # max_bottom is height, (max_bottom-1) is last index

                        if shift_y > 0:
                            # Create a new transparent canvas and paste the shifted character
                            shifted_canvas = Image.new(
                                "RGBA", (width, max_bottom), (0, 0, 0, 0)
                            )
                            shifted_canvas.paste(cropped_rgba, (0, shift_y))
                            processed_rgba = shifted_canvas
                        else:
                            processed_rgba = cropped_rgba
                    else:
                        # If frame has no visible content, keep it transparent on the max_bottom canvas
                        processed_rgba = Image.new(
                            "RGBA", (width, max_bottom), (0, 0, 0, 0)
                        )

                    # --- Build Shadow ---
                    arr_with_shift = np.array(processed_rgba)
                    alpha_shifted = arr_with_shift[:, :, 3]

                    # Create a shadow from the character's alpha, slightly less opaque.
                    shadow_arr = np.zeros_like(arr_with_shift)
                    shadow_arr[:, :, 3] = (alpha_shifted * 0.8).astype(
                        np.uint8
                    )  # 80% opacity for shadow
                    shadow_img = Image.fromarray(shadow_arr, mode="RGBA")

                    # Create a larger canvas to accommodate the shadow shift.
                    # Shadow is shifted 5px to the left, character is shifted 10px to the right
                    # Relative shift of shadow to character is 5px left.
                    canvas_w = width + 10  # Extra 10px on right for character shift
                    canvas_h = max_bottom

                    final_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

                    # Paste shadow shifted 5px to the left
                    final_canvas.paste(
                        shadow_img, (5, 0), shadow_img
                    )  # Shift shadow right by 5px

                    # Paste character on top, shifted 10px to the right
                    final_canvas.paste(processed_rgba, (10, 0), processed_rgba)

                    # Convert the final frame to P mode for GIF compatibility
                    # Use ADAPTIVE palette for better color quality
                    final_frame_p = final_canvas.convert("P", palette=Image.ADAPTIVE)
                    processed_frames.append(final_frame_p)

                # --- 3. Save the processed GIF ---
                if processed_frames:
                    save_kwargs = {
                        "save_all": True,
                        "append_images": processed_frames[
                            1:
                        ],  # All frames except the first one
                        "loop": 0,  # Loop indefinitely
                        "duration": im.info.get(
                            "duration", 100
                        ),  # Preserve original frame duration
                        "disposal": 2,  # Restore background after each frame (prevents ghosting)
                        "optimize": False,  # Optimize might remove transparency in some cases
                    }

                    # Re-apply original GIF transparency index if it existed
                    if transparency_index is not None:
                        save_kwargs["transparency"] = transparency_index

                    # The first frame saves the initial setup and metadata
                    processed_frames[0].save(trimmed_path, **save_kwargs)
                    logger.info(
                        f"Successfully trimmed and saved GIF: {trimmed_filename}"
                    )
                    return trimmed_filename

        except Exception as e:
            logger.error(f"Error trimming GIF {filename}: {e}", exc_info=True)
            # If an error occurs, return the original filename to ensure the GIF is still served.
            return filename

        # Fallback return in case no frames were processed or saved (though 'if processed_frames' handles most of this)
        logger.warning(
            f"Unexpected path: GIF trimming for {filename} did not produce frames."
        )
        return filename

    async def get_resources(self, game_id: str) -> Dict[str, Any]:
        """
        Retrieves and processes all resources for a given game ID, including
        animated GIFs, background images, and detected object JSON data.

        Args:
            game_id: The ID of the game whose resources are to be fetched.

        Returns:
            A dictionary containing:
            - 'game_id': The requested game ID.
            - 'action_gif_urls': A sorted list of URLs for processed action GIFs.
            - 'background_url': The URL for the background image, or None if not found.
            - 'detected_objects': A list of dictionaries from 'detected_objects.json',
                                  or an empty list if the file is not found or invalid.

        Raises:
            HTTPException: If the game ID directory is not found (status 404).
        """
        work_dir = self._get_path(game_id)

        # Check if the directory for the game ID exists.
        if not os.path.exists(work_dir):
            logger.warning(f"Game ID directory not found: {work_dir}")
            raise HTTPException(status_code=404, detail="Game ID not found")

        logger.info(f"Retrieving resources for game_id: {game_id} from {work_dir}")

        # --- 1. Processing GIFs ---
        files = os.listdir(work_dir)
        # Filter for raw GIF files (excluding already trimmed ones).
        raw_gifs = [
            f for f in files if f.lower().endswith(".gif") and "_trimmed" not in f
        ]

        final_gif_urls: List[str] = []

        # Process each raw GIF in a separate thread to avoid blocking the event loop.
        for gif_file in raw_gifs:
            full_path = os.path.join(work_dir, gif_file)
            logger.debug(f"Processing GIF: {gif_file}")

            # The _trim_gif_bottom method is synchronous, so run it in a thread pool.
            processed_filename = await run_in_threadpool(
                self._trim_gif_bottom, full_path
            )
            # Construct the URL for the processed GIF.
            final_gif_urls.append(f"{self.base_url}/{game_id}/{processed_filename}")

        final_gif_urls.sort()  # Ensure consistent order of GIF URLs.

        # --- 2. Discover Background Image ---
        bg_file: Optional[str] = None
        # Check for common image extensions for the background file.
        for ext in ["png", "jpg", "jpeg"]:
            possible_bg = f"background.{ext}"
            if os.path.exists(os.path.join(work_dir, possible_bg)):
                bg_file = possible_bg
                logger.debug(f"Background found: {bg_file}")
                break  # Found background, stop checking other extensions

        # Construct the background URL if a file was found.
        bg_url: Optional[str] = (
            f"{self.base_url}/{game_id}/{bg_file}" if bg_file else None
        )

        # --- 3. Load JSON Objects ---
        json_filename_primary = "detected_objects.json"
        json_filename_fallback = (
            "detected_object.json"  # Fallback for possible singular name
        )

        json_path_primary = os.path.join(work_dir, json_filename_primary)
        json_path_fallback = os.path.join(work_dir, json_filename_fallback)

        detected_objects: List[Dict[str, Any]] = []

        # Check for the primary JSON file, then fallback.
        if os.path.exists(json_path_primary):
            json_path_to_use = json_path_primary
        elif os.path.exists(json_path_fallback):
            json_path_to_use = json_path_fallback
        else:
            json_path_to_use = None
            logger.warning(
                f"No detected_objects.json or detected_object.json found in {work_dir}"
            )

        if json_path_to_use:
            logger.debug(f"Loading detected objects from {json_path_to_use}")
            try:
                # Use aiofiles for async file reading.
                async with aiofiles.open(
                    json_path_to_use, mode="r", encoding="utf-8"
                ) as f:
                    content = await f.read()
                    detected_objects = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding JSON from {json_path_to_use}: {e}", exc_info=True
                )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred while reading JSON from {json_path_to_use}: {e}",
                    exc_info=True,
                )

        logger.info(f"Finished retrieving resources for game_id: {game_id}")
        return {
            "game_id": game_id,
            "action_gif_urls": final_gif_urls,
            "background_url": bg_url,
            "detected_objects": detected_objects,
        }
