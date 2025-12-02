import os
import pygame
import numpy as np
from PIL import Image
from .config import SCREEN_W, SCREEN_H


def get_scaled_points(raw_points, original_w, original_h):
    """
    Transforms a list of coordinates from the original asset resolution 
    to the current screen resolution.

    Args:
        raw_points (list): List of (x, y) tuples in the original resolution.
        original_w (int): The width of the original background/asset.
        original_h (int): The height of the original background/asset.

    Returns:
        list: A list of scaled (x, y) tuples fitting the current screen size.
    """
    # Calculate scaling factors based on current screen configuration
    scale_x = SCREEN_W / original_w
    scale_y = SCREEN_H / original_h
    
    # Apply scaling to each point
    return [(p[0] * scale_x, p[1] * scale_y) for p in raw_points]


def load_image(path, size=None):
    """
    Safely loads an image from disk and optionally resizes it.

    Args:
        path (str): File path to the image.
        size (tuple, optional): Target (width, height) to scale the image.

    Returns:
        pygame.Surface: The loaded image, or a placeholder surface if loading fails.
    """
    if os.path.exists(path):
        # Load and convert for faster blitting
        img = pygame.image.load(path).convert()
        if size:
            img = pygame.transform.scale(img, size)
        return img
    
    # Return a red placeholder if file is missing
    return pygame.Surface(size if size else (50, 50))


def load_gif_frames(path, skip_frames=4, scale=(100, 100)):
    """
    Loads a GIF file, extracts frames, resizes them, and aligns the content 
    to the bottom of the surface.

    This function uses PIL (Pillow) for GIF extraction and NumPy for 
    pixel-perfect alignment (ensuring characters' feet touch the ground).

    Args:
        path (str): Path to the .gif file.
        skip_frames (int): Number of frames to skip to adjust animation speed/memory.
        scale (tuple): Target size (width, height) for each frame.

    Returns:
        list: A list of pygame.Surface objects representing the animation frames.
    """
    # Return a placeholder if the file doesn't exist
    if not os.path.exists(path):
        s = pygame.Surface(scale, pygame.SRCALPHA)
        pygame.draw.rect(s, (255, 0, 0), (0, 0, scale[0], scale[1]))
        return [s]

    frames = []
    try:
        pil_img = Image.open(path)
        # Get total frames (default to 1 if not animated)
        num_frames = getattr(pil_img, 'n_frames', 1)

        for i in range(0, num_frames, skip_frames):
            pil_img.seek(i)
            
            # Resize frame using high-quality Lanczos resampling
            frame = pil_img.convert("RGBA").resize(
                scale, Image.Resampling.LANCZOS)

            # --- AUTOMATIC BOTTOM ALIGNMENT LOGIC ---
            # Convert image to numpy array to analyze pixel data
            arr = np.array(frame)
            alpha = arr[:, :, 3] # Extract Alpha channel
            
            # Find indices of all non-transparent pixels (threshold > 10)
            ys, xs = np.where(alpha > 10)

            if len(ys) > 0:
                height = arr.shape[0]
                bottom = ys.max() # The lowest pixel containing the character
                
                # Calculate how much we need to shift the image down
                shift_y = (height - 1) - bottom

                # Apply shift if the character is floating
                if shift_y != 0:
                    new_frame = Image.new("RGBA", scale, (0, 0, 0, 0))
                    # Paste the original content at the new shifted position
                    new_frame.paste(frame, (0, shift_y))
                    frame = new_frame
            # ----------------------------------------

            # Convert PIL image back to Pygame Surface
            img = pygame.image.fromstring(
                frame.tobytes(), frame.size, frame.mode)
            frames.append(img)
            
        return frames

    except Exception as e:
        print(f"Error loading GIF {path}: {e}")
        # Return a blank surface in case of error
        return [pygame.Surface(scale)]