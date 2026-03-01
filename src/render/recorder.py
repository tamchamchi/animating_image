import imageio
import numpy as np
import pygame


class VideoRecorder:
    def __init__(self, filename, width, height, fps):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps

        # writer tự lo ffmpeg backend
        self.writer = imageio.get_writer(
            self.filename,
            fps=self.fps,
            codec="libx264",
            quality=8,
            pixelformat='yuv420p',
        )

    def write(self, screen):
        """Capture a frame from pygame."""
        frame = pygame.surfarray.array3d(screen)

        # Correct orientation: pygame is (w,h,3) but rotated
        frame = np.rot90(frame, -1)
        frame = np.flip(frame, axis=1)

        self.writer.append_data(frame)

    def close(self):
        self.writer.close()
