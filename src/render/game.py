import json
import os
import sys

import pygame

# --- IMPORTS ---
from .config import CFG, PLAYER_START_X, PLAYER_START_Y, SCREEN_H, SCREEN_W
from .platform import Platform
from .player import Player
from .utils import load_gif_frames


class Game:
    """
    The main Game Engine class.

    This class orchestrates the game loop, manages assets, handles input,
    updates game state, and renders graphics to the screen.
    """

    def __init__(self, data_path):
        """
        Initialize the Game engine.

        Args:
            data_path (str): The directory path containing game assets
                             (gifs, background.jpg, level_data.json).
        """
        # 1. Init Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(CFG["screen"]["caption"])
        self.clock = pygame.time.Clock()
        self.running = True

        # 2. Config Shortcuts
        self.fps = CFG["screen"]["fps"]
        self.player_speed = CFG["physics"]["player_speed"]

        # 3. Load Assets & Dynamic Dimensions
        # Note: This step must happen before level setup to determine
        # the original background dimensions (original_w, original_h).
        self.load_assets(data_path)

        # 4. Setup Level (Platforms)
        # Initializes the sprite group for platforms based on the loaded dimensions.
        self.platforms = pygame.sprite.Group()
        self.setup_level(data_path)

        # 5. Setup Player
        start_x = PLAYER_START_X
        start_y = PLAYER_START_Y
        self.player = Player(start_x, start_y, self.animations)
        self.start_pos = (start_x, start_y)

    def load_assets(self, data_path):
        """
        Loads game assets from the specified directory.

        It loads the background image first to establish the coordinate system
        scale, then loads character animations.
        """
        # --- A. PROCESS BACKGROUND & ORIGINAL DIMENSIONS ---
        bg_path = os.path.join(data_path, "background.jpg")

        if os.path.exists(bg_path):
            # Load the raw image to get the actual source dimensions
            raw_bg = pygame.image.load(bg_path)
            w, h = raw_bg.get_size()

            # Store original dimensions for coordinate scaling later
            self.original_w = w
            self.original_h = h

            # Scale the background to fit the current screen configuration
            self.bg_img = pygame.transform.scale(raw_bg, (SCREEN_W, SCREEN_H))
            print(f"Loaded Background: {w}x{h} -> Scaled to System Config")
        else:
            print(f"Error: Background not found at {bg_path}")
            sys.exit()

        # --- B. LOAD ANIMATIONS ---
        asset_scale = tuple(CFG["assets"]["scale"])

        # Load GIF frames for different player states
        self.animations = {
            "idle": load_gif_frames(
                os.path.join(data_path, "waving.gif"), 1, asset_scale
            ),
            "run": load_gif_frames(
                os.path.join(data_path, "running.gif"), 1, asset_scale
            ),
            "jump": load_gif_frames(
                os.path.join(data_path, "jumping.gif"), 1, asset_scale
            ),
            "dance": load_gif_frames(
                os.path.join(data_path, "jesse_dancing.gif"), 1, asset_scale
            ),
            "speak": load_gif_frames(
                os.path.join(data_path, "speaking.gif"), 1, asset_scale
            ),
        }

    def setup_level(self, data_path):
        """
        Parses level data from JSON and creates platform objects.

        Args:
            data_path (str): Path to the folder containing 'level_data.json'.
        """
        # A. Create the Ground Floor
        # This is a fixed platform based on the original image height.
        ground_poly = [
            [0, self.original_h],
            [0, self.original_h - 50],
            [self.original_w, self.original_h - 50],
            [self.original_w, self.original_h],
        ]
        self.platforms.add(
            Platform(ground_poly, "Ground", self.original_w, self.original_h)
        )

        # B. Load Dynamic Platforms from JSON
        json_path = os.path.join(data_path, "level_data.json")
        try:
            with open(json_path, "r") as f:
                detected_objects = json.load(f)

            for obj in detected_objects:
                # Ensure the polygon has at least 3 points to be valid
                if len(obj.get("polygon", [])) >= 3:
                    p = Platform(
                        obj["polygon"],
                        obj.get("name", "obj"),
                        self.original_w,
                        self.original_h,
                    )
                    self.platforms.add(p)

        except FileNotFoundError:
            print(f"Warning: {json_path} not found. Only ground platform loaded.")

    def handle_events(self):
        """
        Processes external events such as keyboard input and window close requests.
        """
        for event in pygame.event.get():
            # Handle Window Close
            if event.type == pygame.QUIT:
                self.running = False

            # Handle Key Presses
            if event.type == pygame.KEYDOWN:
                # Jump actions
                if event.key in [pygame.K_UP, pygame.K_SPACE]:
                    self.player.jump()

                # Reset player position
                if event.key == pygame.K_r:
                    self.player.reset_position(*self.start_pos)

    def update(self):
        """
        Updates the game state for the current frame.
        """
        keys = pygame.key.get_pressed()
        dx = 0

        # Horizontal movement input
        if keys[pygame.K_RIGHT]:
            dx = self.player_speed
        if keys[pygame.K_LEFT]:
            dx = -self.player_speed

        # Action states
        is_dancing = keys[pygame.K_d]
        is_speaking = keys[pygame.K_s]

        # Priority Logic: Dancing prevents movement
        if is_dancing:
            dx = 0

        # Update player physics and animation
        self.player.update(dx, is_dancing, is_speaking, self.platforms)

    def draw(self):
        """
        Renders all game objects to the screen.
        Order: Background -> Platforms -> Player.
        """
        # Draw background
        self.screen.blit(self.bg_img, (0, 0))

        # Draw all platforms in the group
        for p in self.platforms:
            p.draw(self.screen)

        # Draw the player
        self.player.draw(self.screen)

        # Flip the display buffer
        pygame.display.update()

    def run(self):
        """
        The main game loop. Runs until self.running is False.
        """
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            # Maintain the target FPS
            self.clock.tick(self.fps)

        # Cleanup and exit
        pygame.quit()
        sys.exit()
