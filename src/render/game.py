import json
import os
import sys

import pygame

# --- IMPORTS ---
from .config import CFG, PLAYER_START_X, PLAYER_START_Y, SCREEN_H, SCREEN_W
from .platform import Platform
from .player import Player
from .utils import load_gif_frames, create_spotlight_mask
from .recorder import VideoRecorder


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
            data_path (str): The directory path containing game assets.
        """
        # 1. Init Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.record_surface = pygame.Surface((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(CFG["screen"]["caption"])
        self.clock = pygame.time.Clock()
        self.running = True

        # 2. Config Shortcuts
        self.fps = CFG["screen"]["fps"]
        self.player_speed = CFG["physics"]["player_speed"]

        # 3. Load Assets & Dynamic Dimensions
        self.load_assets(data_path)

        # 4. Setup Level (Platforms)
        self.platforms = pygame.sprite.Group()
        self.setup_level(data_path)

        # 5. Setup Player
        start_x = PLAYER_START_X
        start_y = PLAYER_START_Y
        self.player = Player(start_x, start_y, self.animations)
        self.start_pos = (start_x, start_y)

        # 6. Setup Spotlight / Lighting Effect
        self.spotlight_radius = CFG["lighting"]["radius"]
        self.darkness_opacity = CFG["lighting"]["opacity"]

        # Create the darkness layer (covers the whole screen)
        self.darkness_layer = pygame.Surface(
            (SCREEN_W, SCREEN_H), pygame.SRCALPHA)

        # Generate the light texture once (performance optimization)
        self.spotlight_img = create_spotlight_mask(
            self.spotlight_radius, self.darkness_opacity
        )

    def load_assets(self, data_path):
        """
        Loads game assets from the specified directory.
        """
        # --- A. PROCESS BACKGROUND ---
        bg_path = os.path.join(data_path, "background.jpg")
        record_path = os.path.join(data_path, "record.mp4")

        self.recorder = VideoRecorder(
            record_path, SCREEN_W, SCREEN_H, self.fps)

        if os.path.exists(bg_path):
            raw_bg = pygame.image.load(bg_path)
            w, h = raw_bg.get_size()
            self.original_w = w
            self.original_h = h
            self.bg_img = pygame.transform.scale(raw_bg, (SCREEN_W, SCREEN_H))
            print(f"Loaded Background: {w}x{h} -> Scaled to System Config")
        else:
            print(f"Error: Background not found at {bg_path}")
            sys.exit()

        # --- B. LOAD ANIMATIONS ---
        asset_scale = tuple(CFG["assets"]["scale"])
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
        """
        # A. Create the Ground Floor
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
        json_path = os.path.join(data_path, "detected_objects.json")
        try:
            with open(json_path, "r") as f:
                detected_objects = json.load(f)

            for obj in detected_objects:
                if len(obj.get("polygon", [])) >= 3:
                    p = Platform(
                        obj["polygon"],
                        obj.get("name", "obj"),
                        self.original_w,
                        self.original_h,
                    )
                    self.platforms.add(p)
        except FileNotFoundError:
            print(
                f"Warning: {json_path} not found. Only ground platform loaded.")

    def handle_events(self):
        """Processes external events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_SPACE]:
                    self.player.jump()
                if event.key == pygame.K_r:
                    self.player.reset_position(*self.start_pos)

    def update(self):
        """Updates the game state for the current frame."""
        keys = pygame.key.get_pressed()
        dx = 0

        if keys[pygame.K_RIGHT]:
            dx = self.player_speed
        if keys[pygame.K_LEFT]:
            dx = -self.player_speed

        is_dancing = keys[pygame.K_d]
        is_speaking = keys[pygame.K_s]

        if is_dancing:
            dx = 0

        self.player.update(dx, is_dancing, is_speaking, self.platforms)

    def draw(self):
        """
        Renders all game objects and applies the spotlight effect.
        Order: Background -> Platforms -> Player -> Spotlight -> UI.
        """

        # --- 1. NORMAL GAME DISPLAY (WITH BACKGROUND) ---
        self.screen.blit(self.bg_img, (0, 0))

        for p in self.platforms:
            p.draw(self.screen)
        self.player.draw(self.screen)

        # ---- SPOTLIGHT EFFECT (ONLY ON DISPLAY) ----
        self.darkness_layer.fill((0, 0, 0, self.darkness_opacity))

        light_x = self.player.rect.centerx - self.spotlight_radius
        light_y = self.player.rect.centery - self.spotlight_radius

        self.darkness_layer.blit(self.spotlight_img, (light_x, light_y),
                                special_flags=pygame.BLEND_RGBA_SUB)

        self.screen.blit(self.darkness_layer, (0, 0))

        # --------------------------------------------------------
        # --- 2. RECORDING MODE ---
        # --------------------------------------------------------
        self.record_surface.fill((255, 255, 255))

        self.player.draw(self.record_surface)

        self.record_surface.blit(self.darkness_layer, (0, 0))
        self.recorder.write(self.record_surface)

        # --- FINAL DISPLAY UPDATE ---
        pygame.display.update()


    def run(self):
        """The main game loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()
        self.recorder.close()
        sys.exit()
