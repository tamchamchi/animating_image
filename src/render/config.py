"""
Game Configuration Module.

This script is responsible for loading game settings from an external YAML file
('game_cfg.yaml') and defining global constants used throughout the game, 
such as screen dimensions, physics parameters, styling, and player initialization.
"""

import yaml
import sys
import os
from dotenv import load_dotenv

load_dotenv()

GAME_CONFIG_PATH = os.getenv("GAME_CONFIG_PATH")

# Attempt to load the configuration file.
# If the file is missing, the program will terminate to prevent runtime errors.
try:
    with open(GAME_CONFIG_PATH, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: game_cfg.yaml not found!")
    sys.exit(1)

# --- SCREEN SETTINGS ---
# Define the dimensions of the game window.
SCREEN_W = CFG['screen']['width']
SCREEN_H = CFG['screen']['height']

# --- PHYSICS & ENVIRONMENT ---
# Define the Y-coordinate representing the ground floor.
GROUND_LEVEL = SCREEN_H - 20

# The margin of error (in pixels) for collision detection calculation.
COLLISION_TOLERANCE = CFG['physics'].get('collision_tolerance', 5)

# Dictionary defining the playback speed for various animation states.
ANIMATION_SPEEDS = CFG.get('animation_speeds', {
    "idle": 0.2, "run": 0.3, "jump": 0.3, "dance": 0.2, "speak": 0.2
})

# --- PLATFORM CONSTANTS ---
# Extract platform styling configurations.
PLAT_STYLE = CFG.get('platform_style', {})

# Padding for the surface to ensure borders are not clipped during rendering.
PLATFORM_PADDING = PLAT_STYLE.get('padding', 2)

# Visual Settings: Border color (RGBA) and thickness.
# Default to a semi-transparent white if not specified.
PLATFORM_BORDER_COLOR = tuple(PLAT_STYLE.get(
    'border_color', [255, 255, 255, 80]))
PLATFORM_BORDER_WIDTH = PLAT_STYLE.get('border_width', 2)

# Mask Settings: Color used specifically for creating the collision mask.
# This color is typically not rendered but used for logic.
PLATFORM_MASK_COLOR = tuple(PLAT_STYLE.get('mask_color', [255, 0, 0, 255]))

# --- PLAYER INITIALIZATION SETTINGS ---
PLAYER_CFG = CFG.get('player', {})

# Starting X coordinate for the player.
# Defaults to 400 if the key is missing in the YAML config.
PLAYER_START_X = PLAYER_CFG.get('start_x', 400)

# Starting Y coordinate for the player.
PLAYER_START_Y = PLAYER_CFG.get('start_y', 300)
