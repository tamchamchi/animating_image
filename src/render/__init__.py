"""
Game Engine Package Initialization.

This module exposes the main Game class and configuration settings
to allow for cleaner imports from the main entry point of the application.
"""

# Import the main Game class so it can be accessed directly from the package
from .game import Game

# Optionally expose the configuration dictionary if needed externally
from .config import CFG

# Define the list of public objects exported by this package
# This controls what is imported when using: "from package import *"
__all__ = ["Game", "CFG"]
