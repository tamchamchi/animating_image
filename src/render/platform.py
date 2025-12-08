import pygame

# Import configuration constants
from .config import (
    PLATFORM_BORDER_COLOR,
    PLATFORM_BORDER_WIDTH,
    PLATFORM_MASK_COLOR,
    PLATFORM_PADDING,
)
from .utils import get_scaled_points


class Platform(pygame.sprite.Sprite):
    """
    Represents a static platform in the game world defined by a polygonal shape.

    This class handles:
    1. Scaling raw polygon points to screen dimensions.
    2. Creating a visual representation (border).
    3. Generating a collision mask for precise physics interactions.
    """

    def __init__(self, raw_polygon, name, original_w, original_h):
        """
        Initialize the Platform sprite.

        Args:
            raw_polygon (list): A list of (x, y) tuples representing the normalized
                                or raw vertices of the polygon.
            name (str): An identifier for the platform.
            original_w (int): The original width reference for scaling.
            original_h (int): The original height reference for scaling.
        """
        super().__init__()
        self.name = name

        # Scale raw coordinates to actual screen dimensions
        self.global_points = get_scaled_points(raw_polygon, original_w, original_h)

        # Calculate the bounding box (AABB) of the polygon
        # This helps in determining the size of the surface needed.
        xs = [p[0] for p in self.global_points]
        ys = [p[1] for p in self.global_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width, height = max_x - min_x, max_y - min_y

        # Convert global coordinates to local coordinates relative to the surface's top-left (0,0)
        self.local_points = [(x - min_x, y - min_y) for x, y in self.global_points]

        # Calculate surface dimensions including padding to prevent border clipping
        surface_w = width + PLATFORM_PADDING
        surface_h = height + PLATFORM_PADDING

        # --- VISUAL LAYER (Visible Border) ---
        # Create a transparent surface for rendering the visual border
        self.image = pygame.Surface((surface_w, surface_h), pygame.SRCALPHA)

        # Draw the platform border using configured color and thickness
        pygame.draw.polygon(
            self.image, PLATFORM_BORDER_COLOR, self.local_points, PLATFORM_BORDER_WIDTH
        )

        # --- PHYSICS LAYER (Solid Mask) ---
        # Create a temporary surface to draw the filled shape for collision logic
        mask_surface = pygame.Surface((surface_w, surface_h), pygame.SRCALPHA)

        # Draw a filled polygon (width=0) using the mask color
        pygame.draw.polygon(mask_surface, PLATFORM_MASK_COLOR, self.local_points, 0)

        # Generate a bitmask from the solid surface for pixel-perfect collision detection
        self.mask = pygame.mask.from_surface(mask_surface)

        # Position the sprite's rect at the calculated top-left global coordinate
        self.rect = self.image.get_rect(topleft=(min_x, min_y))

    def draw(self, surface):
        """
        Renders the platform onto the target surface.

        Args:
            surface (pygame.Surface): The target surface to draw on (usually the screen).
        """
        surface.blit(self.image, self.rect)
