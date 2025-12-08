import pygame

from .config import ANIMATION_SPEEDS, CFG, COLLISION_TOLERANCE, GROUND_LEVEL, SCREEN_W

# Extract physics constants for easier access
GRAVITY = CFG["physics"]["gravity"]
JUMP_FORCE = CFG["physics"]["jump_force"]


class Player(pygame.sprite.Sprite):
    """
    Represents the main playable character in the game.

    This class handles:
    1. Physics (gravity, jumping, movement).
    2. Pixel-perfect collision detection.
    3. Animation state management.
    4. Rendering (including a silhouette shadow effect).
    """

    def __init__(self, x, y, animations):
        """
        Initialize the Player sprite.

        Args:
            x (int): Initial X coordinate (center-bottom).
            y (int): Initial Y coordinate (center-bottom).
            animations (dict): A dictionary containing lists of loaded images for each state.
        """
        super().__init__()
        # Define the hit box (AABB) of the player
        self.rect = pygame.Rect(0, 0, 30, 100)
        self.rect.midbottom = (x, y)

        # Physics properties
        self.vel_y = 0
        self.is_jumping = False
        self.on_ground = False
        self.facing_right = True

        # Animation & State properties
        self.animations = animations
        self.state = "idle"
        self.frame_idx = 0
        self.frame_speed = 0.4

        # Interaction properties (Speech bubble)
        self.is_speaking = False
        self.speech_text = "."
        self.dot_timer = 0
        self.font = pygame.font.SysFont("Arial", 24, bold=True)

        # --- SHADOW SETTINGS ---
        # Shadow offset (x, y).
        # (-5, 0) shifts the shadow 5px to the left, implying light comes from the right.
        # Use (0, 0) for a shadow directly behind the character (flash style).
        self.shadow_offset = (-5, 0)

        # Shadow opacity (0-255). 128 is semi-transparent (~50%).
        self.shadow_opacity = 128

    def check_mask_collision(self, platforms):
        """
        Checks for pixel-perfect collisions with a list of platforms.

        Args:
            platforms (list): A list of Platform objects.

        Returns:
            bool: True if a pixel-perfect overlap is detected, False otherwise.
        """
        for plat in platforms:
            # Optimization: Check bounding box collision first
            if self.rect.colliderect(plat.rect):
                offset_x = self.rect.x - plat.rect.x + COLLISION_TOLERANCE
                offset_y = self.rect.y - plat.rect.y + COLLISION_TOLERANCE

                # Create a mask for the player's current rect
                player_mask = pygame.Mask((self.rect.width, self.rect.height))
                player_mask.fill()

                # Check for bitmask overlap
                if plat.mask.overlap(player_mask, (offset_x, offset_y)):
                    return True
        return False

    def update(self, dx, is_dancing, is_speaking, platforms):
        """
        Updates the player's position, physics, and animation state.

        Args:
            dx (int): Horizontal movement input (velocity).
            is_dancing (bool): Whether the dance key is pressed.
            is_speaking (bool): Whether the speak key is pressed.
            platforms (list): List of platforms for collision checking.
        """
        # --- SPEECH LOGIC ---
        self.is_speaking = is_speaking
        if self.is_speaking:
            self.dot_timer += 1
            # Cycle text: "." -> ".." -> "..."
            self.speech_text = "." * ((self.dot_timer // 15) % 3 + 1)

        # --- X MOVEMENT & COLLISION ---
        self.rect.x += dx
        # If colliding with a wall, revert position
        if self.check_mask_collision(platforms):
            self.rect.x -= dx

        # --- BOUNDARY CHECKS ---
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_W:
            self.rect.right = SCREEN_W

        # Update facing direction
        if dx > 0:
            self.facing_right = True
        elif dx < 0:
            self.facing_right = False

        # --- Y MOVEMENT & COLLISION ---
        self.vel_y += GRAVITY
        self.rect.y += self.vel_y
        self.on_ground = False

        if self.check_mask_collision(platforms):
            if self.vel_y > 0:  # Falling down
                # Move up pixel by pixel to align with platform top
                while self.check_mask_collision(platforms):
                    self.rect.y -= 1
                self.vel_y = 0
                self.on_ground = True
                self.is_jumping = False
            elif self.vel_y < 0:  # Jumping up (Bonk head)
                # Move down pixel by pixel
                while self.check_mask_collision(platforms):
                    self.rect.y += 1
                self.vel_y = 0

        # Global ground level check
        if self.rect.bottom >= GROUND_LEVEL:
            self.rect.bottom = GROUND_LEVEL
            self.vel_y = 0
            self.on_ground = True
            self.is_jumping = False

        # --- ANIMATION STATE LOGIC ---
        if self.is_jumping:
            self.set_state("jump", ANIMATION_SPEEDS["jump"])
        elif dx != 0:
            self.set_state("run", ANIMATION_SPEEDS["run"])
        elif is_speaking and self.on_ground:
            self.set_state("speak", ANIMATION_SPEEDS["speak"])
        elif is_dancing and self.on_ground:
            self.set_state("dance", ANIMATION_SPEEDS["dance"])
        else:
            self.set_state("idle", ANIMATION_SPEEDS["idle"])

        # Advance animation frame
        frames = self.animations.get(self.state)
        if frames:
            self.frame_idx = (self.frame_idx + self.frame_speed) % len(frames)

    def set_state(self, new_state, speed):
        """
        Transitions the player to a new animation state.
        Resets the frame index to 0 if the state changes.
        """
        if new_state not in self.animations:
            new_state = "idle"
        if self.state != new_state:
            self.state = new_state
            self.frame_idx = 0
        self.frame_speed = speed

    def draw(self, surface):
        """
        Renders the player sprite with a SILHOUETTE SHADOW and UI elements.
        """
        # Get current animation frame
        frames = self.animations.get(self.state)
        if frames:
            img = frames[int(self.frame_idx)]

            # Flip image if facing left
            if not self.facing_right:
                img = pygame.transform.flip(img, True, False)

            # --- RENDER SILHOUETTE SHADOW ---
            # 1. Create a copy of the current character sprite
            shadow = img.copy()

            # 2. Fill the copy with black using multiplication blending.
            # This turns all colored pixels black but preserves the Alpha channel (transparency).
            shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)

            # 3. Apply opacity (transparency) to the shadow
            shadow.set_alpha(self.shadow_opacity)

            # 4. Determine drawing coordinates
            # Since self.rect tracks midbottom, we calculate topleft for blitting
            img_rect = img.get_rect(midbottom=self.rect.midbottom)

            # 5. Draw the shadow first (behind the player) with the defined offset
            surface.blit(
                shadow,
                (
                    img_rect.x + self.shadow_offset[0],
                    img_rect.y + self.shadow_offset[1],
                ),
            )

            # --- RENDER PLAYER ---
            # Draw the actual player sprite on top of the shadow
            surface.blit(img, img_rect)

        # Draw Speech Bubble (if active)
        if self.is_speaking:
            bx, by = self.rect.topright
            # Bubble background
            pygame.draw.ellipse(surface, (255, 255, 255), (bx - 5, by - 45, 80, 40))
            # Bubble outline
            pygame.draw.ellipse(surface, (0, 0, 0), (bx - 5, by - 45, 80, 40), 2)
            # Bubble text
            ts = self.font.render(self.speech_text, True, (0, 0, 0))
            surface.blit(ts, ts.get_rect(center=(bx + 35, by - 25)))

    def jump(self):
        """Initiates a jump if the player is currently on the ground."""
        if self.on_ground:
            self.vel_y = JUMP_FORCE
            self.is_jumping = True
            self.on_ground = False

    def reset_position(self, x, y):
        """Teleports the player to the specified coordinates."""
        self.rect.midbottom = (x, y)
        self.vel_y = 0
        self.is_jumping = False
        self.on_ground = True
