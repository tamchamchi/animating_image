import pygame

from .config import ANIMATION_SPEEDS, CFG, COLLISION_TOLERANCE, GROUND_LEVEL, SCREEN_W

# Extract physics constants for easier access
GRAVITY = CFG["physics"]["gravity"]
JUMP_FORCE = CFG["physics"]["jump_force"]


class Player(pygame.sprite.Sprite):
    """
    Represents the main playable character in the game.

    This class handles:
    1. Movement physics (gravity, jumping, running).
    2. Pixel-perfect collision detection with platforms.
    3. Animation state management (idle, run, jump, etc.).
    4. Rendering the player and speech bubbles.
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

    def check_mask_collision(self, platforms):
        """
        Checks for pixel-perfect collisions with a list of platforms.

        Args:
            platforms (list): A list of Platform objects.

        Returns:
            bool: True if a collision overlaps, False otherwise.
        """
        for plat in platforms:
            # First, check simple AABB collision (Rect vs Rect) for performance
            if self.rect.colliderect(plat.rect):
                # Calculate the relative offset between player and platform
                offset_x = self.rect.x - plat.rect.x + COLLISION_TOLERANCE
                offset_y = self.rect.y - plat.rect.y + COLLISION_TOLERANCE

                # Create a temporary mask for the player (consider optimizing this)
                player_mask = pygame.Mask((self.rect.width, self.rect.height))
                player_mask.fill()

                # Check if the bits of the masks overlap
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
            # Cycle through dots " . " -> " .. " -> " ..."
            self.speech_text = "." * ((self.dot_timer // 15) % 3 + 1)

        # --- X MOVEMENT & COLLISION ---
        self.rect.x += dx
        # If moving into a wall, revert the position
        if self.check_mask_collision(platforms):
            self.rect.x -= dx

        # --- BOUNDARY CHECKS ---
        # Clamp player within screen width
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_W:
            self.rect.right = SCREEN_W

        # Update facing direction based on movement
        if dx > 0:
            self.facing_right = True
        elif dx < 0:
            self.facing_right = False

        # --- Y MOVEMENT & COLLISION ---
        self.vel_y += GRAVITY
        self.rect.y += self.vel_y
        self.on_ground = False

        # Resolve vertical collisions
        if self.check_mask_collision(platforms):
            if self.vel_y > 0:  # Falling down onto a platform
                # Move up pixel by pixel until no longer colliding
                while self.check_mask_collision(platforms):
                    self.rect.y -= 1
                self.vel_y = 0
                self.on_ground = True
                self.is_jumping = False
            elif self.vel_y < 0:  # Jumping up into a platform (Bonk head)
                # Move down pixel by pixel until no longer colliding
                while self.check_mask_collision(platforms):
                    self.rect.y += 1
                self.vel_y = 0

        # Floor check (Global ground level)
        if self.rect.bottom >= GROUND_LEVEL:
            self.rect.bottom = GROUND_LEVEL
            self.vel_y = 0
            self.on_ground = True
            self.is_jumping = False

        # --- ANIMATION STATE LOGIC ---
        # Determine the current state based on priority
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

        Resets the frame index if the state changes to ensure smooth transitions.
        """
        if new_state not in self.animations:
            new_state = "idle"

        if self.state != new_state:
            self.state = new_state
            self.frame_idx = 0  # Reset animation to first frame

        self.frame_speed = speed

    def draw(self, surface):
        """
        Renders the player sprite and optional UI elements (speech bubble).
        """
        # Draw Speech Bubble
        if self.is_speaking:
            bx, by = self.rect.topright
            # Draw bubble background
            pygame.draw.ellipse(surface, (255, 255, 255), (bx - 5, by - 45, 80, 40))
            # Draw bubble outline
            pygame.draw.ellipse(surface, (0, 0, 0), (bx - 5, by - 45, 80, 40), 2)
            # Render text
            ts = self.font.render(self.speech_text, True, (0, 0, 0))
            surface.blit(ts, ts.get_rect(center=(bx + 35, by - 25)))

        # Draw Player Sprite
        frames = self.animations.get(self.state)
        if frames:
            img = frames[int(self.frame_idx)]
            # Flip image if facing left
            if not self.facing_right:
                img = pygame.transform.flip(img, True, False)
            surface.blit(img, img.get_rect(midbottom=self.rect.midbottom))

    def jump(self):
        """Initiates a jump if the player is currently on the ground."""
        if self.on_ground:
            self.vel_y = JUMP_FORCE
            self.is_jumping = True
            self.on_ground = False

    def reset_position(self, x, y):
        """Teleports the player to a specific coordinate."""
        self.rect.midbottom = (x, y)
        self.vel_y = 0
        self.is_jumping = False
        self.on_ground = True
