import pygame
import sys
import os
from PIL import Image

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
FPS = 30
CAPTION = "Run, Jump, Dance & Obstacles"

# Physics Constants
GRAVITY = 1.5
JUMP_FORCE = -22
GROUND_LEVEL = 500
PLAYER_SPEED = 8

# Colors
COLOR_BG_FALLBACK = (135, 206, 235)  # Sky Blue
COLOR_OBS_FALLBACK = (139, 69, 19)  # Saddle Brown
COLOR_ERROR_PLACEHOLDER = (255, 0, 0)  # Red

# Asset Configuration
# NOTE: Update these paths to match your local file structure
ASSET_SCALE = (256, 256)
BASE_CHAR_PATH = "/home/anhndt/animating_image/src/configs/characters/char13/"  # Example relative path
BG_IMAGE_PATH = "/home/anhndt/animating_image/src/configs/characters/char10/background_stylized.png"

# =============================================================================
# HELPER FUNCTIONS (ASSET LOADING)
# =============================================================================


def load_gif_frames(path, skip_frames=4, scale=(100, 100)):
    """
    Extracts frames from a GIF file using PIL and converts them to Pygame surfaces.

    Args:
        path (str): Path to the .gif file.
        skip_frames (int): Number of frames to skip to reduce animation speed/memory.
        scale (tuple): (width, height) to resize the image.

    Returns:
        list: A list of pygame.Surface objects.
    """
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}. Using placeholder.")
        placeholder = pygame.Surface(scale, pygame.SRCALPHA)
        pygame.draw.rect(placeholder, COLOR_ERROR_PLACEHOLDER,
                         (0, 0, scale[0], scale[1]))
        return [placeholder]

    frames = []
    try:
        pil_img = Image.open(path)
        # Iterate through GIF frames
        for i in range(0, pil_img.n_frames, skip_frames):
            pil_img.seek(i)
            # Convert to RGBA (keeps transparency)
            frame = pil_img.convert("RGBA").resize(scale)
            pygame_img = pygame.image.fromstring(
                frame.tobytes(), frame.size, frame.mode
            )
            frames.append(pygame_img)
        return frames
    except Exception as e:
        print(f"[ERROR] Loading GIF {path}: {e}")
        return [pygame.Surface(scale)]


def load_image(path, size=None):
    """
    Loads a static image. Returns a colored block if the file is missing.
    """
    if os.path.exists(path):
        img = pygame.image.load(path).convert_alpha()
        if size:
            img = pygame.transform.scale(img, size)
        return img
    else:
        print(f"[WARNING] Image not found: {path}. Using color block.")
        surf = pygame.Surface(size if size else (50, 50))
        surf.fill(COLOR_BG_FALLBACK)
        return surf

# =============================================================================
# CLASSES
# =============================================================================


class Obstacle:
    """Represents a static object in the game world."""

    def __init__(self, x, y, width, height, image=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.image = image
        if image:
            self.image = pygame.transform.scale(image, (width, height))

    def draw(self, surface):
        if self.image:
            surface.blit(self.image, (self.rect.x, self.rect.y))
        else:
            # Draw a solid color rect if no image is provided
            pygame.draw.rect(surface, COLOR_OBS_FALLBACK, self.rect)


class Player:
    """Represents the controllable character with physics and animations."""

    def __init__(self, x, y, animations):
        # Physical properties
        self.rect = pygame.Rect(x, y, 40, 100)  # Hitbox size
        self.rect.midbottom = (x, y)
        self.vel_y = 0
        self.is_jumping = False
        self.on_ground = False
        self.facing_right = True

        # Animation properties
        self.animations = animations
        self.state = "idle"
        self.frame_idx = 0
        self.frame_speed = 0.5  # How fast the animation plays

    def set_state(self, new_state):
        """Switches animation state only if it's different from the current one."""
        # Fallback: If the requested state (e.g., 'walk') doesn't exist, use 'run' or 'idle'
        if new_state not in self.animations:
            if "run" in self.animations:
                new_state = "run"
            else:
                new_state = "idle"

        if self.state != new_state:
            self.state = new_state
            self.frame_idx = 0

    def move_and_collide(self, dx, obstacles):
        """Handles horizontal movement and X-axis collisions."""
        self.rect.x += dx

        # Check collision with obstacles
        for obs in obstacles:
            if self.rect.colliderect(obs.rect):
                if dx > 0:  # Moving right; hit left side of obstacle
                    self.rect.right = obs.rect.left
                elif dx < 0:  # Moving left; hit right side of obstacle
                    self.rect.left = obs.rect.right

        # Update facing direction
        if dx > 0:
            self.facing_right = True
        elif dx < 0:
            self.facing_right = False

    def apply_gravity_and_collide(self, obstacles):
        """Handles vertical movement (gravity) and Y-axis collisions."""
        self.vel_y += GRAVITY
        self.rect.y += self.vel_y

        self.on_ground = False

        # Check collision with obstacles
        for obs in obstacles:
            if self.rect.colliderect(obs.rect):
                if self.vel_y > 0:  # Falling down; hit top of obstacle
                    self.rect.bottom = obs.rect.top
                    self.vel_y = 0
                    self.is_jumping = False
                    self.on_ground = True
                elif self.vel_y < 0:  # Jumping up; hit bottom of obstacle
                    self.rect.top = obs.rect.bottom
                    self.vel_y = 0

        # Check collision with the floor (Global Ground Level)
        if self.rect.bottom >= GROUND_LEVEL:
            self.rect.bottom = GROUND_LEVEL
            self.vel_y = 0
            self.is_jumping = False
            self.on_ground = True

    def jump(self):
        """Initiates a jump if the player is not already in the air."""
        if not self.is_jumping and self.on_ground:
            self.vel_y = JUMP_FORCE
            self.is_jumping = True
            self.on_ground = False

    def update(self, dx, is_dancing, obstacles):
        """Main update loop for the player."""
        # 1. Physics & Movement
        self.move_and_collide(dx, obstacles)
        self.apply_gravity_and_collide(obstacles)

        # 2. State Management & Animation Speed
        if self.is_jumping:
            self.set_state("jump")
            self.frame_speed = 0.5
        elif dx != 0:
            if abs(dx) > 5:
                self.set_state("run")
                self.frame_speed = 0.8
            else:
                # Optional: Add a 'walk' state if you have the asset
                self.set_state("walk")
                self.frame_speed = 0.5
        elif is_dancing and self.on_ground:
            self.set_state("dance")
            self.frame_speed = 0.4
        else:
            self.set_state("idle")
            self.frame_speed = 0.8

        # 3. Advance Animation Frame
        current_frames = self.animations.get(self.state)
        if current_frames:
            self.frame_idx += self.frame_speed
            if self.frame_idx >= len(current_frames):
                self.frame_idx = 0

    def draw(self, surface):
        """Draws the current frame to the screen."""
        current_frames = self.animations.get(self.state)
        if not current_frames:
            # Draw a red rectangle if animation is missing
            pygame.draw.rect(surface, (255, 0, 0), self.rect)
            return

        # Get the image for the current frame index
        frame_image = current_frames[int(self.frame_idx)]

        # Flip image if facing left
        if not self.facing_right:
            frame_image = pygame.transform.flip(frame_image, True, False)

        # Center the image rect on the physics rect
        # (Because the image might be 256x256 but the hitbox is smaller)
        img_rect = frame_image.get_rect()
        img_rect.midbottom = self.rect.midbottom

        surface.blit(frame_image, img_rect)

# =============================================================================
# MAIN GAME INIT
# =============================================================================


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(CAPTION)
    clock = pygame.time.Clock()

    # ---------------------------------------------------------
    # LOAD ASSETS
    # ---------------------------------------------------------
    # Note: Replace paths with your actual file locations
    animations = {
        "idle":  load_gif_frames(os.path.join(BASE_CHAR_PATH, "standing.gif"), skip_frames=5, scale=ASSET_SCALE),
        "run":   load_gif_frames(os.path.join(BASE_CHAR_PATH, "running.gif"), skip_frames=12, scale=ASSET_SCALE),
        "jump":  load_gif_frames(os.path.join(BASE_CHAR_PATH, "jumping.gif"), skip_frames=15, scale=ASSET_SCALE),
        "dance": load_gif_frames(os.path.join(BASE_CHAR_PATH, "jesse_dancing.gif"), skip_frames=10, scale=ASSET_SCALE),
        # "walk": load_gif_frames(...) # Uncomment if you have walking.gif
    }

    bg_image = load_image(BG_IMAGE_PATH, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Optional: Load an obstacle texture (e.g., box.png)
    # box_img = load_image("./assets/objects/box.png", (60, 60))

    # ---------------------------------------------------------
    # CREATE OBJECTS
    # ---------------------------------------------------------
    obstacles_list = [
        # Obstacle(x, y, width, height, image)
        Obstacle(550, 370, 100, 20),           # Floating Platform 1
        Obstacle(650, 250, 100, 20),           # Floating Platform 2
        Obstacle(250, 150, 100, 20),           # Floating Platform 3
    ]

    player = Player(100, GROUND_LEVEL, animations)

    # ---------------------------------------------------------
    # GAME LOOP
    # ---------------------------------------------------------
    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_SPACE:
                    player.jump()

        # 2. Input Handling (Continuous)
        keys = pygame.key.get_pressed()
        dx = 0
        is_dancing = False

        if keys[pygame.K_RIGHT]:
            dx = PLAYER_SPEED
        elif keys[pygame.K_LEFT]:
            dx = -PLAYER_SPEED

        # 'D' key triggers dance if standing still
        if keys[pygame.K_d] and dx == 0:
            is_dancing = True

        # 3. Update Game Logic
        player.update(dx, is_dancing, obstacles_list)

        # 4. Drawing
        # Draw Background
        screen.blit(bg_image, (0, 0))

        # Draw Obstacles
        for obs in obstacles_list:
            obs.draw(screen)

        # Draw Player
        player.draw(screen)

        # 5. Refresh Display
        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
