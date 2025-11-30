import pygame
import sys
import os
from PIL import Image

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 960
FPS = 60
CAPTION = "Lab Platformer"

# ORIGINAL IMAGE DIMENSIONS (Used for scaling calculations)
ORIGINAL_WIDTH = 2560
ORIGINAL_HEIGHT = 1920

# PHYSICS CONSTANTS
GRAVITY = 0.8
JUMP_FORCE = -20
PLAYER_SPEED = 7
GROUND_LEVEL = SCREEN_HEIGHT - 20

# ASSET CONFIGURATION
# NOTE: Ensure the path points to your actual background image
BG_IMAGE_PATH = (
    "/home/anhndt/animating_image/src/configs/characters/char13/background.jpg"
)
ASSET_SCALE = (256, 256)
BASE_CHAR_PATH = "/home/anhndt/animating_image/src/configs/characters/char13/"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_scaled_rect(x, y, w, h):
    """
    Converts coordinates from the original high-res image (2560x1920)
    to the current screen resolution (1280x960).
    """
    scale_x = SCREEN_WIDTH / ORIGINAL_WIDTH
    scale_y = SCREEN_HEIGHT / ORIGINAL_HEIGHT
    return pygame.Rect(x * scale_x, y * scale_y, w * scale_x, h * scale_y)


def load_gif_frames(path, skip_frames=4, scale=(100, 100)):
    """
    Extracts frames from a GIF file and converts them to Pygame surfaces.
    """
    if not os.path.exists(path):
        # Return a red placeholder if file not found
        s = pygame.Surface(scale, pygame.SRCALPHA)
        pygame.draw.rect(s, (255, 0, 0), (0, 0, scale[0], scale[1]))
        return [s]
    frames = []
    try:
        pil_img = Image.open(path)
        for i in range(0, pil_img.n_frames, skip_frames):
            pil_img.seek(i)
            frame = pil_img.convert("RGBA").resize(scale)
            img = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
            frames.append(img)
        return frames
    except:  # noqa: E722
        return [pygame.Surface(scale)]


def load_image(path, size=None):
    """
    Loads a static image from disk.
    """
    if os.path.exists(path):
        img = pygame.image.load(path).convert()
        if size:
            img = pygame.transform.scale(img, size)
        return img
    return pygame.Surface(size if size else (50, 50))


# =============================================================================
# CLASSES
# =============================================================================


class Platform:
    def __init__(self, raw_x, raw_y, raw_w, raw_h, name, visible=False):
        # Calculate scale based on original image dimensions
        self.rect = get_scaled_rect(raw_x, raw_y, raw_w, raw_h)
        self.name = name
        self.visible = visible  # Default is hidden (invisible walls)

    def draw(self, surface):
        # Only draw the outline if visible is True (Debug mode)
        if self.visible:
            pygame.draw.rect(surface, (255, 165, 0), self.rect, 2)


class Player:
    def __init__(self, x, y, animations):
        # Create a hitbox (50x100 pixels)
        self.rect = pygame.Rect(0, 0, 50, 100)
        # Position the bottom center of the player at (x, y)
        self.rect.midbottom = (x, y)

        self.vel_y = 0
        self.is_jumping = False
        self.on_ground = False
        self.facing_right = True
        self.animations = animations
        self.state = "idle"
        self.frame_idx = 0
        self.frame_speed = 0.4

    def update(self, dx, is_dancing, platforms):
        # --- X AXIS MOVEMENT ---
        self.rect.x += dx

        # Screen boundary checks
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

        # Horizontal collision with platforms
        for plat in platforms:
            if self.rect.colliderect(plat.rect):
                if dx > 0:
                    self.rect.right = plat.rect.left
                elif dx < 0:
                    self.rect.left = plat.rect.right

        # Update facing direction
        if dx > 0:
            self.facing_right = True
        elif dx < 0:
            self.facing_right = False

        # --- Y AXIS MOVEMENT ---
        self.vel_y += GRAVITY
        self.rect.y += self.vel_y
        self.on_ground = False

        # Vertical collision with platforms
        for plat in platforms:
            if self.rect.colliderect(plat.rect):
                if self.vel_y > 0:  # Falling down onto a platform
                    self.rect.bottom = plat.rect.top
                    self.vel_y = 0
                    self.on_ground = True
                    self.is_jumping = False
                elif self.vel_y < 0:  # Jumping up into a platform (head bump)
                    self.rect.top = plat.rect.bottom
                    self.vel_y = 0

        # Floor collision (Ground Level)
        if self.rect.bottom >= GROUND_LEVEL:
            self.rect.bottom = GROUND_LEVEL
            self.vel_y = 0
            self.on_ground = True
            self.is_jumping = False

        # --- ANIMATION STATE ---
        if self.is_jumping:
            self.set_state("jump", 0.3)
        elif dx != 0:
            self.set_state("run", 0.6)
        elif is_dancing and self.on_ground:
            self.set_state("dance", 0.3)
        else:
            self.set_state("idle", 0.4)

        # Advance animation frame
        frames = self.animations.get(self.state)
        if frames:
            self.frame_idx = (self.frame_idx + self.frame_speed) % len(frames)

    def set_state(self, new_state, speed):
        """Helper to switch animation states cleanly"""
        if new_state not in self.animations:
            new_state = "idle"
        if self.state != new_state:
            self.state = new_state
            self.frame_idx = 0
        self.frame_speed = speed

    def draw(self, surface):
        frames = self.animations.get(self.state)
        if frames:
            img = frames[int(self.frame_idx)]
            if not self.facing_right:
                img = pygame.transform.flip(img, True, False)
            # Align the image to the hitbox
            r = img.get_rect(midbottom=self.rect.midbottom)
            surface.blit(img, r)

    def jump(self):
        if self.on_ground:
            self.vel_y = JUMP_FORCE
            self.is_jumping = True
            self.on_ground = False

    def reset_position(self, x, y):
        """Resets the player to specific coordinates"""
        self.rect.midbottom = (x, y)
        self.vel_y = 0
        self.is_jumping = False
        self.on_ground = True


# =============================================================================
# MAIN GAME
# =============================================================================


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(CAPTION)
    clock = pygame.time.Clock()

    # --- LOAD ASSETS ---
    animations = {
        "idle": load_gif_frames(
            os.path.join(BASE_CHAR_PATH, "standing.gif"), 5, ASSET_SCALE
        ),
        "run": load_gif_frames(
            os.path.join(BASE_CHAR_PATH, "running.gif"), 12, ASSET_SCALE
        ),
        "jump": load_gif_frames(
            os.path.join(BASE_CHAR_PATH, "jumping.gif"), 15, ASSET_SCALE
        ),
        "dance": load_gif_frames(
            os.path.join(BASE_CHAR_PATH, "jesse_dancing.gif"), 10, ASSET_SCALE
        ),
    }
    bg_img = load_image(BG_IMAGE_PATH, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # --- INITIALIZE PLATFORMS (Invisible: visible=False) ---
    platforms = [
        # Box 1: Picture Frame / Rules Board (Starting Point)
        Platform(662, 561, 304, 382, "Khung Tranh", visible=False),
        # Box 2: Electric Socket
        Platform(1291, 945, 452, 145, "Ổ Điện", visible=False),
        # Support Platforms (Invisible steps to help climbing)
        Platform(200, 1500, 400, 50, "Bục Thấp", visible=False),
        Platform(800, 1200, 300, 50, "Bục Trung", visible=False),
        Platform(450, 800, 250, 50, "Bục Cao", visible=False),
    ]

    # --- CALCULATE STARTING POSITION ---
    # Get the target object (The Picture Frame - first item in list)
    start_platform = platforms[0]

    # Calculate the center-top coordinate of the platform
    start_x = start_platform.rect.centerx
    start_y = start_platform.rect.top

    # Initialize player at that position
    player = Player(start_x, start_y, animations)

    # --- GAME LOOP ---
    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                # Jump inputs
                if event.key in [pygame.K_UP, pygame.K_SPACE]:
                    player.jump()

                # 'R' Key: Reset to initial position (On the rules board)
                if event.key == pygame.K_r:
                    print("Resetting position...")
                    player.reset_position(start_x, start_y)

        # 2. Movement Inputs
        keys = pygame.key.get_pressed()
        dx = 0
        is_dancing = False
        if keys[pygame.K_RIGHT]:
            dx = PLAYER_SPEED
        if keys[pygame.K_LEFT]:
            dx = -PLAYER_SPEED
        if keys[pygame.K_d] and dx == 0:
            is_dancing = True

        # 3. Update Logic
        player.update(dx, is_dancing, platforms)

        # 4. Drawing
        screen.blit(bg_img, (0, 0))

        for p in platforms:
            p.draw(screen)  # Won't draw anything because visible=False

        player.draw(screen)

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
