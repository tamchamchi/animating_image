import pygame
import sys
import os
from PIL import Image

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 1000
FPS = 60
CAPTION = "Lab Platformer - Speaking Action on 'S'"

# ORIGINAL IMAGE DIMENSIONS
# Used for scaling coordinates from the large background to the screen size
ORIGINAL_WIDTH = 2560
ORIGINAL_HEIGHT = 1920

# Physics Settings
GRAVITY = 0.8
JUMP_FORCE = -20
PLAYER_SPEED = 7
GROUND_LEVEL = SCREEN_HEIGHT - 20

# Asset Configuration
BG_IMAGE_PATH = "/home/anhndt/animating_image/src/configs/characters/char13/background.jpg"
ASSET_SCALE = (256, 256)
BASE_CHAR_PATH = "/home/anhndt/animating_image/src/configs/characters/char13/"

# =============================================================================
# HELPER FUNCTIONS & CLASSES
# =============================================================================


def get_scaled_rect(x, y, w, h):
    """
    Calculates the scaled position and size of a rectangle based on
    the ratio between the screen size and the original image size.
    """
    scale_x = SCREEN_WIDTH / ORIGINAL_WIDTH
    scale_y = SCREEN_HEIGHT / ORIGINAL_HEIGHT
    return pygame.Rect(x * scale_x, y * scale_y, w * scale_x, h * scale_y)


def load_gif_frames(path, skip_frames=4, scale=(100, 100)):
    """
    Loads individual frames from a GIF file using PIL (Pillow) and 
    converts them into Pygame surfaces.
    """
    if not os.path.exists(path):
        # Return a red placeholder square if file is missing
        s = pygame.Surface(scale, pygame.SRCALPHA)
        pygame.draw.rect(s, (255, 0, 0), (0, 0, scale[0], scale[1]))
        return [s]
    frames = []
    try:
        pil_img = Image.open(path)
        # Iterate through frames, skipping some to adjust animation speed
        for i in range(0, pil_img.n_frames, skip_frames):
            pil_img.seek(i)
            frame = pil_img.convert("RGBA").resize(scale)
            img = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
            frames.append(img)
        return frames
    except:  # noqa: E722
        return [pygame.Surface(scale)]


def load_image(path, size=None):
    """Loads a static image and scales it optionally."""
    if os.path.exists(path):
        img = pygame.image.load(path).convert()
        if size:
            img = pygame.transform.scale(img, size)
        return img
    return pygame.Surface(size if size else (50, 50))


class Platform:
    def __init__(self, raw_x, raw_y, raw_w, raw_h, name, visible=False):
        # Convert raw coordinates to scaled screen coordinates
        self.rect = get_scaled_rect(raw_x, raw_y, raw_w, raw_h)
        self.name = name
        self.visible = visible

    def draw(self, surface):
        if self.visible:
            pygame.draw.rect(surface, (255, 165, 0), self.rect, 2)


class Player:
    def __init__(self, x, y, animations):
        # Hitbox: 50x100 pixels
        self.rect = pygame.Rect(0, 0, 50, 100)
        self.rect.midbottom = (x, y)

        self.vel_y = 0
        self.is_jumping = False
        self.on_ground = False
        self.facing_right = True
        self.animations = animations
        self.state = "idle"
        self.frame_idx = 0
        self.frame_speed = 0.4

        # --- CALL OUT FEATURE VARIABLES ---
        # Variables to handle the speaking mechanic
        self.is_speaking = False
        self.speech_text = "."
        self.dot_timer = 0
        self.font = pygame.font.SysFont("Arial", 24, bold=True)

    def update(self, dx, is_dancing, is_speaking, platforms):
        self.is_speaking = is_speaking

        # --- LOGIC FOR "..." SPEECH BUBBLE EFFECT ---
        # Cycles through ".", "..", "..." to simulate waiting/talking
        if self.is_speaking:
            self.dot_timer += 1
            speed = 15
            stage = (self.dot_timer // speed) % 3
            self.speech_text = "." * (stage + 1)
        else:
            self.dot_timer = 0
            self.speech_text = "."

        # X Movement (Horizontal)
        self.rect.x += dx
        # Keep player within screen bounds
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

        # Horizontal Collision Detection
        for plat in platforms:
            if self.rect.colliderect(plat.rect):
                if dx > 0:
                    self.rect.right = plat.rect.left
                elif dx < 0:
                    self.rect.left = plat.rect.right

        # Determine facing direction
        if dx > 0:
            self.facing_right = True
        elif dx < 0:
            self.facing_right = False

        # Y Movement (Vertical/Gravity)
        self.vel_y += GRAVITY
        self.rect.y += self.vel_y
        self.on_ground = False

        # Vertical Collision Detection
        for plat in platforms:
            if self.rect.colliderect(plat.rect):
                if self.vel_y > 0:  # Falling down
                    self.rect.bottom = plat.rect.top
                    self.vel_y = 0
                    self.on_ground = True
                    self.is_jumping = False
                elif self.vel_y < 0:  # Jumping up hitting head
                    self.rect.top = plat.rect.bottom
                    self.vel_y = 0

        # Floor Collision
        if self.rect.bottom >= GROUND_LEVEL:
            self.rect.bottom = GROUND_LEVEL
            self.vel_y = 0
            self.on_ground = True
            self.is_jumping = False

        # --- ANIMATION STATE LOGIC (Updated) ---
        # Priority order: Jump > Run > Speak > Dance > Idle
        if self.is_jumping:
            self.set_state("jump", 0.3)
        elif dx != 0:
            self.set_state("run", 0.3)
        elif is_speaking and self.on_ground:  # <--- ASSIGN SPEAK ACTION HERE
            self.set_state("speak", 0.2)
        elif is_dancing and self.on_ground:
            self.set_state("dance", 0.3)
        else:
            self.set_state("idle", 0.2)

        # Update animation frame
        frames = self.animations.get(self.state)
        if frames:
            self.frame_idx = (self.frame_idx + self.frame_speed) % len(frames)

    def set_state(self, new_state, speed):
        """Switches animation state only if it changes, resetting the frame index."""
        if new_state not in self.animations:
            new_state = "idle"

        if self.state != new_state:
            self.state = new_state
            self.frame_idx = 0
        self.frame_speed = speed

    def draw_speech_bubble(self, surface):
        """Draws a comic-style speech bubble above the player's head."""
        anchor_x, anchor_y = self.rect.topright
        bubble_w = 80
        bubble_h = 40
        bubble_x = anchor_x - 5
        bubble_y = anchor_y - bubble_h - 5

        bubble_rect = pygame.Rect(bubble_x, bubble_y, bubble_w, bubble_h)
        
        # Draw bubble body (white fill, black outline)
        pygame.draw.ellipse(surface, (255, 255, 255), bubble_rect)
        pygame.draw.ellipse(surface, (0, 0, 0), bubble_rect, 2)

        # Draw bubble tail/pointer
        p1 = (bubble_x + 20, bubble_y + bubble_h - 3)
        p2 = (bubble_x + 30, bubble_y + bubble_h - 2)
        p3 = (anchor_x, anchor_y)

        pygame.draw.polygon(surface, (255, 255, 255), [p1, p2, p3])
        pygame.draw.line(surface, (0, 0, 0), p1, p3, 2)
        pygame.draw.line(surface, (0, 0, 0), p2, p3, 2)

        # Draw text inside bubble
        text_surf = self.font.render(self.speech_text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(
            center=(bubble_rect.centerx, bubble_rect.centery - 2)
        )
        surface.blit(text_surf, text_rect)

    def draw(self, surface):
        """Renders the player sprite and speech bubble."""
        frames = self.animations.get(self.state)
        if frames:
            img = frames[int(self.frame_idx)]
            if not self.facing_right:
                img = pygame.transform.flip(img, True, False)
            r = img.get_rect(midbottom=self.rect.midbottom)
            surface.blit(img, r)

        if self.is_speaking:
            self.draw_speech_bubble(surface)

    def jump(self):
        if self.on_ground:
            self.vel_y = JUMP_FORCE
            self.is_jumping = True
            self.on_ground = False

    def reset_position(self, x, y):
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

    # Load Assets and animations
    # Note: 'waving' is used as idle, 'speaking' is the new action
    animations = {
        "idle": load_gif_frames(
            os.path.join(BASE_CHAR_PATH, "waving.gif"), 1, ASSET_SCALE
        ),
        "run": load_gif_frames(
            os.path.join(BASE_CHAR_PATH, "running.gif"), 1, ASSET_SCALE
        ),
        "jump": load_gif_frames(
            os.path.join(BASE_CHAR_PATH, "jumping.gif"), 1, ASSET_SCALE
        ),
        "dance": load_gif_frames(
            os.path.join(BASE_CHAR_PATH, "jesse_dancing.gif"), 1, ASSET_SCALE
        ),
        "speak": load_gif_frames(
            os.path.join(BASE_CHAR_PATH, "speaking.gif"), 1, ASSET_SCALE
        ),
    }
    bg_img = load_image(BG_IMAGE_PATH, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Define Platforms (Frames, Power sockets, Steps)
    platforms = [
        Platform(662, 581, 310, 382, "Frame", visible=False),
        Platform(1291, 945, 452, 145, "Socket", visible=False),
        Platform(200, 1500, 400, 50, "Low Platform", visible=True),
        Platform(800, 1200, 300, 50, "Mid Platform", visible=True),
        Platform(450, 800, 250, 50, "High Platform", visible=True),
    ]

    start_platform = platforms[0]
    start_x = start_platform.rect.centerx
    start_y = start_platform.rect.top

    player = Player(start_x, start_y, animations)

    running = True
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_SPACE]:
                    player.jump()

                if event.key == pygame.K_r:
                    player.reset_position(start_x, start_y)

        # Input State Handling
        keys = pygame.key.get_pressed()
        dx = 0
        is_dancing = False
        is_speaking = False

        if keys[pygame.K_RIGHT]:
            dx = PLAYER_SPEED
        if keys[pygame.K_LEFT]:
            dx = -PLAYER_SPEED
        
        # 'D' to dance (only if not moving)
        if keys[pygame.K_d] and dx == 0:
            is_dancing = True

        # Hold 'S' to speak
        if keys[pygame.K_s]:
            is_speaking = True

        # Update Game State
        player.update(dx, is_dancing, is_speaking, platforms)

        # Draw Everything
        screen.blit(bg_img, (0, 0))

        for p in platforms:
            p.draw(screen)

        player.draw(screen)

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()