import pygame
import sys
from PIL import Image
import os

# ============================
# INIT PYGAME
# ============================
pygame.init()
WIDTH, HEIGHT = 800, 600
FPS = 20
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Optimized GIF Animation Game")
clock = pygame.time.Clock()

# ============================
# FUNCTION: Load GIF (Optimized)
# ============================
def load_gif_frames(path, skip=4, scale=(100, 100)):
    """
    Load GIF as a list of Pygame surfaces
    skip: number of frames to skip (reduce memory)
    scale: (width, height) to resize each frame
    """
    if not os.path.exists(path):
        print(f"WARNING: File not found: {path}")
        placeholder = pygame.Surface((50,50))
        placeholder.fill((255,0,0))
        return [placeholder]
    
    frames = []
    try:
        pil_img = Image.open(path)
        for i in range(0, pil_img.n_frames, skip):
            pil_img.seek(i)
            frame = pil_img.convert("RGBA").resize(scale)
            pygame_img = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
            frames.append(pygame_img)
        print(f"Loaded {len(frames)} frames from {path}")
        return frames
    except Exception as e:
        print(f"ERROR loading GIF {path}: {e}")
        placeholder = pygame.Surface((50,50))
        placeholder.fill((255,0,0))
        return [placeholder]

# ============================
# PLAYER CLASS
# ============================
class Player:
    def __init__(self, x, y, animations):
        self.x = x
        self.y = y
        self.animations = animations
        self.state = "idle"
        self.frame_idx = 0
        self.frame_speed = 0.15

    def set_state(self, state):
        if self.state != state:
            self.state = state
            self.frame_idx = 0

    def update(self):
        frames = self.animations.get(self.state, [pygame.Surface((50,50))])
        self.frame_idx += self.frame_speed
        if self.frame_idx >= len(frames):
            self.frame_idx = 0

    def draw(self, surface):
        frames = self.animations.get(self.state, [pygame.Surface((50,50))])
        frame = frames[int(self.frame_idx)]
        surface.blit(frame, (self.x, self.y))

# ============================
# LOAD ANIMATIONS
# ============================
animations = {
    "idle": load_gif_frames(""),
    "run": load_gif_frames("/home/anhndt/animating_image/src/configs/characters/char4/running.gif", skip=4, scale=(100,100)),
    "walk": load_gif_frames("/home/anhndt/animating_image/src/configs/characters/char4/walking.gif", skip=4, scale=(100,100)),
    "jump": load_gif_frames("/home/anhndt/animating_image/src/configs/characters/char4/jumping.gif", skip=4, scale=(100,100))
}

player = Player(300, 300, animations)

# ============================
# MAIN LOOP
# ============================
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Keyboard control
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        player.set_state("run")
        player.x += 5
    elif keys[pygame.K_LEFT]:
        player.set_state("walk")
        player.x -= 3
    elif keys[pygame.K_SPACE]:
        player.set_state("jump")
    else:
        player.set_state("idle")

    player.update()
    SCREEN.fill((30,30,30))
    player.draw(SCREEN)
    pygame.display.update()
    clock.tick(FPS)
