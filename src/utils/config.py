from dotenv import load_dotenv
import os
import yaml
from pathlib import Path

load_dotenv()


def get_env(key: str, default=None):
    return os.getenv(key, default)


def parse_float_list(value: str):
    """Convert '0.1,1.3,2.7' → [0.1, 1.3, 2.7]"""
    return [float(x) for x in value.split(",")]


MVC_ROOT = Path(get_env("MVC_DIR"))


def build_mcv_config(action: str, char_dir: str, char_name: str) -> str:
    """
    Load MVC template:   MVC_ROOT/{action}.yaml
    Replace {CHAR_DIR} inside YAML
    Save new file:       MVC_ROOT/mvc_{char_name}_{action}.yaml

    Args:
        action (str): tên hành động (jumping, walking, ...)
        char_dir (str): full path tới thư mục character
        char_name (str): tên nhân vật (char1, char2, ...)

    Returns:
        str: đường dẫn file mvc_{char_name}_{action}.yaml
    """

    template_path = MVC_ROOT / f"{action}.yaml"
    output_path = MVC_ROOT / f"mvc_{char_name}_{action}.yaml"

    char_dir = str(Path(char_dir))

    if not template_path.exists():
        raise FileNotFoundError(f"MVC template not found: {template_path}")

    # Load YAML template
    with open(template_path, "r") as f:
        cfg = yaml.safe_load(f)

    def replace(obj):
        if isinstance(obj, dict):
            return {k: replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace(i) for i in obj]
        elif isinstance(obj, str):
            return obj.replace("{CHAR_PATH}", char_dir)
        return obj

    cfg = replace(cfg)

    # Save MVC file new
    with open(output_path, "w") as f:
        yaml.dump(cfg, f)

    return str(output_path)


PROMPT_SUBJECT_STYLE_TRANSFER = """
Use the uploaded image strictly as a reference for drawing style.
Create an image in a hand-drawn style of {subject}.
KEEP IMAGE STRUCTURE.
Both arms extended to the sides. Both legs standing straight.
"""

PROMPT_IMAGE_STYLE_TRANSFER = """
Use the uploaded image strictly as a reference for drawing style.
Recreate the input image in a hand-drawn style.
KEEP THE ORIGINAL POSE, STRUCTURE, AND COMPOSITION.
Ensure both arms remain extended to the sides as in the style reference imageinput.
"""

PROMPT_BG_STYLE_TRANSFER = """
Create a 2D side-scrolling game background designed for horizontal tiling.
The scene should be a stylized cartoon beach environment with clear, separate layers: sky, distant islands, ocean, and foreground sand.
Each layer must be seamless so they can repeat horizontally.
Use vibrant colors and soft gradients similar to a platformer game.
Do not include characters, obstacles, or UI.
Produce a wide panoramic image suitable for parallax scrolling in a 2D game."""

PROMPT_SUBJECT_GENERATION = """
{subject} with both feet pointing to the right, drawn in the style of a colored pencil sketch on white paper. 
Use a naive, childlike art style with visible pencil stroke textures. 
Simple facial features (dot eyes), cute proportions, and a hand-drawn, 
grainy aesthetic. Isolated on white background."""
