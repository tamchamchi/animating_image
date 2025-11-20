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
