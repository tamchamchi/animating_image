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
    Build an MVC configuration file from a template.

    This function loads a template YAML file located at:
        MVC_ROOT/{action}.yaml

    It replaces every occurrence of "{CHAR_PATH}" in the template with the
    provided character directory, then saves the processed file as:
        MVC_ROOT/mvc_{char_name}_{action}.yaml

    Args:
        action (str): Name of the action (e.g., "jumping", "walking").
        char_dir (str): Full path to the character directory.
        char_name (str): Character name (e.g., "char1", "char2").

    Returns:
        str: Path to the generated MVC config file.
    """

    template_path = MVC_ROOT / f"{action}.yaml"
    output_path = MVC_ROOT / f"mvc_{char_name}_{action}.yaml"

    # Normalize character directory path
    char_dir = str(Path(char_dir))

    # Check if the template file exists
    if not template_path.exists():
        raise FileNotFoundError(f"MVC template not found: {template_path}")

    # Load YAML template
    with open(template_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Recursive function that replaces "{CHAR_PATH}" in all strings
    def replace(obj):
        if isinstance(obj, dict):
            return {k: replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace(i) for i in obj]
        elif isinstance(obj, str):
            return obj.replace("{CHAR_PATH}", char_dir)
        return obj

    # Apply replacements
    cfg = replace(cfg)

    # Save the new MVC YAML config
    with open(output_path, "w") as f:
        yaml.dump(cfg, f)

    return str(output_path)