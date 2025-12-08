from animated_drawings import render

from src.utils.config import build_mcv_config

from .interface import IAnimator


class MetaAnimator(IAnimator):
    """High-level animator that delegates to the animated_drawings renderer.

    MetaAnimator builds a runtime config for a requested action and character,
    then invokes the renderer. This class acts as a small orchestration layer
    that converts caller parameters into the concrete config used by the
    rendering backend.
    """

    def __init__(self):
        super().__init__()

    def animate(self, action: str, char_path: str, char_name: str):
        """Generate config and start rendering for the given action/character.

        Args:
            action (str): Action name or identifier to animate (e.g. "walk").
            char_path (str): Path to the character assets or character config.
            char_name (str): Logical character name used in the config.

        Returns:
            str: Path to the generated mcv config file used to run the renderer.
        """
        mcv_cfg_path = build_mcv_config(action, char_path, char_name)
        print("[MetaAnimator] Running animation:", mcv_cfg_path)
        render.start(mcv_cfg_path)
        return mcv_cfg_path
