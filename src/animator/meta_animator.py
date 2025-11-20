from animated_drawings import render

from src.utils.config import build_mcv_config

from .interface import IAnimator


class MetaAnimator(IAnimator):
    def __init__(self):
        super().__init__()

    def animate(self, action: str, char_path: str, char_name: str):
        mcv_cfg_path = build_mcv_config(action, char_path, char_name)
        print("[MetaAnimator] Running animation:", mcv_cfg_path)
        render.start(mcv_cfg_path)
        return mcv_cfg_path