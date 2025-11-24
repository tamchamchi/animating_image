from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
from pathlib import Path


@dataclass
class AnimationPipelineInput:
    style_ref_path: Path                  # đường dẫn ảnh style reference
    content_image_path: Optional[Path]    # đường dẫn ảnh content
    system_prompt: Optional[str]          # prompt text
    char_folder: Path                      # folder chứa character assets
    char_name: str
    actions: List[str]

    def load_data(self):
        """Load ảnh style_ref, content_image và các resource khác từ char_folder."""
        char_path = self.char_folder / self.char_name
        char_path.mkdir(parents=True, exist_ok=True)

        # Load style reference image
        style_ref = Image.open(self.style_ref_path).convert("RGB")

        # Load content image nếu có
        content_image = (
            Image.open(self.content_image_path).convert("RGB")
            if self.content_image_path is not None else None
        )

        return {
            "style_ref": style_ref,
            "content_image": content_image,
            "prompt": self.system_prompt,
            "char_folder": char_path,
            "char_name": self.char_name,
            "actions": self.actions
        }
