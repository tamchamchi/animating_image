from abc import ABC, abstractmethod
from PIL import Image

class IStyleTransfer(ABC):

    @abstractmethod
    def transfer(
        self,
        style_ref: Image.Image,
        image: Image.Image = None,
        prompt: str = None
    ) -> Image.Image:
        """Perform style transfer using style_ref + (image OR prompt)."""
        pass
