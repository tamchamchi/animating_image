from abc import ABC, abstractmethod
from PIL import Image
from typing import List


class IConverter(ABC):
    @abstractmethod
    def convert(self, images: List[Image.Image], limit: int):
        pass