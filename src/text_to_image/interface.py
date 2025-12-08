from abc import ABC, abstractmethod
from PIL import Image

class IImgeGenerator(ABC):
    """
    Abstract base class (interface) for image generation models.

    This class defines the contract that any concrete image generator 
    implementation must follow.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Image.Image:
        """
        Generates an image based on the provided text prompt.

        Args:
            prompt (str): The text description or input prompt used to 
                          guide the image generation process.
            **kwargs: Arbitrary keyword arguments allowed for flexibility in 
                      concrete implementations (e.g., seed, inference steps, 
                      negative prompt, image size).

        Returns:
            Image.Image: The generated image as a PIL Image object.
        """
        pass