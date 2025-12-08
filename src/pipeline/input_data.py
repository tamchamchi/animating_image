from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from PIL import Image
from pathlib import Path


@dataclass
class AnimationPipelineInput:
    """
    Data Transfer Object (DTO) for the Animation Pipeline.

    This class encapsulates all necessary inputs required to run the 
    animation generation process, including image references, prompts, 
    and output configuration.
    """

    # The reference image defining the artistic style (e.g., colored pencil, oil painting).
    style_ref: Image.Image

    # The source content image (the character to be animated). Optional if generating from scratch.
    content_image: Optional[Image.Image]

    # The text prompt to guide the generation or style transfer model.
    system_prompt: Optional[str]

    # The base directory where character assets are stored. Can be a string or a Path object.
    char_folder: Union[str, Path]

    # The specific name of the character (used for creating sub-directories).
    char_name: str

    # A list of specific actions/motions to generate (e.g., ["running", "jumping"]).
    actions: List[str]

    def __post_init__(self):
        """
        Post-initialization processing.
        
        Automatically converts 'char_folder' to a pathlib.Path object if 
        it was provided as a string. This ensures consistent path handling 
        methods (like .mkdir, / operator) can be used later.
        """
        if isinstance(self.char_folder, str):
            self.char_folder = Path(self.char_folder)

    def load_data(self) -> Dict[str, Any]:
        """
        Prepares the workspace and converts the input data into a dictionary.

        This method performs two main tasks:
        1. Creates the specific directory for the character if it doesn't exist.
        2. Returns a dictionary representation of the data, which is compatible 
           with pipelines expecting dictionary inputs.

        Returns:
            Dict[str, Any]: A dictionary containing all pipeline parameters, 
                            where 'char_folder' is updated to the specific character path.
        """
        # Construct the full path to the specific character's directory.
        # Since __post_init__ ensures self.char_folder is a Path, the '/' operator works safely.
        char_path = self.char_folder / self.char_name

        # Ensure the directory exists; create parents if needed, ignore if already exists.
        char_path.mkdir(parents=True, exist_ok=True)

        # Return the data as a dictionary for easy consumption by the pipeline.
        return {
            "style_ref": self.style_ref,
            "content_image": self.content_image,
            "prompt": self.system_prompt,
            "char_folder": char_path,  # Passing the specific sub-folder
            "char_name": self.char_name,
            "actions": self.actions
        }