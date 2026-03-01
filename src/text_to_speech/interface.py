from abc import ABC, abstractmethod
from pathlib import Path


class ITextToSpeech(ABC):
    """
    Abstract base class (interface) for Text-to-Speech models.

    This class defines the contract that any concrete TTS 
    implementation must follow.
    """

    @abstractmethod
    def convert(self, prompt: str, **kwargs) -> Path:
        """
        Converts input text into speech and saves it as a .wav file.

        Args:
            prompt (str): Input text to synthesize into speech.
            **kwargs: Optional parameters (e.g., voice, speed, sample_rate, output_path).

        Returns:
            Path: Path to the generated .wav audio file.
        """
        pass
