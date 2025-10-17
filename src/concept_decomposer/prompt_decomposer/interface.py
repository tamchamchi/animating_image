from abc import ABC, abstractmethod

class IPromptDecomposer(ABC):
    @abstractmethod
    def decompose(prompt: str) -> list[str]:
        """
        Decompose the input prompt into a list of objects.

        Args:
            prompt (str): The input prompt provided by the user.

        Returns:
            list[str]: A list of objects extracted from the prompt.

        Example:
            Input: "A man walking on the beach with a dog"
            Output: ["man", "beach", "dog"]
        """
        pass
