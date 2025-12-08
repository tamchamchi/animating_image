from abc import ABC, abstractmethod


class IAnimator(ABC):
    """Abstract interface for animator implementations.

    Implementations should provide a concrete `animate` method that triggers
    an animation workflow for a given character and action.

    Method contract:
    - animate(action, char_path, char_name) -> Any
      - action (str): identifier of the animation/action to run.
      - char_path (str): filesystem path to the character asset or config.
      - char_name (str): logical name of the character within the project.
      - return value depends on implementation (e.g., path to generated output
        or a config used to run the renderer).
    """

    @abstractmethod
    def animate(self, action: str, char_path: str, char_name: str):
        """Run animation for the specified character and action.

        Subclasses must implement this method to launch the animation process
        (rendering, scheduling, or returning a config).
        """
        pass
