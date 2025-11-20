from abc import ABC, abstractmethod


class IAnimator(ABC):

    @abstractmethod
    def animate():
        pass
