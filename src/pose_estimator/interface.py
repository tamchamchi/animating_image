from abc import ABC, abstractmethod

class IPoseEstimator(ABC):
    """
    Interface for a pose estimator.

    This abstract base class defines the structure that any pose estimation
    implementation should follow. Implementing classes must provide a 
    `predict` method that takes an input image and returns pose predictions.
    """

    @abstractmethod
    def predict(self, image):
        """
        Perform pose estimation on a single image.

        Args:
            image (np.ndarray or str): Input image array (H, W, C) or a file path to the image.

        Returns:
            preds (list or dict): Pose predictions. The exact format depends on the implementation.
        """
        pass
