import cv2
import numpy as np

# Interfaces
from src.animator import IAnimator
from src.concept_decomposer import IObjectDecomposer
from src.image_style_transfer import IStyleTransfer
from src.pose_estimator import IPoseEstimator

from .input_data import AnimationPipelineInput


class AnimationGenerationPipeline:
    """
    A full animation generation pipeline that performs:
    1. Style transfer (Optional)
    2. Object segmentation & texture extraction (Optional)
    3. Pose estimation (Optional)
    4. Animation rendering (Optional)

    Each step checks if the corresponding component is provided.
    If a component is None, that step is skipped, and the pipeline attempts
    to fallback to existing data on disk or previous pipeline outputs.
    """

    def __init__(
        self,
        style_transfer: IStyleTransfer = None,
        object_decomposer: IObjectDecomposer = None,
        pose_estimator: IPoseEstimator = None,
        animator: IAnimator = None,
    ):
        # Components are optional (can be None) to allow partial pipeline execution
        self.style_transfer = style_transfer
        self.object_decomposer = object_decomposer
        self.pose_estimator = pose_estimator
        self.animator = animator

    def _expand_bbox(self, bbox, pad=2):
        """
        Expand a bounding box by a given pixel padding.
        Args:
            bbox: (x1, y1, x2, y2)
            pad: number of pixels to extend on each side
        Returns:
            Expanded bounding box tuple
        """
        x1, y1, x2, y2 = bbox
        return (max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad)

    def run(self, input_data: AnimationPipelineInput):
        """
        Execute the pipeline steps if their components exist.

        Flow:
        1. Load Data.
        2. Style Transfer: Uses input image -> updates 'current_image_np'.
        3. Decomposition: Uses 'current_image_np' -> creates crops -> updates 'image_crop_for_pose'.
           (Fallback: Loads existing texture.png if decomposition is skipped).
        4. Pose Estimation: Uses 'image_crop_for_pose' -> creates config files.
        5. Animation: Uses config files -> generates GIFs/Videos.

        Returns:
            Dictionary containing results/paths from executed steps.
        """

        # Load all structured input data (images, paths, parameters)
        data = input_data.load_data()
        results = {}

        # Default image source is the original content image.
        # This will be overwritten if Style Transfer runs successfully.
        current_image_np = np.array(data["content_image"])

        # ------------------------------------------------------------
        # Step 1: Style Transfer
        # ------------------------------------------------------------
        if self.style_transfer:
            print(">>> [Pipeline] Running Style Transfer...")
            object_stylized = self.style_transfer.transfer(
                style_ref=data["style_ref"],
                image=data["content_image"],
                prompt=data["prompt"],
            )

            # Save results
            stylized_path = data["char_folder"] / "object_stylized.png"
            object_stylized.save(stylized_path)

            # Update the image used for the next step (Segmentation)
            current_image_np = np.array(object_stylized)

            results["object_stylized"] = object_stylized
            results["stylized_path"] = stylized_path
        else:
            print(
                ">>> [Pipeline] Skipping Style Transfer (Component not provided). Using original content image.")

        # ------------------------------------------------------------
        # Step 2: Segmentation + Crop
        # ------------------------------------------------------------
        # This variable holds the cropped character texture for Pose Estimation
        image_crop_for_pose = None
        texture_path = data["char_folder"] / "texture.png"
        mask_path = data["char_folder"] / "mask.png"

        if self.object_decomposer:
            print(">>> [Pipeline] Running Object Decomposition...")

            # Decompose the current image (either stylized or original)
            seg_result = self.object_decomposer.decompose(current_image_np)
            mask_image_viz = seg_result["mask_image_viz"]
            bbox = seg_result["bounding_box"]

            # Expand bounding box to ensure full coverage
            x1, y1, x2, y2 = self._expand_bbox(bbox, pad=4)

            # Crop texture and mask
            mask_crop = mask_image_viz[y1:y2, x1:x2]
            image_crop = current_image_np[y1:y2, x1:x2]

            # Save to disk
            mask_crop_cv2 = cv2.cvtColor(mask_crop, cv2.COLOR_RGB2BGR)
            image_crop_cv2 = cv2.cvtColor(image_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(mask_path), mask_crop_cv2)
            cv2.imwrite(str(texture_path), image_crop_cv2)

            results["segmentation"] = seg_result

            # Pass this cropped image to the next step
            image_crop_for_pose = image_crop
        else:
            print(
                ">>> [Pipeline] Skipping Object Decomposition (Component not provided).")

            # Fallback: Attempt to load an existing texture from disk
            if texture_path.exists():
                print(
                    ">>> [Fallback] Found existing texture.png, loading for Pose Estimation...")
                loaded_crop = cv2.imread(str(texture_path))
                # Convert BGR (OpenCV default) to RGB (Pipeline standard)
                image_crop_for_pose = cv2.cvtColor(
                    loaded_crop, cv2.COLOR_BGR2RGB)
            else:
                print(
                    ">>> [Warning] No existing texture found. Pose Estimation might be skipped.")

        # ------------------------------------------------------------
        # Step 3: Pose Estimation
        # ------------------------------------------------------------
        if self.pose_estimator:
            # Check if we have a valid input image (from step 2 or fallback)
            if image_crop_for_pose is not None:
                print(">>> [Pipeline] Running Pose Estimation...")
                joint_overlay_path = str(
                    data["char_folder"] / "joint_overlay.png")
                char_cfg_path = str(data["char_folder"] / "char_cfg.yaml")

                pose_result = self.pose_estimator.predict(
                    image_crop_for_pose,
                    output_file=joint_overlay_path,
                    output_yaml=char_cfg_path,
                )

                results["pose"] = pose_result
                results["joint_overlay_path"] = joint_overlay_path
                results["char_cfg_path"] = char_cfg_path
            else:
                print(
                    ">>> [Error] Cannot run Pose Estimation: Input image crop is missing.")
        else:
            print(
                ">>> [Pipeline] Skipping Pose Estimation (Component not provided).")

        # ------------------------------------------------------------
        # Step 4: Animation Rendering
        # ------------------------------------------------------------
        if self.animator:
            print(">>> [Pipeline] Running Animator...")
            animation_results = {}

            # The animator relies on 'char_cfg.yaml' existing in the folder.
            # If previous steps ran (or files exist), this will succeed.
            for action in data["actions"]:
                try:
                    animation_path = self.animator.animate(
                        action, data["char_folder"], data["char_name"]
                    )
                    animation_results[action] = animation_path
                except Exception as e:
                    print(
                        f">>> [Error] Failed to animate action '{action}': {e}")

            results["animations"] = animation_results
        else:
            print(">>> [Pipeline] Skipping Animator (Component not provided).")

        return results
