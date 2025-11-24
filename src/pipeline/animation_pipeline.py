import cv2

from src.animator import IAnimator
from src.concept_decomposer import IObjectDecomposer
from src.image_style_transfer import IStyleTransfer
from src.pose_estimator import IPoseEstimator

from .input_data import AnimationPipelineInput
import numpy as np


class AnimationGenerationPipeline:
    """
    A full animation generation pipeline that performs:
    1. Style transfer
    2. Object segmentation & texture extraction
    3. Pose estimation
    4. Animation rendering

    Each step delegates processing to modular components:
    - IStyleTransfer
    - IObjectDecomposer
    - IPoseEstimator
    - IAnimator
    """

    def __init__(
        self,
        style_transfer: IStyleTransfer,
        object_decomposer: IObjectDecomposer,
        pose_estimator: IPoseEstimator,
        animator: IAnimator,
    ):
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
        Execute the complete animation pipeline.

        Steps:
        1. Load pipeline data
        2. Apply style transfer to the content image
        3. Perform object segmentation and texture cropping
        4. Run pose estimation to generate joints & YAML config
        5. Animate character for each requested action

        Returns:
            A dictionary containing outputs of all stages
        """

        # Load all structured input data (images, paths, parameters)
        data = input_data.load_data()
        results = {}

        # ------------------------------------------------------------
        # Step 1: Style Transfer
        # ------------------------------------------------------------
        object_stylized = self.style_transfer.transfer(
            style_ref=data["style_ref"],
            image=data["content_image"],
            prompt=data["prompt"],
        )

        # Save stylized object
        stylized_path = data["char_folder"] / "object_stylized.png"
        object_stylized.save(stylized_path)

        object_stylized_np = np.array(object_stylized)
        results["object_stylized"] = object_stylized
        results["stylized_path"] = stylized_path

        # ------------------------------------------------------------
        # Step 2: Segmentation + Crop
        # ------------------------------------------------------------
        seg_result = self.object_decomposer.decompose(object_stylized_np)
        mask_image_viz = seg_result["mask_image_viz"]
        bbox = seg_result["bounding_box"]

        # Expand bounding box a bit to ensure full coverage
        x1, y1, x2, y2 = self._expand_bbox(bbox, pad=4)

        # Crop texture and mask using the bounding box
        mask_crop = mask_image_viz[y1:y2, x1:x2]
        image_crop = object_stylized_np[y1:y2, x1:x2]

        # Save cropped mask and texture
        mask_path = data["char_folder"] / "mask.png"
        texture_path = data["char_folder"] / "texture.png"
        mask_crop_cv2 = cv2.cvtColor(mask_crop, cv2.COLOR_RGB2BGR)
        image_crop_cv2 = cv2.cvtColor(image_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(mask_path), mask_crop_cv2)
        cv2.imwrite(str(texture_path), image_crop_cv2)

        results["segmentation"] = seg_result
        results["mask_crop"] = mask_crop
        results["mask_path"] = mask_path
        results["image_crop"] = image_crop
        results["texture_path"] = texture_path

        # ------------------------------------------------------------
        # Step 3: Pose Estimation
        # ------------------------------------------------------------
        joint_overlay_path = str(data["char_folder"] / "joint_overlay.png")
        char_cfg_path = str(data["char_folder"] / "char_cfg.yaml")

        pose_result = self.pose_estimator.predict(
            image_crop,
            output_file=joint_overlay_path,
            output_yaml=char_cfg_path,
        )

        results["pose"] = pose_result
        results["joint_overlay_path"] = joint_overlay_path
        results["char_cfg_path"] = char_cfg_path

        # ------------------------------------------------------------
        # Step 4: Animation Rendering
        # ------------------------------------------------------------
        animation_results = {}
        for action in data["actions"]:
            animation_path = self.animator.animate(
                action, data["char_folder"], data["char_name"]
            )
            animation_results[action] = animation_path

        results["animations"] = animation_results

        return results
