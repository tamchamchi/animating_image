import cv2

from src.animator import IAnimator
from src.concept_decomposer import IObjectDecomposer
from src.image_style_transfer import IStyleTransfer
from src.pose_estimator import IPoseEstimator

from .input_data import AnimationPipelineInput
import numpy as np


class AnimationGenerationPipeline:
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
        x1, y1, x2, y2 = bbox
        return (max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad)

    def run(self, input_data: AnimationPipelineInput):
        # Load data
        data = input_data.load_data()

        results = {}

        # Step 1: Style Transfer
        image_stylized = self.style_transfer.transfer(
            style_ref=data["style_ref"],
            image=data["content_image"],
            prompt=data["prompt"],
        )
        stylized_path = data["char_folder"] / "image_stylized.png"
        image_stylized.save(stylized_path)

        image_stylized_np = np.array(image_stylized)
        results["image_stylized"] = image_stylized
        results["stylized_path"] = stylized_path

        # Step 2: Segmentation
        seg_result = self.object_decomposer.decompose(image_stylized_np)
        mask_image_viz = seg_result["mask_image_viz"]
        bbox = seg_result["bounding_box"]
        x1, y1, x2, y2 = self._expand_bbox(bbox, pad=4)

        mask_crop = mask_image_viz[y1:y2, x1:x2]
        image_crop = image_stylized_np[y1:y2, x1:x2]

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

        # Step 3: Pose Estimation
        joint_overlay_path = str(data["char_folder"] / "joint_overlay.png")
        char_cfg_path = str(data["char_folder"] / "char_cfg.yaml")

        pose_result = self.pose_estimator.predict(
            image_crop, output_file=joint_overlay_path, output_yaml=char_cfg_path
        )

        results["pose"] = pose_result
        results["joint_overlay_path"] = joint_overlay_path
        results["char_cfg_path"] = char_cfg_path

        # Step 4: Animation
        animation_results = {}
        for action in data["actions"]:
            animation_path = self.animator.animate(
                action, data["char_folder"], data["char_name"]
            )
            animation_results[action] = animation_path

        results["animations"] = animation_results

        return results
