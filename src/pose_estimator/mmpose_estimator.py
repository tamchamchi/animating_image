from typing import Union
import numpy as np
import mmcv
import cv2
import yaml
import os

from mmpose.apis import (
    init_pose_model,
    inference_top_down_pose_model,
    inference_bottom_up_pose_model
)
from mmpose.models.detectors import TopDown, AssociativeEmbedding

from .interface import IPoseEstimator


class MMPoseEstimator(IPoseEstimator):
    """
    Pose estimator for MMPose 0.x (legacy API) with YAML output.

    Args:
        cfg_path (str): Path to MMPose config (.py).
        ckpt_path (str): Path to model checkpoint (.pth).
        device (str): Device to load model ('cpu' or 'cuda:0').
    """

    # COCO17 joint indices
    JOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(self, cfg_path: str, ckpt_path: str, device: str = "cpu"):
        super().__init__()
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.device = device
        self.model = self._init_model()

    def _init_model(self):
        """Load MMPose model using legacy init_pose_model API."""
        model = init_pose_model(
            config=self.cfg_path,
            checkpoint=self.ckpt_path,
            device=self.device
        )
        return model

    def predict(
        self,
        image: Union[np.ndarray, str],
        output_file: str = None,
        output_yaml: str = None,
        joint_height: int = 392
    ):
        """
        Perform pose estimation on a single image, optionally saving the keypoints
        visualization and a YAML skeleton file.

        Args:
            image (np.ndarray or str): Input image array (H,W,C) or image path.
            output_file (str, optional): If provided, save the keypoints visualization.
            output_yaml (str, optional): If provided, save skeleton as YAML file.
            joint_height (int, optional): Height of the character for YAML scaling.

        Returns:
            preds (list[dict]): List of pose predictions with keys 'keypoints' and 'bbox'.
            used_bboxes (np.ndarray): Bounding boxes used for top-down inference.
        """

        if isinstance(image, str):
            image = mmcv.imread(image)

        # Top-down model
        if isinstance(self.model, TopDown):
            preds, used_bboxes = inference_top_down_pose_model(self.model, image)

        # Bottom-up model
        elif isinstance(self.model, AssociativeEmbedding):
            preds, used_bboxes = inference_bottom_up_pose_model(self.model, image)

        else:
            raise NotImplementedError(f"Unsupported model type: {type(self.model)}")

        # ---- Draw keypoints if output_file specified ----
        if output_file:
            vis_image = image.copy()
            for person in preds:
                kpts = person["keypoints"]
                for idx, (x, y, s) in enumerate(kpts):
                    if s > 0.2:
                        # Vẽ điểm khớp
                        cv2.circle(vis_image, (int(x), int(y)), 3, (0, 0, 255), -1)
                        # Vẽ label tên khớp
                        cv2.putText(
                            vis_image,
                            self.JOINT_NAMES[idx],          # tên khớp từ danh sách JOINT_NAMES
                            (int(x) + 2, int(y) - 2),      # vị trí label
                            cv2.FONT_HERSHEY_SIMPLEX,       # font chữ
                            0.4,                            # scale
                            (0, 255, 0),                    # màu xanh lá
                            1,                              # độ dày nét chữ
                            cv2.LINE_AA                     # kiểu line
                        )
            cv2.imwrite(output_file, vis_image)

        # ---- Generate YAML skeleton if requested ----
        if output_yaml and preds:
            # Only process the first person for YAML
            kpts = preds[0]["keypoints"]

            # Compute root (average of left_hip & right_hip)
            hip_indices = [11, 12]  # left_hip, right_hip
            root_x = float(np.mean(kpts[hip_indices, 0]))
            root_y = float(np.mean(kpts[hip_indices, 1]))

            # Compute torso/chest (average of left_shoulder & right_shoulder)
            shoulder_indices = [5, 6]
            torso_x = float(np.mean(kpts[shoulder_indices, 0]))
            torso_y = float(np.mean(kpts[shoulder_indices, 1]))

            # Build YAML skeleton
            skeleton = [
                {"loc": [int(root_x), int(root_y)], "name": "root", "parent": None},
                {"loc": [int(root_x), int(root_y)], "name": "hip", "parent": "root"},
                {"loc": [int(torso_x), int(torso_y)], "name": "torso", "parent": "hip"},
                {"loc": [int(kpts[0,0]), int(kpts[0,1])], "name": "neck", "parent": "torso"},
                {"loc": [int(kpts[6,0]), int(kpts[6,1])], "name": "right_shoulder", "parent": "torso"},
                {"loc": [int(kpts[8,0]), int(kpts[8,1])], "name": "right_elbow", "parent": "right_shoulder"},
                {"loc": [int(kpts[10,0]), int(kpts[10,1])], "name": "right_hand", "parent": "right_elbow"},
                {"loc": [int(kpts[5,0]), int(kpts[5,1])], "name": "left_shoulder", "parent": "torso"},
                {"loc": [int(kpts[7,0]), int(kpts[7,1])], "name": "left_elbow", "parent": "left_shoulder"},
                {"loc": [int(kpts[9,0]), int(kpts[9,1])], "name": "left_hand", "parent": "left_elbow"},
                {"loc": [int(kpts[12,0]), int(kpts[12,1])], "name": "right_hip", "parent": "root"},
                {"loc": [int(kpts[14,0]), int(kpts[14,1])], "name": "right_knee", "parent": "right_hip"},
                {"loc": [int(kpts[16,0]), int(kpts[16,1])], "name": "right_foot", "parent": "right_knee"},
                {"loc": [int(kpts[11,0]), int(kpts[11,1])], "name": "left_hip", "parent": "root"},
                {"loc": [int(kpts[13,0]), int(kpts[13,1])], "name": "left_knee", "parent": "left_hip"},
                {"loc": [int(kpts[15,0]), int(kpts[15,1])], "name": "left_foot", "parent": "left_knee"},
            ]


            yaml_dict = {
                "height": int(image.shape[0]),
                "skeleton": skeleton,
                "width": int(image.shape[1])
            }

            # Save YAML
            os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
            with open(output_yaml, "w") as f:
                yaml.dump(yaml_dict, f)

        return preds, used_bboxes
