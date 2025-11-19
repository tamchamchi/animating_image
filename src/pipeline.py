import os

import cv2

import torch

# from .concept_decomposer.object_decomposer.object_decomposer import (
#     ConcreteObjectDecomposer,
# )
from .concept_decomposer.pose_estimator.mmpose_estimator import MMPoseEstimator

char_dis = "/home/anhndt/animating_image/src/configs/character/char1"
image = cv2.imread("/home/anhndt/animating_image/src/configs/character/char1/texture.png", cv2.IMREAD_COLOR)


# Step1: Style Transfer

# Step2: Segmentation
# object_decomposer = ConcreteObjectDecomposer()
# seg_result = object_decomposer.decompose(image)

# mask_image_viz = seg_result["mask_image_viz"]
# bbox = seg_result["bounding_box"]

# mask_path = os.path.join(char_dis, "mask.png")
# cv2.imwrite(mask_path, mask_image_viz)

# x1, y1, x2, y2 = bbox
# crop = image[y1:y2, x1:x2] 

# crop_path = os.path.join(char_dis, "texture.png")
# cv2.imwrite(crop_path, crop)


# Step3: Pose Estemation
cfg_path = "/home/anhndt/animating_image/external/mmpose_install/config.py"
ckpt_path = "/home/anhndt/animating_image/external/mmpose_install/best_AP_epoch_72.pth"
joint_overlay_path = "/home/anhndt/animating_image/src/configs/character/char1/joint_overlay.png"
char_cfg_path = "/home/anhndt/animating_image/src/configs/character/char1/char_cfg.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pose_estimator = MMPoseEstimator(cfg_path=cfg_path, ckpt_path=ckpt_path, device=DEVICE)
res = pose_estimator.predict(image, output_file=joint_overlay_path, output_yaml=char_cfg_path)


# Step4: Create Animation
# Step5: Prompting