import os

import cv2

import torch

from .concept_decomposer.object_decomposer.object_decomposer import ConcreteObjectDecomposer
from .pose_estimator.mmpose_estimator import MMPoseEstimator
from .animator.meta_animator import MetaAnimator

# __init__:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

object_decomposer = ConcreteObjectDecomposer()

cfg_path = "/home/anhndt/animating_image/external/mmpose_install/config.py"
ckpt_path = "/home/anhndt/animating_image/external/mmpose_install/best_AP_epoch_72.pth"
pose_estimator = MMPoseEstimator(cfg_path=cfg_path, ckpt_path=ckpt_path, device=DEVICE)

animator = MetaAnimator()

#==============================================
char_name = "char10"
char_dir = "/home/anhndt/animating_image/src/configs/characters/char10"
actions = ["jumping", "walking", "jumping", "dancing"]

joint_overlay_path = f"{char_dir}/joint_overlay.png"
char_cfg_path = f"{char_dir}/char_cfg.yaml"

image = cv2.imread(f"{char_dir}/object_stylized.png", cv2.IMREAD_COLOR)
# ==============================================

# Step1: Style Transfer

# Step2: Segmentation
seg_result = object_decomposer.decompose(image)
mask_image_viz = seg_result["mask_image_viz"]
bbox = seg_result["bounding_box"]

def expand_bbox(bbox, pad=2):
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        x2 + pad,
        y2 + pad
    )

x1, y1, x2, y2 = expand_bbox(bbox, 2)

mask_crop = mask_image_viz[y1:y2, x1:x2]

image_crop = image[y1:y2, x1:x2]


cv2.imwrite(os.path.join(char_dir, "mask.png"), mask_crop)
cv2.imwrite(os.path.join(char_dir, "texture.png"), image_crop)

# Step3: Pose Estemation
res = pose_estimator.predict(
    image_crop, output_file=joint_overlay_path, output_yaml=char_cfg_path
)

# Step4: Create Animation
for action in actions:
    animator.animate(action, char_dir, char_name)
# Step5: Prompting
