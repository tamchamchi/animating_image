from .pipeline.animation_pipeline import AnimationGenerationPipeline
from .pipeline.input_data import AnimationPipelineInput
from .animator.meta_animator import MetaAnimator
from .pose_estimator.mmpose_estimator import MMPoseEstimator
from .concept_decomposer.object_decomposer.object_decomposer import ConcreteObjectDecomposer
from .image_style_transfer.nano_banana_style_transfer import NanoBananaStyleTransfer
from .utils.config import PROMPT_IMAGE_STYLE_TRANSFER
import os
from dotenv import load_dotenv
from pathlib import Path

# import cv2

import torch

load_dotenv()


# # __init__:
API_KEY = os.getenv("GOOGLE_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg_path = "/home/anhndt/animating_image/external/mmpose_install/config.py"
ckpt_path = "/home/anhndt/animating_image/external/mmpose_install/best_AP_epoch_72.pth"

style_transfer = NanoBananaStyleTransfer(API_KEY)

object_decomposer = ConcreteObjectDecomposer()

pose_estimator = MMPoseEstimator(
    cfg_path=cfg_path, ckpt_path=ckpt_path, device=DEVICE)

animator = MetaAnimator()

animation_generation_pipeline = AnimationGenerationPipeline(
    style_transfer=style_transfer,
    object_decomposer=object_decomposer,
    pose_estimator=pose_estimator,
    animator=animator
)

data = AnimationPipelineInput(
    style_ref_path="/home/anhndt/animating_image/src/configs/style_ref/image_ref_2.png",
    content_image_path="/home/anhndt/animating_image/src/configs/characters/char12/input.png",
    system_prompt=PROMPT_IMAGE_STYLE_TRANSFER,
    char_folder=Path("/home/anhndt/animating_image/src/configs/characters"),
    char_name="char12",
    actions=["standing", "jumping", "running", "jesse_dancing"]
)

## Inference

animation_generation_pipeline.run(data)

# #==============================================
# char_name = "char10"
# char_dir = f"/home/anhndt/animating_image/src/configs/characters/{char_name}"
# actions = ["jumping", "walking", "jumping", "dancing"]

# joint_overlay_path = f"{char_dir}/joint_overlay.png"
# char_cfg_path = f"{char_dir}/char_cfg.yaml"

# image = cv2.imread(f"{char_dir}/object_stylized.png", cv2.IMREAD_COLOR)
# # ==============================================

# # Step1: Style Transfer

# # Step2: Segmentation
# seg_result = object_decomposer.decompose(image)
# mask_image_viz = seg_result["mask_image_viz"]
# bbox = seg_result["bounding_box"]

# def expand_bbox(bbox, pad=2):
#     x1, y1, x2, y2 = bbox
#     return (
#         max(0, x1 - pad),
#         max(0, y1 - pad),
#         x2 + pad,
#         y2 + pad
#     )

# x1, y1, x2, y2 = expand_bbox(bbox, 2)

# mask_crop = mask_image_viz[y1:y2, x1:x2]

# image_crop = image[y1:y2, x1:x2]


# cv2.imwrite(os.path.join(char_dir, "mask.png"), mask_crop)
# cv2.imwrite(os.path.join(char_dir, "texture.png"), image_crop)

# # Step3: Pose Estemation
# res = pose_estimator.predict(
#     image_crop, output_file=joint_overlay_path, output_yaml=char_cfg_path
# )

# # Step4: Create Animation
# for action in actions:
#     animator.animate(action, char_dir, char_name)
# # Step5: Prompting
