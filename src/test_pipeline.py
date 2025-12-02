import os
import subprocess
from pathlib import Path

# import cv2
import torch
from dotenv import load_dotenv
from PIL import Image

from .animator.meta_animator import MetaAnimator
from .concept_decomposer.object_decomposer.object_decomposer import (
    ConcreteObjectDecomposer,
)
from .image_style_transfer.nano_banana_style_transfer import NanoBananaStyleTransfer
from .pipeline.animation_pipeline import AnimationGenerationPipeline
from .pipeline.input_data import AnimationPipelineInput
from .pose_estimator.mmpose_estimator import MMPoseEstimator
from .utils.prompt import PROMPT_IMAGE_STYLE_TRANSFER, PROMPT_SUBJECT_GENERATION
from .text_to_image import NanoBananaGenerator

load_dotenv()


# # __init__:
API_KEY = os.getenv("GOOGLE_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POSE_MODEL_CFG_PATH = os.getenv("POSE_MODEL_CFG_PATH")
POSE_MODEL_CKPT_PATH = os.getenv("POSE_MODEL_CKPT_PATH")

image_generator = NanoBananaGenerator(API_KEY)

style_transfer = NanoBananaStyleTransfer(API_KEY)

object_decomposer = ConcreteObjectDecomposer()

pose_estimator = MMPoseEstimator(
    cfg_path=POSE_MODEL_CFG_PATH, ckpt_path=POSE_MODEL_CKPT_PATH, device=DEVICE)

animator = MetaAnimator()

animation_generation_pipeline = AnimationGenerationPipeline(
    style_transfer=style_transfer,
    object_decomposer=object_decomposer,
    pose_estimator=pose_estimator,
    animator=animator
)

if __name__ == "__main__":

    style_ref_path = "/home/anhndt/animating_image/src/configs/style_ref/image_ref_3.png"
    content_image_path = "/home/anhndt/animating_image/src/configs/characters/char15/input.png"
    char_folder = Path("/home/anhndt/animating_image/src/configs/characters")
    char_name = "char15"

    try:
        style_ref = Image.open(style_ref_path).convert("RGB")
        content_image = Image.open(content_image_path).convert("RGB")
    except Exception as e:
        print(e)

    prompt = "a child with blue glass wearing a red hoodie and green pant"
    prompt = PROMPT_SUBJECT_GENERATION.format(subject=prompt)

    content_image = image_generator.generate(prompt=prompt)

    data = AnimationPipelineInput(
        style_ref=style_ref,
        content_image=content_image,
        system_prompt=PROMPT_IMAGE_STYLE_TRANSFER,
        char_folder=char_folder,
        char_name=char_name,
        actions=["standing", "jumping", "running", "jesse_dancing", "waving"]
    )

    animation_generation_pipeline.run(data)

    subprocess.run(
        ["python3", "./src/test_pygame.py"],
        check=True
    )
