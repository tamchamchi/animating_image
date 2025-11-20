# object_decomposer.py
from .interface import IObjectDecomposer
from .utils import compute_bbox_from_mask, create_mask_visualization
import numpy as np
import cv2
from typing import Dict, Optional
from PIL import Image

# --- IMPORTS CHO GROUNDINGDINO VÀ SAM ---
import torch
from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, GroundingDinoForObjectDetection

# --- CẤU HÌNH MÔ HÌNH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GROUNDING_DINO_HF_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_CHECKPOINT_PATH = "/home/anhndt/animating_image/external/checkpoints/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
BOX_THRESHOLD = 0.35  # Ngưỡng tin cậy cho BBox


class GroundedSamIntegrator:
    """Tích hợp Grounding DINO (HF) và SAM."""

    def __init__(self):
        print(f"Loading models on device: {DEVICE}...")
        try:
            self.gd_processor = AutoProcessor.from_pretrained(
                GROUNDING_DINO_HF_MODEL)
            self.gd_model = GroundingDinoForObjectDetection.from_pretrained(
                GROUNDING_DINO_HF_MODEL).to(DEVICE)
        except Exception as e:
            print(f"LỖI TẢI GROUNDING DINO: {e}")
            raise
        try:
            self.sam = sam_model_registry[SAM_MODEL_TYPE](
                checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
            self.sam_predictor = SamPredictor(self.sam)
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file SAM tại {SAM_CHECKPOINT_PATH}.")
            raise
        print("Grounded-SAM models loaded successfully.")

    @torch.no_grad()
    def get_best_mask(self, image: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """
        Tìm BBox bằng GroundingDINO và tạo mask bằng SAM cho 1 prompt.
        Chỉ trả về mask tốt nhất.
        """
        self.sam_predictor.set_image(image)
        image_pil = Image.fromarray(image)

        # 1. --- GROUNDING DINO ---
        text_prompt_formatted = f"{prompt.lower().strip()}"
        inputs = self.gd_processor(
            images=image_pil, text=text_prompt_formatted, return_tensors="pt").to(DEVICE)
        outputs = self.gd_model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs["input_ids"], target_sizes=target_sizes
        )

        boxes_xyxy = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()

        # Lọc các box có độ tin cậy thấp
        high_confidence_indices = np.where(scores >= BOX_THRESHOLD)[0]
        if len(high_confidence_indices) == 0:
            print(f"[GDINO]: Không tìm thấy BBox cho '{prompt}'.")
            return None

        # Lấy BBox có score cao nhất
        boxes_xyxy = boxes_xyxy[high_confidence_indices]
        scores = scores[high_confidence_indices]
        best_box_index = np.argmax(scores)
        box_coords = boxes_xyxy[best_box_index].astype(int)

        # 2. --- SAM ---
        # Dùng BBox tìm được để làm prompt cho SAM
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_coords[None, :],  # Đưa BBox dưới dạng [1, 4]
            multimask_output=False  # Chỉ lấy 1 mask tốt nhất
        )

        mask_np = masks[0].astype(np.uint8) * 255
        return mask_np


class ConcreteObjectDecomposer(IObjectDecomposer):
    def __init__(self):
        self.sam_integrator = GroundedSamIntegrator()

        # --- PROMPT CỐ ĐỊNH ---
        # Chúng ta dùng prompt này cho TẤT CẢ các ảnh
        # Dùng dấu "." để kết hợp nhiều khái niệm
        self.HARDCODED_PROMPT = "person . character . drawing . figure"

    def decompose(self, image: np.ndarray) -> Optional[Dict]:
        """
        Triển khai hàm decompose, sử dụng prompt cố định.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape

        # 1. Chạy pipeline với prompt cố định
        mask = self.sam_integrator.get_best_mask(
            image_rgb, self.HARDCODED_PROMPT)

        if mask is None or np.sum(mask) == 0:
            print("GroundingDINO+SAM không tìm thấy đối tượng.")
            return None

        # 2. TẠO ẢNH TEXTURE (output_texture.png)
        alpha_channel = mask
        image_bgra = cv2.merge((image, alpha_channel))

        # 3. TẠO ẢNH MASK (output_mask.png)
        mask_image_viz = create_mask_visualization(mask)

        # 4. Tính BBox
        bbox = compute_bbox_from_mask(mask)
        if bbox is None:
            bbox = (0, 0, W, H)

        return {
            "image": image_bgra,  # <-- Đây là output_texture.png (trong suốt)
            "mask": mask,
            "bounding_box": bbox,
            # <-- Đây là output_mask.png (trắng/đen)
            "mask_image_viz": mask_image_viz
        }
