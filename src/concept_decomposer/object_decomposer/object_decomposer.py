from .interface import IObjectDecomposer
from .utils import compute_bbox_from_mask, create_mask_visualization
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from PIL import Image

# --- GROUNDED-SAM IMPORTS (Hugging Face Transformers) ---
import torch
from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, GroundingDinoForObjectDetection

# --- CẤU HÌNH MÔ HÌNH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grounding DINO sẽ được tải tự động từ Hugging Face
GROUNDING_DINO_HF_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_CHECKPOINT_PATH = "/home/anhndt/animating_image/src/concept_decomposer/object_decomposer/checkpoints/sam_vit_h_4b8939.pth" # Đường dẫn file đã tải
SAM_MODEL_TYPE = "vit_h"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

class GroundedSamIntegrator:
    """Tích hợp Grounding DINO (HF) và SAM."""
    def __init__(self):
        print(f"Loading models on device: {DEVICE}...")
        
        # 1. Load Grounding DINO (Tự động tải model/trọng số từ HF)
        try:
            self.gd_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_HF_MODEL)
            self.gd_model = GroundingDinoForObjectDetection.from_pretrained(GROUNDING_DINO_HF_MODEL).to(DEVICE)
        except Exception as e:
            print(f"LỖI TẢI GROUNDING DINO từ HF: {e}")
            raise
        
        # 2. Load SAM
        try:
            self.sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
            self.sam_predictor = SamPredictor(self.sam)
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file SAM tại {SAM_CHECKPOINT_PATH}. Vui lòng kiểm tra lại Bước tải file.")
            raise

        print("Grounded-SAM models loaded successfully.")

    @torch.no_grad()

    def get_masks(self, image: np.ndarray, prompts: List[str]) -> Dict[str, np.ndarray]:
        """Thực hiện phát hiện và phân đoạn."""
        all_masks = {}
        self.sam_predictor.set_image(image)
        image_pil = Image.fromarray(image)

        for prompt in prompts:
            # --- GROUNDING DINO (HF) ---
            text_prompt_formatted = f"{prompt.lower().strip()}."
            inputs = self.gd_processor(images=image_pil, text=text_prompt_formatted, return_tensors="pt").to(DEVICE)
            outputs = self.gd_model(**inputs)
            target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
    
            # CHÚ Ý: Bỏ tham số 'box_threshold' và 'text_threshold' khỏi hàm post_process.
            # Thay vào đó, ta sẽ lọc boxes thủ công.
            results = self.gd_processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes
            )
            # --- Lọc boxes thủ công (Thay thế cho tham số bị xóa) ---
            boxes_xyxy = results[0]["boxes"].cpu().numpy()
            scores = results[0]["scores"].cpu().numpy()
        
            # Lọc theo ngưỡng đã định nghĩa
            high_confidence_indices = np.where(scores >= BOX_THRESHOLD)[0]
            boxes_xyxy = boxes_xyxy[high_confidence_indices]
            # Sắp xếp theo score và lấy BBox có score cao nhất
            sorted_indices = np.argsort(scores[high_confidence_indices])[::-1]

            if len(sorted_indices) == 0:
                print(f"[GDINO]: Không tìm thấy BBox cho '{prompt}'.")
                continue

            # Lấy BBox đầu tiên có confidence cao nhất
            box_coords = boxes_xyxy[sorted_indices[0]].astype(int)

            # --- SAM ---
            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_coords[None, :], # Đưa BBox dưới dạng [1, 4]
                multimask_output=False
            )
            mask_np = masks[0].astype(np.uint8) * 255
            all_masks[prompt] = mask_np
        
        return all_masks

def _is_background_prompt(prompt: str, mask: Optional[np.ndarray] = None, image_shape: Optional[Tuple[int, ...]] = None) -> bool:
    """Logic phân biệt Background tổng quát (dựa trên từ khóa và diện tích)."""

    prompt_lower = prompt.lower()
    background_keywords = ['beach', 'sky', 'wall', 'floor', 'background', 'street', 'grass', 'sea', 'water', 'road', 'mountain', 'air', 'cloud', 'terrain', 'ground']

    if any(keyword in prompt_lower for keyword in background_keywords):
        return True

    if mask is not None and image_shape is not None:
        H, W = image_shape[:2]
        total_pixels = H * W
        mask_pixels = np.sum(mask > 0)
        coverage_ratio = mask_pixels / total_pixels

        if coverage_ratio > 0.7 and mask_pixels > 0:
            return True
           
    return False

class ConcreteObjectDecomposer(IObjectDecomposer):
    def __init__(self):
        self.sam_integrator = GroundedSamIntegrator()
    def decompose(self, prompt_i: List[str], image: np.ndarray) -> Dict[str, Dict]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_masks = self.sam_integrator.get_masks(image_rgb, prompt_i)       
        results = {}
        for prompt in prompt_i:
            mask = all_masks.get(prompt)
            if mask is None or np.sum(mask) == 0:
                continue
            is_bg = _is_background_prompt(prompt, mask=mask, image_shape=image.shape)
            mask_image = create_mask_visualization(mask)

            if not is_bg:
                bbox = compute_bbox_from_mask(mask)
                if bbox is None: continue
                x_min, y_min, x_max, y_max = bbox
               
                # --- PHẦN ĐÃ SỬA ĐỔI: TẠO ẢNH RGBA TRONG SUỐT THEO MASK ---
                # 1. Tạo kênh Alpha (Alpha channel) từ mask (mask là HxW với giá trị 0/255)
                alpha_channel = mask
                # 2. Kết hợp kênh RGB và Alpha để tạo ảnh BGRA (4 kênh)
                # Lưu ý: OpenCV sử dụng định dạng BGR theo mặc định
                image_bgra = cv2.merge((image, alpha_channel))
               
                # 3. Cắt vùng hình chữ nhật từ ảnh BGRA
                cropped_image = image_bgra[y_min:y_max+1, x_min:x_max+1]
                # --- KẾT THÚC PHẦN ĐÃ SỬA ĐỔI ---
               
                results[prompt] = {
                    "is_background": False,
                    "image": cropped_image, # <-- Giờ là ảnh BGRA
                    "mask": mask,
                    "bounding_box": bbox,
                    "mask_image": mask_image
                }
            else:
                results[prompt] = {
                    "is_background": True,
                    "image": None,
                    "mask": mask,
                    "bounding_box": None,
                    "mask_image": mask_image
                }
        return results