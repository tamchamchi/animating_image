# object_decomposer.py
from interface import IObjectDecomposer
from utils import compute_bbox_from_mask, create_mask_visualization
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from PIL import Image

# --- IMPORTS CHO GROUNDINGDINO VÀ SAM ---
import torch
from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, GroundingDinoForObjectDetection
import torchvision

# --- CẤU HÌNH MÔ HÌNH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GROUNDING_DINO_HF_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_CHECKPOINT_PATH = "/mnt/mmlab2024nas/anhndt/sam_vit_h_4b8939.pth" 
SAM_MODEL_TYPE = "vit_h"
BOX_THRESHOLD = 0.35 

class GroundedSamIntegrator:
    """Tích hợp Grounding DINO (HF) và SAM."""
    def __init__(self):
        print(f"Loading models on device: {DEVICE}...")
        try:
            self.gd_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_HF_MODEL)
            self.gd_model = GroundingDinoForObjectDetection.from_pretrained(GROUNDING_DINO_HF_MODEL).to(DEVICE)
        except Exception as e:
            print(f"LỖI TẢI GROUNDING DINO: {e}")
            raise
        try:
            self.sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
            self.sam_predictor = SamPredictor(self.sam)
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file SAM tại {SAM_CHECKPOINT_PATH}.")
            raise
        print("Grounded-SAM models loaded successfully.")

    @torch.no_grad()
    def predict_boxes(self, image_pil: Image.Image, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chỉ chạy GroundingDINO để lấy BBox (nhanh hơn chạy cả SAM).
        Returns: boxes_xyxy (numpy), scores (numpy)
        """
        # GroundingDINO yêu cầu prompt dạng chữ thường, kết thúc bằng dấu chấm
        text_prompt_formatted = f"{prompt.lower().strip()}." 
        
        # 1. Tạo inputs (bao gồm cả input_ids)
        inputs = self.gd_processor(images=image_pil, text=text_prompt_formatted, return_tensors="pt").to(DEVICE)
        
        # 2. Chạy model
        outputs = self.gd_model(**inputs)
        
        # 3. Xử lý kết quả (cần truyền thêm input_ids từ bước 1)
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(DEVICE) # (H, W)
        
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs, 
            input_ids=inputs.input_ids, # <--- ĐÃ THÊM DÒNG NÀY
            box_threshold=BOX_THRESHOLD, 
            text_threshold=BOX_THRESHOLD,
            target_sizes=target_sizes
        )
        
        # Lấy kết quả từ batch đầu tiên (vì batch size = 1)
        boxes_xyxy = results[0]["boxes"].cpu().numpy().astype(int)
        scores = results[0]["scores"].cpu().numpy()
        
        return boxes_xyxy, scores

    @torch.no_grad()
    def get_best_mask(self, image: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """Tìm BBox và tạo mask bằng SAM (Logic cũ)."""
        self.sam_predictor.set_image(image)
        image_pil = Image.fromarray(image)

        boxes_xyxy, scores = self.predict_boxes(image_pil, prompt)
        
        if len(scores) == 0:
            print(f"[GDINO]: Không tìm thấy BBox cho '{prompt}'.")
            return None

        # Lấy BBox có score cao nhất
        best_box_index = np.argmax(scores)
        box_coords = boxes_xyxy[best_box_index]

        # SAM Predict
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_coords[None, :],
            multimask_output=False
        )
        
        return masks[0].astype(np.uint8) * 255

class ConcreteObjectDecomposer(IObjectDecomposer):
    def __init__(self):
        self.sam_integrator = GroundedSamIntegrator()
        self.HARDCODED_PROMPT = "person . character . drawing . figure"

    def decompose(self, image: np.ndarray) -> Optional[Dict]:
        """Triển khai hàm decompose cũ."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        
        mask = self.sam_integrator.get_best_mask(image_rgb, self.HARDCODED_PROMPT)
        
        if mask is None or np.sum(mask) == 0:
            return None
        
        alpha_channel = mask
        image_bgra = cv2.merge((image, alpha_channel))
        mask_image_viz = create_mask_visualization(mask)
        bbox = compute_bbox_from_mask(mask)
        if bbox is None: bbox = (0, 0, W, H) 

        return {
            "image": image_bgra,
            "mask": mask, 
            "bounding_box": bbox, 
            "mask_image_viz": mask_image_viz
        }

    def detect_objects(self, image: np.ndarray, prompts: List[str], threshold: float = 0.40) -> List[Dict]:
        """
        Args:
            threshold (float): Ngưỡng tin cậy. Tăng lên 0.40-0.45 để lọc bớt rác.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        results = []

        # GroundingDINO cho phép dùng dấu . để ngăn cách các từ đồng nghĩa trong 1 lần chạy
        # Nhưng để dễ kiểm soát tên output, ta vẫn loop qua từng prompt
        
        for prompt_text in prompts:
            # Lưu ý: Cần truyền threshold vào hàm predict_boxes nếu bạn muốn chỉnh sâu bên trong,
            # nhưng ở đây ta lọc output cũng được.
            
            # Gọi hàm dự đoán (mặc định threshold trong model config là 0.35, ta sẽ lọc lại ở dưới)
            boxes, scores = self.sam_integrator.predict_boxes(image_pil, prompt_text)
            
            if len(boxes) == 0:
                continue

            # --- LỌC NGƯỠNG (THRESHOLD FILTERING) ---
            # Chỉ lấy những box có score >= threshold truyền vào (VD: 0.40)
            keep_indices = scores >= threshold
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            
            if len(boxes) == 0: continue

            # --- NMS (Non-Maximum Suppression) ---
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            
            # iou_threshold=0.5: Giữ lại box tốt nhất nếu trùng nhau
            nms_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
            
            final_boxes = boxes[nms_indices.numpy()]
            final_scores = scores[nms_indices.numpy()]

            for box, score in zip(final_boxes, final_scores):
                item = {
                    "name": prompt_text, # Tên trả về
                    "bbox": box.tolist(),
                    "score": float(score)
                }
                results.append(item)

        return results