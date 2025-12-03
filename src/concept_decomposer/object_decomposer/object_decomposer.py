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
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
GROUNDING_DINO_HF_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_CHECKPOINT_PATH = "/mnt/mmlab2024nas/anhndt/sam_vit_h_4b8939.pth" 
SAM_MODEL_TYPE = "vit_h"
BOX_THRESHOLD = 0.2

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
    def get_mask_from_box(self, box: np.ndarray) -> np.ndarray:
        """Tạo mask từ 1 bbox"""
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None, point_labels=None,
            box=box[None, :], multimask_output=False
        )
        return masks[0].astype(np.uint8) * 255
    
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

    def convert_mask_to_polygon(self, mask, approx_factor=0.005):
        """Helper chuyển đổi Mask -> Polygon"""
        if mask is None: return []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return []
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_factor * peri, True)
        return approx.reshape(-1, 2).tolist()

    def detect_objects(self, image: np.ndarray, prompts: List[str], threshold: float = 0.20) -> List[Dict]:
        """
        Input: Ảnh + Prompt
        Output: List Dict chứa {name, bbox, score, polygon}
        """
        if DEVICE == "cuda": torch.cuda.empty_cache()

        # 1. Setup ảnh cho SAM (Làm 1 lần duy nhất cho toàn bộ prompt)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        self.sam_integrator.sam_predictor.set_image(image_rgb)
        
        results = []

        for prompt_text in prompts:
            # 2. Detect BBox (GroundingDINO)
            boxes, scores = self.sam_integrator.predict_boxes(image_pil, prompt_text)
            if len(boxes) == 0: continue

            # 3. Lọc ngưỡng (Threshold)
            keep_indices = scores >= threshold
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            if len(boxes) == 0: continue
            
            # 4. Lọc trùng lặp nội bộ (NMS per class)
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            nms_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
            
            final_boxes = boxes[nms_indices.numpy()]
            final_scores = scores[nms_indices.numpy()]

            # 5. Duyệt qua các box đã lọc để tạo Mask & Polygon
            for box, score in zip(final_boxes, final_scores):
                # Tạo Mask bằng SAM
                mask = self.sam_integrator.get_mask_from_box(box)
                
                # Tạo Polygon
                polygon = self.convert_mask_to_polygon(mask, approx_factor=0.005)

                item = {
                    "name": prompt_text,
                    "score": float(f"{score:.4f}"),
                    "bbox": box.tolist(),
                    "polygon": polygon
                }
                results.append(item)
                
        return results

    def draw_visual_result(self, image, detections):
        """Hàm vẽ visualize"""
        vis_img = image.copy()
        for det in detections:
            bbox = det['bbox']
            name = det['name']
            score = det['score']
            polygon = det.get('polygon', [])
            
            color = (0, 255, 255) if "ac_" in name else (0, 255, 0)
            
            if polygon:
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], True, color, 2)
                for point in polygon:
                    cv2.circle(vis_img, tuple(point), 3, (0, 0, 255), -1)

            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), color, 1)
            cv2.putText(vis_img, f"{name} ({score:.2f})", (x_min, y_min - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return vis_img