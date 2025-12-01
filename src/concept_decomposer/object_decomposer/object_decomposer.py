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
BOX_THRESHOLD = 0.2

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

    def detect_objects(self, image: np.ndarray, prompts: List[str], threshold: float = 0.20) -> List[Dict]:
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
    
    def generate_masks_from_boxes(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Input: Ảnh gốc và danh sách kết quả chứa bbox.
        Output: Danh sách kết quả đã được bổ sung thêm trường 'mask'.
        """
        if not detections:
            return []

        # 1. Setup ảnh cho SAM (Chỉ cần làm 1 lần cho cả bức ảnh)
        # SAM yêu cầu ảnh RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_integrator.sam_predictor.set_image(image_rgb)

        # 2. Duyệt qua từng bbox để tạo mask
        for det in detections:
            bbox = det['bbox'] # [x1, y1, x2, y2]
            
            # Chuyển bbox thành numpy array đúng format SAM yêu cầu
            box_np = np.array(bbox)

            # SAM predict: Dùng box làm gợi ý
            masks, _, _ = self.sam_integrator.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np[None, :], # Format [1, 4]
                multimask_output=False # Chỉ lấy 1 mask tốt nhất
            )
            
            # masks trả về shape [1, H, W], ta lấy [H, W]
            mask_binary = masks[0].astype(np.uint8) # 0 và 1
            
            # Lưu mask vào dictionary kết quả
            det['mask'] = mask_binary

        return detections