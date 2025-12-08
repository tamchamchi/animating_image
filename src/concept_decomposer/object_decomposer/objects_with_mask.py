import numpy as np
import cv2 
from object_decomposer import ConcreteObjectDecomposer 
import os 
import json
import torch
import torchvision
import re # <--- [THÊM] Thư viện này để xử lý format text

def apply_global_nms(detections, iou_threshold=0.3):
    """Lọc trùng lặp (Global NMS)"""
    if not detections: return []
    
    boxes_list = [d['bbox'] for d in detections]
    scores_list = [d['score'] for d in detections]
    
    boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
    scores_tensor = torch.tensor(scores_list, dtype=torch.float32)
    
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=iou_threshold)
    return [detections[i] for i in keep_indices.numpy()]

# --- [THÊM] HÀM LƯU JSON GỌN GÀNG ---
def save_compact_json(data, file_path):
    """Lưu file JSON và ép các list số về 1 dòng."""
    # 1. Tạo string JSON gốc
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    # 2. Hàm gom dòng cho Regex
    def collapse_list(match):
        content = match.group(1)
        compact = re.sub(r'\s+', ' ', content).strip()
        compact = compact.replace(" ,", ",")
        return f"[{compact}]"

    # 3. Regex tìm các list số và gom lại
    compact_json_str = re.sub(r'\[\s*([0-9\.,\s\-]+)\s*\]', collapse_list, json_str)
    
    # 4. Ghi file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(compact_json_str)
    print(f"-> Đã lưu file JSON tại: {file_path}")

if __name__ == "__main__":
    # --- CONFIG ---
    IMG_PATH = "test_img/background.jpg"
    OUTPUT_DIR = "detection_output"
    CONFIDENCE_THRESHOLD = 0.20 
    IOU_THRESHOLD = 0.3
    TARGETS = {
        "ac_controller": "thermostat . electronic device with screen . digital wall controller",
        "notice_board": "framed text . framed certificate"
    }

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    input_image = cv2.imread(IMG_PATH)
    if input_image is None: exit()

    decomposer = ConcreteObjectDecomposer()
    print("Processing...")
    
    # 1. GỌI HÀM DUY NHẤT: detect_objects (Trả về luôn Polygon)
    raw_results = []
    for output_name, model_prompt in TARGETS.items():
        # Hàm này giờ đã trả về full info: bbox, score, polygon
        detections = decomposer.detect_objects(input_image, [model_prompt], threshold=CONFIDENCE_THRESHOLD)
        
        # Gán tên output (VD: ac_controller) thay vì tên prompt dài
        for det in detections:
            det['name'] = output_name 
            raw_results.append(det)

    # 2. Lọc chồng lấn chéo (Global NMS)
    # Ví dụ: Xóa ac_controller nếu nó nằm đè lên notice_board
    final_results = apply_global_nms(raw_results, iou_threshold=IOU_THRESHOLD)
    
    # 3. Lưu kết quả
    save_compact_json(final_results, os.path.join(OUTPUT_DIR, "result.json"))
    
    # 4. Visualize
    if final_results:
        vis_image = decomposer.draw_visual_result(input_image, final_results)
        vis_path = os.path.join(OUTPUT_DIR, "result_polygon.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"-> Đã lưu ảnh visualize tại: {vis_path}")