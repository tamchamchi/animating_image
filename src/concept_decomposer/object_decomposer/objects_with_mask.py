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

def convert_mask_to_polygon(mask, approx_factor=0.005):
    """Chuyển đổi Mask nhị phân thành list tọa độ Polygon."""
    if mask is None: return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, approx_factor * peri, True)
    return approx.reshape(-1, 2).tolist()

def draw_visual_result(image, detections):
    """Vẽ cả BBox và Polygon lên ảnh để kiểm tra"""
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
        
        label = f"{name} ({score:.2f})"
        cv2.putText(vis_img, label, (x_min, y_min - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis_img

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
    if input_image is None: 
        print("Không tìm thấy ảnh đầu vào!")
        exit()

    decomposer = ConcreteObjectDecomposer()
    print("Processing...")
    
    # 1. Detect
    raw_results = []
    for output_name, model_prompt in TARGETS.items():
        detections = decomposer.detect_objects(input_image, [model_prompt], threshold=CONFIDENCE_THRESHOLD)
        for det in detections:
            raw_results.append({"name": output_name, "bbox": det['bbox'], "score": det['score']})

    # 2. NMS
    nms_results = apply_global_nms(raw_results, iou_threshold=IOU_THRESHOLD)
    
    # 3. Mask & Polygon
    print("Generating masks & polygons...")
    results_with_mask = decomposer.generate_masks_from_boxes(input_image, nms_results)

    # 4. Format Output
    final_json_output = []
    for item in results_with_mask:
        mask = item.get('mask')
        polygon_points = convert_mask_to_polygon(mask, approx_factor=0.005)
        
        clean_item = {
            "name": item['name'],
            "score": float(f"{item['score']:.4f}"),
            "bbox": item['bbox'],
            "polygon": polygon_points
        }
        final_json_output.append(clean_item)

    # 5. LƯU OUTPUT RA FILE JSON (Thay vì chỉ in ra màn hình)
    json_path = os.path.join(OUTPUT_DIR, "result.json")
    save_compact_json(final_json_output, json_path)
    
    # 6. Lưu ảnh Visualize
    if final_json_output:
        vis_image = draw_visual_result(input_image, final_json_output)
        vis_path = os.path.join(OUTPUT_DIR, "result_polygon.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"-> Đã lưu ảnh visualize tại: {vis_path}")