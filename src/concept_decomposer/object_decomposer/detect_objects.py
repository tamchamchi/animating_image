import numpy as np
import cv2 
from object_decomposer import ConcreteObjectDecomposer 
import os 
import json

def draw_results(image, detections):
    vis_img = image.copy()
    for det in detections:
        bbox = det['bbox']
        name = det['name']
        score = det['score']
        
        x_min, y_min, x_max, y_max = bbox
        
        # Vẽ box
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        label = f"{name} ({score:.2f})"
        cv2.putText(vis_img, label, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return vis_img

if __name__ == "__main__":
    # --- CONFIG ---
    IMG_PATH = "test_img/image.png"
    OUTPUT_DIR = "detection_output"
    CONFIDENCE_THRESHOLD = 0.40 # Ngưỡng lọc rác (TV)

    # --- MAPPING PROMPTS ---
    # Key: Tên hiển thị trong Output (Ngắn gọn)
    # Value: Prompt giúp model hiểu (Mô tả hình dáng)
    TARGETS = {
        "ac_controller": "thermostat . electronic device with screen . digital wall controller",
        "notice_board": "framed text . framed certificate"          # Gọi là notice_board
    }

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load ảnh
    input_image = cv2.imread(IMG_PATH)
    if input_image is None:
        print("Error: Image not found!")
        exit()

    # 2. Khởi tạo
    decomposer = ConcreteObjectDecomposer()
    
    print(f"Processing...")
    
    final_output = []

    # 3. Detect loop
    for output_name, model_prompt in TARGETS.items():
        # Gọi hàm detect (với list prompt chứa 1 phần tử)
        detections = decomposer.detect_objects(input_image, [model_prompt], threshold=CONFIDENCE_THRESHOLD)
        
        for det in detections:
            # Format lại object theo đúng yêu cầu
            item = {
                "name": output_name,      # "switch_panel" hoặc "notice_board"
                "bbox": det['bbox'],       # [x1, y1, x2, y2]
                "score": det['score']   # (Optional) Bỏ comment dòng này nếu muốn xem điểm
            }
            final_output.append(item)

    # 4. IN KẾT QUẢ ĐÚNG FORMAT JSON LIST
    # indent=None để nó in trên 1 dòng hoặc indent=2 để dễ nhìn
    print("\n--- RESULT ---")
    print(json.dumps(final_output, indent=2))
    
    # 5. Lưu ảnh visualize (Optional)
    if final_output:
        # Cần map lại format để vẽ (hàm vẽ cần key 'score' nếu có)
        vis_data = []
        for item in final_output:
            vis_data.append({"name": item["name"], "bbox": item["bbox"], "score": item["score"]})
            
        vis_image = draw_results(input_image, vis_data)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "result_visualized.jpg"), vis_image)