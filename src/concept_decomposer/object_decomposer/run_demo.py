# run_demo.py

import numpy as np
import cv2 
import requests
from .object_decomposer import ConcreteObjectDecomposer 
import os 
from typing import Optional

# --- CẤU HÌNH HIỂN THỊ ---
MASK_ALPHA = 0.5  # Độ trong suốt của mask khi chồng lên ảnh gốc
BBOX_COLOR = (0, 255, 0) # Màu xanh lá cây cho bounding box (BGR)
BBOX_THICKNESS = 2       # Độ dày đường viền bounding box

# --- HÀM TẢI VÀ CHUYỂN ĐỔI ẢNH TỪ URL ---
def load_image_from_url(url: str) -> Optional[np.ndarray]:
    """Tải ảnh từ URL và chuyển thành định dạng numpy array (BGR)."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError("Không thể giải mã ảnh từ URL. Định dạng ảnh không hợp lệ.")
            
        return image_bgr

    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi tải ảnh từ URL: {e}")
        return None
    except ValueError as e:
        print(f"Lỗi: {e}")
        return None


# --- CHẠY THỬ VÀ LƯU KẾT QUẢ ---
if __name__ == "__main__":
    
    # URL ẢNH VÀ PROMPTS ĐÃ CHỈ ĐỊNH
    IMAGE_URL = "https://images.pexels.com/photos/7995887/pexels-photo-7995887.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    PROMPTS = ["man", "dog", "beach"] 
    OUTPUT_DIR = "decomposition_output" 

    # Tạo thư mục output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"--- BẮT ĐẦU PHÂN TÁCH ẢNH TỪ URL, kết quả sẽ được lưu vào /{OUTPUT_DIR} ---")
    
    input_image = load_image_from_url(IMAGE_URL)
    if input_image is None:
        exit()

    decomposer = ConcreteObjectDecomposer()
    
    print(f"\nBẮT ĐẦU PHÂN TÁCH ĐỐI TƯỢNG với prompts: {PROMPTS}...")
    decomposition_results = decomposer.decompose(PROMPTS, input_image)
    print("\nPHÂN TÁCH HOÀN TẤT. Đang lưu ảnh output.")

    # Tạo một bản sao của ảnh gốc để vẽ mask và bbox lên đó
    annotated_image = input_image.copy() 
    
    # Lưu ảnh gốc sạch
    cv2.imwrite(os.path.join(OUTPUT_DIR, "0_original_image.jpg"), input_image)
    
    H, W, _ = input_image.shape
    
    # Mask tổng hợp cho TẤT CẢ đối tượng (chỉ để tham khảo)
    combined_object_mask_viz = np.zeros((H, W, 3), dtype=np.uint8)
    
    for prompt, data in decomposition_results.items():
        is_bg = data["is_background"]
        mask = data["mask"] 
        
        # Tạo màu ngẫu nhiên cho mask, giới hạn dải màu để tránh quá sáng
        mask_color = np.random.randint(50, 180, size=3, dtype=np.uint8) 

        print(f"\n--- Kết quả cho: '{prompt}' (Background: {is_bg}) ---")
        
        # --- LƯU TỪNG MASK RIÊNG BIỆT (Yêu cầu của bạn) ---
        # 1. Tạo mask riêng (nền đen, đối tượng màu trắng)
        single_mask_image = np.zeros((H, W), dtype=np.uint8)
        single_mask_image[mask > 0] = 255

        # 3. Tạo mask màu cho visualization (nền đen)
        colored_mask_image = np.zeros((H, W, 3), dtype=np.uint8)
        colored_mask_image[mask > 0] = mask_color
        # -----------------------------------------------------------
        
        # Luôn thêm mask màu vào ảnh tổng hợp 
        combined_object_mask_viz[mask > 0] = mask_color 


        if not is_bg:
            # --- XỬ LÝ FOREGROUND (Man, Dog) ---
            
            # 4. Overlay Mask trong suốt lên ảnh gốc (annotated_image)
            # Dùng kỹ thuật trộn màu thủ công để giữ độ sáng ảnh gốc.
            
            # Lấy vùng ảnh gốc (BG)
            background_pixels = annotated_image[mask > 0]
            # Lấy vùng màu mask (FG)
            foreground_pixels = colored_mask_image[mask > 0]
            
            # Trộn màu thủ công: (1-alpha)*BG + alpha*FG
            new_pixels = (1 - MASK_ALPHA) * background_pixels.astype(np.float32) + \
                         MASK_ALPHA * foreground_pixels.astype(np.float32)
            
            annotated_image[mask > 0] = new_pixels.astype(np.uint8)

            # 5. Vẽ Bounding Box và Lưu ảnh Crop (RGBA)
            bbox = data["bounding_box"]
            cropped_image = data["image"] # Ảnh BGRA trong suốt

            print(f"  - Bounding Box: {bbox}")
            
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), BBOX_COLOR, BBOX_THICKNESS)
            
            # Ghi ảnh đã cắt ra file PNG (để giữ kênh Alpha/trong suốt)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"1_crop_{prompt}.png"), cropped_image)
            
        else:
            # --- XỬ LÝ BACKGROUND (Beach) ---
            print("  - Đây là Background. Mask đã được lưu riêng biệt.")
            
    # Lưu ảnh gốc đã được chú thích (chỉ có mask và bbox của foreground)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_annotated_image_with_masks_and_bboxes.jpg"), annotated_image)

    print(f"\nĐã lưu ảnh kết quả vào thư mục '{OUTPUT_DIR}'. Vui lòng kiểm tra thư mục này.")