# run_demo.py
import numpy as np
import cv2 
from object_decomposer import ConcreteObjectDecomposer 
import os 
from typing import Optional
import glob 

def load_image_from_path(path: str) -> Optional[np.ndarray]:
    """Tải ảnh từ đường dẫn và chuyển thành định dạng numpy array (BGR)."""
    try:
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Không thể đọc ảnh từ: {path}")
        return image_bgr
    except Exception as e:
        print(f"Lỗi khi tải ảnh: {e}")
        return None

# --- CHẠY THỬ VÀ LƯU KẾT QUẢ ---
if __name__ == "__main__":
    
    TEST_IMG_DIR = "test_img" 
    BASE_OUTPUT_DIR = "decomposition_output_all" 

    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(TEST_IMG_DIR, ext)))
        
    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong thư mục: {TEST_IMG_DIR}")
        exit()
        
    print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu xử lý...")

    # Khởi tạo decomposer (việc này sẽ tải model)
    decomposer = ConcreteObjectDecomposer()
    
    for image_path in image_paths:
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        current_output_dir = os.path.join(BASE_OUTPUT_DIR, image_name)
        
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        
        print(f"\n--- BẮT ĐẦU PHÂN TÁCH ẢNH: {image_path} ---")
        
        input_image = load_image_from_path(image_path)
        if input_image is None:
            continue

        # --- GỌI HÀM DECOMPOSE MÀ KHÔNG CẦN PROMPT ---
        result_data = decomposer.decompose(input_image) 
        
        if result_data is None:
            print("PHÂN TÁCH THẤT BẠI. Không tìm thấy đối tượng.")
            continue 

        print(f"PHÂN TÁCH HOÀN TẤT. Đang lưu ảnh output vào: {current_output_dir}")

        # Lấy dữ liệu
        full_texture_image = result_data["image"] 
        mask_image_viz = result_data["mask_image_viz"]

        cv2.imwrite(os.path.join(current_output_dir, "0_input.png"), input_image)
        cv2.imwrite(os.path.join(current_output_dir, "1_output_mask.png"), mask_image_viz) 
        cv2.imwrite(os.path.join(current_output_dir, "2_output_texture.png"), full_texture_image)

    print("\n\n--- HOÀN TẤT ---")
    print(f"Đã xử lý tất cả {len(image_paths)} ảnh. Vui lòng kiểm tra thư mục '{BASE_OUTPUT_DIR}'.")