# Animating Image
# Object Decomposer

sử dụng Grounding DINO và Segment Anything (SAM) để phát hiện, phân đoạn và tách các đối tượng từ một hình ảnh dựa trên mô tả văn bản (text prompts).

## Yêu cầu hệ thống

- **Python 3.12** (Đã kiểm tra trên Google Colab `3.12.12`)
- **NVIDIA GPU** (Rất khuyến khích để chạy mô hình)

---

## Hướng dẫn cài đặt

Vui lòng thực hiện các bước theo **đúng thứ tự**. Môi trường Google Colab đã cài đặt sẵn nhiều thư viện, nhưng khi chạy trên máy cá nhân (local), bạn phải cài đặt chúng.

### Bước 1: Clone dự án và tạo Môi trường ảo

1.  Clone repository này về máy của bạn.
2.  Mở terminal, điều hướng vào thư mục dự án và tạo một môi trường ảo:

    ```bash
    # Đảm bảo bạn đang dùng Python 3.12
    python3 -m venv venv
    ```

3.  Kích hoạt môi trường ảo:

    - Trên macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
    - Trên Windows (Command Prompt):
      ```bash
      venv\Scripts\activate
      ```

### Bước 2: Cài đặt PyTorch (Bước quan trọng nhất)

cài đặt `torch` và `torchvision` thủ công để chúng phù hợp với phần cứng (GPU hoặc CPU) của bạn.

Truy cập [Trang chủ chính thức của PyTorch](https://pytorch.org/get-started/locally/) để lấy lệnh cài đặt mới nhất.

**Ví dụ phổ biến (cho Python 3.12):**

- **Nếu bạn có GPU NVIDIA (CUDA 12.1):** (Đây là bản phổ biến nhất)

  ```bash
  pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
  ```

- **Nếu bạn có GPU NVIDIA (CUDA 11.8):**

  ```bash
  pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
  ```

- **Nếu bạn chỉ dùng CPU (Không khuyến nghị, sẽ rất chậm):**
  ```bash
  pip install torch torchvision
  ```

### Bước 3: Cài đặt các thư viện còn lại

Chỉ sau khi `torch` và `torchvision` đã được cài đặt thành công, bạn mới chạy lệnh này để cài đặt các thư viện phụ thuộc khác:

```bash
pip install -r requirements.txt
```
### Bước 4: Tải trọng số SAM

Code này yêu cầu file trọng số của mô hình SAM...

1.  Tạo thư mục `checkpoints`:
    ```bash
    mkdir -p object_decomposer/checkpoints/
    ```
2.  Tải file trọng số `sam_vit_h_4b8939.pth` vào thư mục đó:
    ```bash
    wget [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) -P object_decomposer/checkpoints/
    ```
