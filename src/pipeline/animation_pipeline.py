import cv2
import numpy as np
from PIL import Image # Cần import thêm để load ảnh fallback

# Interfaces
from src.animator import IAnimator
from src.concept_decomposer import IObjectDecomposer
from src.image_style_transfer import IStyleTransfer
from src.pose_estimator import IPoseEstimator

from .input_data import AnimationPipelineInput

class AnimationGenerationPipeline:
    def __init__(
        self,
        style_transfer: IStyleTransfer = None,
        object_decomposer: IObjectDecomposer = None,
        pose_estimator: IPoseEstimator = None,
        animator: IAnimator = None,
    ):
        self.style_transfer = style_transfer
        self.object_decomposer = object_decomposer
        self.pose_estimator = pose_estimator
        self.animator = animator

    def _expand_bbox(self, bbox, pad=2):
        x1, y1, x2, y2 = bbox
        return (max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad)

    def run(self, input_data: AnimationPipelineInput, use_style_transfer: bool = True):
        # Load all structured input data
        data = input_data.load_data()
        results = {}

        # Mặc định lấy ảnh gốc
        current_image_np = np.array(data["content_image"])

        # ------------------------------------------------------------
        # Step 1: Style Transfer
        # ------------------------------------------------------------
        stylized_path = data["char_folder"] / "object_stylized.png"
        
        # Logic: Có module + Muốn dùng + File CHƯA tồn tại
        if self.style_transfer and use_style_transfer:
            print(">>> [Pipeline] Running Style Transfer...")
            object_stylized = self.style_transfer.transfer(
                style_ref=data["style_ref"],
                image=data["content_image"],
                prompt=data["prompt"],
            )
            object_stylized.save(stylized_path)
            
            current_image_np = np.array(object_stylized)
            results["object_stylized"] = object_stylized
            results["stylized_path"] = stylized_path
            
        elif stylized_path.exists() and use_style_transfer:
            # Nếu file đã có, load lên để dùng cho Step 2
            print(">>> [Pipeline] Style Transfer result exists. Loading from disk...")
            object_stylized = Image.open(stylized_path)
            current_image_np = np.array(object_stylized)
            results["stylized_path"] = stylized_path
        else:
            print(">>> [Pipeline] Skipping Style Transfer.")

        # ------------------------------------------------------------
        # Step 2: Segmentation + Crop
        # ------------------------------------------------------------
        image_crop_for_pose = None
        texture_path = data["char_folder"] / "texture.png"
        mask_path = data["char_folder"] / "mask.png"

        # Logic: Có module + File texture CHƯA tồn tại
        if self.object_decomposer and not texture_path.exists() and not mask_path.exists():
            print(">>> [Pipeline] Running Object Decomposition...")

            seg_result = self.object_decomposer.decompose(current_image_np)
            mask_image_viz = seg_result["mask_image_viz"]
            bbox = seg_result["bounding_box"]

            x1, y1, x2, y2 = self._expand_bbox(bbox, pad=4)
            mask_crop = mask_image_viz[y1:y2, x1:x2]
            image_crop = current_image_np[y1:y2, x1:x2]

            # Save to disk
            cv2.imwrite(str(mask_path), cv2.cvtColor(mask_crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(texture_path), cv2.cvtColor(image_crop, cv2.COLOR_RGB2BGR))

            results["segmentation"] = seg_result
            image_crop_for_pose = image_crop
        else:
            # Fallback: Load file đã tồn tại
            if texture_path.exists():
                print(">>> [Fallback] Found existing texture.png, mask.png loading...")
                loaded_crop = cv2.imread(str(texture_path))
                image_crop_for_pose = cv2.cvtColor(loaded_crop, cv2.COLOR_BGR2RGB)
            else:
                print(">>> [Warning] No texture found. Pose Estimation might fail.")

        # ------------------------------------------------------------
        # Step 3: Pose Estimation
        # ------------------------------------------------------------
        # LƯU Ý: Giữ nguyên Path object để check .exists()
        char_cfg_path = data["char_folder"] / "char_cfg.yaml"

        # Logic: Có module + File config CHƯA tồn tại
        if self.pose_estimator and not char_cfg_path.exists():
            if image_crop_for_pose is not None:
                print(">>> [Pipeline] Running Pose Estimation...")
                joint_overlay_path = data["char_folder"] / "joint_overlay.png"

                pose_result = self.pose_estimator.predict(
                    image_crop_for_pose,
                    output_file=str(joint_overlay_path), # Convert str khi truyền vào hàm
                    output_yaml=str(char_cfg_path),      # Convert str khi truyền vào hàm
                )

                results["pose"] = pose_result
                results["char_cfg_path"] = char_cfg_path
            else:
                print(">>> [Error] Input image crop is missing.")
        elif char_cfg_path.exists():
             print(">>> [Pipeline] char_cfg.yaml exists. Skipping Pose Estimation.")
             results["char_cfg_path"] = char_cfg_path

        # ------------------------------------------------------------
        # Step 4: Animation Rendering
        # ------------------------------------------------------------
        if self.animator:
            print(">>> [Pipeline] Checking Animations...")
            animation_results = {}

            for action in data["actions"]:
                # Giả định animator output ra file tên dạng {action}.gif hoặc .mp4
                # Bạn cần chỉnh lại extension tùy theo output thực tế của animator
                expected_output = data["char_folder"] / f"{action}.gif" 

                if not expected_output.exists():
                    try:
                        print(f">>> [Pipeline] Animating action: {action}")
                        animation_path = self.animator.animate(
                            action, data["char_folder"], data["char_name"]
                        )
                        animation_results[action] = animation_path
                    except Exception as e:
                        print(f">>> [Error] Failed to animate '{action}': {e}")
                else:
                    print(f">>> [Pipeline] Action '{action}' exists. Skipping.")
                    animation_results[action] = expected_output

            results["animations"] = animation_results

        return results