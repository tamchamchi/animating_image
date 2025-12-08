import os
import cv2
import numpy as np
import torch
import supervision as sv
from pathlib import Path
from typing import List, Dict, Any
from supervision.draw.color import ColorPalette

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .interface import IObjectDecomposer


class GroundedSAMDecomposer(IObjectDecomposer):
    """
    Combines Grounding DINO (text-based object detection) and SAM2 (segmentation)
    to decompose an image into labeled object masks.
    """

    def __init__(
        self,
        sam2_checkpoint: Path = "./checkpoints/sam2.1_hiera_large.pt",
        sam2_config: Path = "configs/sam2.1/sam2.1_hiera_l.yaml",
        gd_model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cpu",
    ):
        super().__init__()
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.gd_model_name = gd_model_name
        self.device = device

        (
            self.sam2_model,
            self.sam2_predictor,
            self.gd_model,
            self.gd_processor,
        ) = self.__init_model()

    # ---------------------------------------------------------------
    def __init_model(self):
        """Initialize SAM2 and Grounding DINO models."""
        try:
            sam2_model = build_sam2(
                self.sam2_config, self.sam2_checkpoint, device=self.device
            )
            sam2_predictor = SAM2ImagePredictor(sam2_model)

            gd_processor = AutoProcessor.from_pretrained(self.gd_model_name)
            gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.gd_model_name
            ).to(self.device)

            return sam2_model, sam2_predictor, gd_model, gd_processor
        except Exception as e:
            raise RuntimeError(f"Error initializing models: {str(e)}")

    # ---------------------------------------------------------------
    @torch.no_grad()
    def decompose(
        self,
        prompt_i: List[str],
        image: np.ndarray,
        visualize: bool = False,
        output_dir: str = "outputs",
        custom_color_map: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect and segment objects in an image using text prompts.

        Returns:
            dict: A dictionary containing information for each object, including:
                - image: the cropped image of the object
                - mask: the segmentation mask of the object
                - bounding_box: the bounding box coordinates of the object
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            # --- 1. Grounding DINO detection ---
            text_prompt = " . ".join(prompt_i)
            print(f"🔍 Text prompt: {text_prompt}")
            inputs = self.gd_processor(
                images=image, text=text_prompt, return_tensors="pt"
            ).to(self.device)

            outputs = self.gd_model(**inputs)
            results = self.gd_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.25,
                target_sizes=[image.shape[:2]],
            )[0]

            boxes = results["boxes"].cpu().numpy()
            labels = results["labels"]
            scores = results["scores"].cpu().numpy()
            class_ids = np.arange(len(labels))

            if len(boxes) == 0:
                print("⚠️ No objects detected.")
                return {}

            # --- 2. SAM2 segmentation ---
            self.sam2_predictor.set_image(image)
            objects_info = {}
            masks = []

            for i, box in enumerate(boxes):
                x0, y0, x1, y1 = box.astype(int)
                mask, _, _ = self.sam2_predictor.predict(box=np.array([x0, y0, x1, y1]))
                mask = mask[0].astype(np.uint8)
                masks.append(mask)

                # Cắt vùng ảnh theo mask
                cropped = image.copy()
                cropped[mask == 0] = 0
                crop_region = cropped[y0:y1, x0:x1]

                label = labels[i].lower().strip()
                obj_name = f"{label}_{i}"

                objects_info[obj_name] = {
                    "image": crop_region,
                    "mask": mask,
                    "bounding_box": [int(x0), int(y0), int(x1), int(y1)],
                    "score": float(scores[i]),
                    "label": label,
                }

            # --- 3. Visualization ---
            if visualize:
                self._visualize_results(
                    image=image,
                    input_boxes=boxes,
                    labels=labels,
                    class_ids=class_ids,
                    masks=np.array(masks),
                    output_dir=output_dir,
                    objects_info=objects_info,
                    custom_color_map=custom_color_map,
                )

            print(f"✅ {len(objects_info)} objects decomposed successfully.")
            return objects_info

        except Exception as e:
            raise RuntimeError(f"❌ Error during decomposition: {str(e)}") from e
    
    # ---------------------------------------------------------------
    def _visualize_results(
        self,
        image: np.ndarray,
        input_boxes: np.ndarray,
        labels: List[str],
        class_ids: np.ndarray,
        masks: np.ndarray,
        output_dir: str,
        objects_info: Dict[str, Any],
        custom_color_map: List[str] = None,
    ):
        """Visualize and save results including per-object textures and masks."""
        img = image.copy()

        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
        )

        color_palette = (
            ColorPalette.from_hex(custom_color_map)
            if custom_color_map
            else ColorPalette.DEFAULT
        )

        # --- Combined visualization ---
        box_annotator = sv.BoxAnnotator(color=color_palette)
        annotated = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=color_palette)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        mask_annotator = sv.MaskAnnotator(color=color_palette)
        annotated_with_mask = mask_annotator.annotate(scene=annotated, detections=detections)

        # --- Save combined visual ---
        cv2.imwrite(
            os.path.join(output_dir, "grounded_combined_visual.jpg"),
            cv2.cvtColor(annotated_with_mask, cv2.COLOR_RGB2BGR),
        )

        # --- Save combined binary mask ---
        combined_mask = np.any(masks.astype(bool), axis=0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, "mask_combined.jpg"), combined_mask)

        # --- Save per-object textures & masks ---
        for obj_name, obj_data in objects_info.items():
            obj_dir = os.path.join(output_dir, obj_name)
            os.makedirs(obj_dir, exist_ok=True)

            # Save mask
            mask_path = os.path.join(obj_dir, "mask.png")
            cv2.imwrite(mask_path, obj_data["mask"].astype(np.uint8) * 255)

            # Save texture
            texture_path = os.path.join(obj_dir, "texture.png")
            cropped_img = obj_data["image"]
            cv2.imwrite(texture_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

            obj_data["mask_path"] = mask_path
            obj_data["texture_path"] = texture_path

        print(f"✅ Visualization & textures saved to: {output_dir}")