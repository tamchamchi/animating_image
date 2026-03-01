import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from .interface import IFaceSegmenter


class SegFormerB5FaceSegmenter(IFaceSegmenter):
    """
    Face parser using SegFormer MIT-B5 pretrained on CelebAMask-HQ.
    Model: jonathandinu/face-parsing
    """

    def __init__(self, device=None):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.processor = SegformerImageProcessor.from_pretrained(
            "jonathandinu/face-parsing"
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "jonathandinu/face-parsing"
        ).to(self.device)
        self.model.eval()

        self.face_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

    # ============================================================
    #                   HELPER FUNCTIONS
    # ============================================================

    @staticmethod
    def extract_face_bbox(face_mask: np.ndarray):
        """
        Input:
            face_mask shape (H, W) with values 0 or 255

        Output:
            bbox = (x1, y1, x2, y2)
        """
        ys, xs = np.where(face_mask > 0)
        if len(xs) == 0:
            return None

        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()

        return (x1, y1, x2, y2)

    @staticmethod
    def crop_face_by_bbox(image: np.ndarray, bbox):
        """
        image: np.ndarray (H, W, 3)
        bbox: (x1, y1, x2, y2)

        return cropped_face (np.ndarray)
        """
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    # ============================================================
    #                     SEGMENT FUNCTION
    # ============================================================

    def segment(self, image: np.ndarray):
        """
        Return:
            face_mask   - uint8 mask (0/255)
            face_region - masked face (same size as input)
            bbox        - (x1, y1, x2, y2)
            face_crop   - cropped using bbox
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            raise ValueError("image must be numpy array")

        H, W = image.shape[:2]

        # preprocess
        inputs = self.processor(
            images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # resize segmentation output
        upsampled = nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        labels = upsampled.argmax(dim=1)[0].cpu().numpy()

        # mask
        face_mask = np.isin(labels, list(self.face_ids)).astype(np.uint8) * 255

        # region (same size as image)
        face_mask_3 = np.stack([face_mask] * 3, axis=-1)
        face_region = image * (face_mask_3 > 0)

        # ---- NEW: compute bbox ----
        bbox = self.extract_face_bbox(face_mask)

        # ---- NEW: get cropped region ----
        face_crop = self.crop_face_by_bbox(face_region, bbox)

        return face_mask, face_region, bbox, face_crop

    # ============================================================
    #                     APPLY FACE TO BODY
    # ============================================================

    def apply_to_body(
        self,
        body_image: np.ndarray,
        face_region: np.ndarray,
        x_offset: int = 5,
        y_offset: int = 15,
        ratio: float = 0.6
    ):
        body_pil = Image.fromarray(body_image).convert("RGBA")
        face_pil = Image.fromarray(face_region).convert("RGBA")

        # --- Scale ---
        face_w, face_h = face_pil.size
        new_w, new_h = int(face_w * ratio), int(face_h * ratio)
        face_pil = face_pil.resize((new_w, new_h), Image.LANCZOS)

        # --- detect alpha for head ---
        face_np = np.array(face_region)
        alpha_mask = np.any(face_np > 0, axis=-1).astype(np.uint8) * 255
        alpha_mask = Image.fromarray(alpha_mask).resize((new_w, new_h))
        face_pil.putalpha(alpha_mask)

        face_w, face_h = face_pil.size
        body_w, body_h = body_pil.size

        # ===== AUTO FIND HEAD ANCHOR =====
        body_np = np.array(body_pil)

        ys = np.where(np.any(body_np < 255, axis=-1))[0]
        if len(ys) == 0:
            first_pixel_y = body_h // 4
        else:
            first_pixel_y = ys[0]

        ax = body_w // 2 - x_offset
        ay = first_pixel_y - face_h // 2 + y_offset

        tx = ax - face_w // 2
        ty = ay - face_h // 2

        pad_left = max(0, -tx)
        pad_top = max(0, -ty)
        pad_right = max(0, tx + face_w - body_w)
        pad_bottom = max(0, ty + face_h - body_h)

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            new_w = body_w + pad_left + pad_right
            new_h = body_h + pad_top + pad_bottom
            canvas = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 0))
            canvas.paste(body_pil, (pad_left, pad_top))
            body_pil = canvas
            tx += pad_left
            ty += pad_top

        body_pil.paste(face_pil, (tx, ty), face_pil)

        return np.array(body_pil.convert("RGB"))
