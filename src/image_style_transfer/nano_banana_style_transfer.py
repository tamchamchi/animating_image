import requests
import base64
from io import BytesIO
from PIL import Image
from .interface import IStyleTransfer


GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class NanoBananaStyleTransfer(IStyleTransfer):
    """
    Style transfer using Google Gemini 2.5 Flash Image.
    Supports:
        - Image-only: style_ref + content image
        - Prompt-only: style_ref + prompt
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-image"):
        super().__init__()
        if not api_key:
            raise ValueError(
                "API key must be provided either via .env or parameter")
        self.api_key = api_key
        self.model_name = model_name

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def transfer(self, style_ref: Image.Image, image: Image.Image = None, prompt: str = None) -> Image.Image:
        if image is None and prompt is None:
            raise ValueError("Either 'image' or 'prompt' must be provided.")

        parts = []

        # Style reference
        parts.append({"text": "Style reference image:"})
        parts.append({
            "inlineData": {"mimeType": "image/png", "data": self._encode_image(style_ref)}
        })

        # Content image (optional)
        if image is not None:
            parts.append(
                {"text": "Content image to recreate in the above style:"})
            parts.append({
                "inlineData": {"mimeType": "image/png", "data": self._encode_image(image)}
            })

        # Prompt (optional)
        if prompt is not None:
            parts.append({"text": prompt})

        generation_cfg = {
            "response_modalities": ['Image'],
            "imageConfig": {
                "aspectRatio": "1:1",
            }
        }

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_cfg
        }

        url = GEMINI_URL_TEMPLATE.format(model=self.model_name)

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Parse output image
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if "inlineData" in part:
                    img_b64 = part["inlineData"]["data"]
                    img_bytes = base64.b64decode(img_b64)
                    return Image.open(BytesIO(img_bytes))

        raise RuntimeError("Model returned no image output.")
