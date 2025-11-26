from .interface import IImgeGenerator
import requests
import base64
from io import BytesIO
from PIL import Image

# Template for the Google Gemini API endpoint
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class NanoBananaGenerator(IImgeGenerator):
    """
    Concrete implementation of IImgeGenerator using Google's Gemini API.

    This class interacts with the Generative Language API (Gemini) to generate
    or transform images based on text prompts and optional reference images.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-image"):
        """
        Initializes the Gemini-based generator.

        Args:
            api_key (str): The Google Cloud API key for authentication.
            model_name (str): The specific model version to target 
                              (default: "gemini-2.5-flash-image").

        Raises:
            ValueError: If the api_key is empty or None.
        """
        super().__init__()
        if not api_key:
            raise ValueError(
                "API key must be provided either via .env or parameter")
        self.api_key = api_key
        self.model_name = model_name

    def _encode_image(self, image: Image.Image) -> str:
        """
        Helper method to encode a PIL Image into a Base64 string.

        Args:
            image (Image.Image): The input PIL image.

        Returns:
            str: The Base64 encoded string of the image in PNG format.
        """
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate(self, prompt: str = None, style_ref: Image.Image = None) -> Image.Image:
        """
        Generates content (typically an image) by sending a prompt and an 
        optional style reference image to the Gemini API.

        Args:
            prompt (str): The text description for the generation.
            style_ref (Image.Image, optional): An optional reference image to 
                                               guide the style or content.

        Returns:
            Image.Image: The generated image returned by the model.

        Raises:
            ValueError: If the prompt is not provided.
            requests.exceptions.HTTPError: If the API request fails.
            RuntimeError: If the model completes but returns no image data.
        """
        if prompt is None:
            raise ValueError("Either 'prompt' must be provided.")

        parts = []

        # Process the style reference image if provided
        if style_ref is not None:
            parts.append({"text": "Style reference image:"})
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": self._encode_image(style_ref)
                }
            })

        # Add the text prompt to the payload
        if prompt is not None:
            parts.append({"text": prompt})

        generation_cfg = {
            "response_modalities": ['Image'],
            "imageConfig": {
                "aspectRatio": "1:1",
            }
        }

        # Construct the API payload
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_cfg
        }

        url = GEMINI_URL_TEMPLATE.format(model=self.model_name)

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Send the POST request to Google's API
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for 4xx/5xx errors
        data = response.json()

        # Parse the response to extract the output image
        # Gemini API returns candidates -> content -> parts -> inlineData
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if "inlineData" in part:
                    img_b64 = part["inlineData"]["data"]
                    img_bytes = base64.b64decode(img_b64)
                    return Image.open(BytesIO(img_bytes))

        # If we loop through candidates and find no image data
        raise RuntimeError("Model returned no image output.")
