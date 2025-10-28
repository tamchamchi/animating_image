# zero_shot.py
from interface import IPromptDecomposer
import google.generativeai as genai

class ZeroShotPromptDecomposer(IPromptDecomposer):
    def __init__(self, api_key: str):
        # Cấu hình Gemini API
        genai.configure(api_key=api_key)
        # Dùng model hiện hoạt
        self.model = genai.GenerativeModel("gemini-pro-latest")

    def decompose(self, prompt: str) -> list[str]:
        """
        Decompose the input prompt into a list of main objects.
        Zero-shot approach using Gemini API (gemini-pro-latest).
        """
        system_prompt = (
            "You are a visual scene analyzer. "
            "Extract and list all main objects or entities from the user's prompt describing an image. "
            "Return only a clean Python list of lowercase strings without explanations.\n\n"
            "Example:\n"
            "Input: 'A man walking on the beach with a dog'\n"
            "Output: ['man', 'beach', 'dog']\n\n"
            f"Now analyze the following:\nPrompt: {prompt}\nOutput:"
        )

        try:
            response = self.model.generate_content(system_prompt)
            text = response.text.strip()

            # Parse nếu Gemini trả về list
            if "[" in text and "]" in text:
                list_str = text[text.find("["):text.find("]")+1]
                try:
                    result = eval(list_str)
                    if isinstance(result, list):
                        return [str(x).lower().strip() for x in result]
                except Exception:
                    pass

            # fallback nếu không parse được
            return [w.lower() for w in prompt.split() if w.isalpha()]

        except Exception as e:
            print("⚠️ Gemini API error:", e)
            return []
