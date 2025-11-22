from interface import IPromptDecomposer
import google.generativeai as genai


class FewShotPromptDecomposer(IPromptDecomposer):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro-latest")

    def decompose(self, prompt: str) -> list[str]:
        """
        Decompose input prompt using few-shot examples.
        Keeps both concrete objects and scene/environment elements.
        """
        system_prompt = """
You are a visual scene analyzer.
Extract and list all main physical objects, people, animals, places, and scene components visible in an image.
Return a clean Python list of lowercase strings. 
Do NOT explain anything, only return the list.

Include:
- People or animals (man, cat, bird, artist)
- Objects (table, candle, brush)
- Natural elements (sunset, cloud, forest, mountain)
- Scene components or locations (view, café, rooftop, park, beach)

Examples:
Input: "A man walking on the beach with a dog"
Output: ["man", "beach", "dog"]

Input: "A woman reading a book in a quiet library with wooden shelves"
Output: ["woman", "book", "library", "shelves"]

Input: "A couple watching fireworks above a city skyline at night"
Output: ["couple", "fireworks", "city", "skyline", "night"]

"""

        try:
            response = self.model.generate_content(system_prompt + prompt)
            text = response.text.strip()

            if "[" in text and "]" in text:
                list_str = text[text.find("["):text.find("]") + 1]
                try:
                    result = eval(list_str)
                    if isinstance(result, list):
                        return [str(x).lower().strip() for x in result]
                except Exception:
                    pass

            # fallback
            return [w.lower() for w in prompt.split() if w.isalpha()]

        except Exception as e:
            print("⚠️ Gemini API error:", e)
            return []
