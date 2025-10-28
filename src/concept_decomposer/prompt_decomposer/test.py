from dotenv import load_dotenv
import os
from zero_shot import ZeroShotPromptDecomposer
from few_shot import FewShotPromptDecomposer

# Load biến môi trường từ file .env
load_dotenv()

prompt = input("prompt: ")

api_key = os.environ.get("GOOGLE_API_KEY")
decomposer_instance = ZeroShotPromptDecomposer(api_key)

objects_one_shot = decomposer_instance.decompose(prompt)

print("\n🟢 Prompt:", prompt)
print("🔹 Decomposed objects with one-shot prompting:", objects_one_shot)

decomposer = FewShotPromptDecomposer(api_key)
objects_few_shot = decomposer.decompose(prompt)

print("🔹 Decomposed objects with few-shot prompting:", objects_few_shot)
