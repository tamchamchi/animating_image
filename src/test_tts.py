from src.text_to_speech.google_tts_convetor import GoogleTTSConvetor
from dotenv import load_dotenv
import os

load_dotenv()

tts = GoogleTTSConvetor(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = """
Tổng thư ký AFC, Windsor Paul John tiết lộ Việt Nam đã gửi khiếu nại sau trận thua 0-4 ở vòng loại Asian Cup 2027, khiến FIFA điều tra 7 cầu thủ nhập tịch Malaysia.
"""

output = tts.convert(prompt=prompt)