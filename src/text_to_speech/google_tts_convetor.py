from .interface import ITextToSpeech
import requests
import base64
import subprocess
from pathlib import Path
import tempfile


GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

class GoogleTTSConvetor(ITextToSpeech):

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-preview-tts"):
        self.api_key = api_key
        self.model = model

    def convert(
        self,
        prompt: str,
        output_path: str = "output.wav",
        voice: str = "Zephyr",
        sample_rate: int = 24000
    ) -> Path:

        url = GEMINI_URL_TEMPLATE.format(model=self.model)

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice
                        }
                    }
                }
            }
        }

        # --- Call API ---
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # --- Extract base64 PCM ---
        audio_base64 = (
            data["candidates"][0]
            ["content"]["parts"][0]
            ["inlineData"]["data"]
        )

        pcm_bytes = base64.b64decode(audio_base64)

        output_path = Path(output_path)

        # --- Save temporary PCM ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcm") as tmp_pcm:
            tmp_pcm.write(pcm_bytes)
            pcm_path = Path(tmp_pcm.name)

        # --- Convert PCM → WAV using ffmpeg ---
        subprocess.run([
            "ffmpeg",
            "-y",
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-i", str(pcm_path),
            str(output_path)
        ], check=True)

        pcm_path.unlink(missing_ok=True)

        return output_path
