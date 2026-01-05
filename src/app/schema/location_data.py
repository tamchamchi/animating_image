from typing import List, Optional, Union
from pydantic import BaseModel, Field


class LocationData(BaseModel):
    id: Union[str, int] = "test-location-001"
    name: str = "Test Location"

    # Bounding box: [x_min, y_min, x_max, y_max]
    bbox: Optional[List[float]] = Field(default_factory=lambda: [0.1, 0.1, 0.9, 0.9])

    audio_base64: Optional[str] = Field(
        None, description="Audio data encoded in Base64 (e.g., MP3, WAV)"
    )

    audio_format: Optional[str] = Field(
        "audio/mpeg",
        description="MIME type of the audio (e.g., 'audio/mpeg', 'audio/wav')",
    )
