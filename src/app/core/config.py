import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ===== API KEYS =====
    GEMINI_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # ===== PATHS =====
    BASE_DIR = os.getenv("STORAGE_ROOT")
    if not BASE_DIR:
        raise RuntimeError("STORAGE_ROOT is not set")

    STATIC_DIR = os.path.join(BASE_DIR, "static")
    ASSETS_DIR = os.path.join(STATIC_DIR, "assets")

    # ===== MODEL PATHS =====
    MMPOSE_CONFIG = os.getenv("POSE_MODEL_CFG_PATH")
    MMPOSE_CHECKPOINT = os.getenv("POSE_MODEL_CKPT_PATH")

    if not MMPOSE_CONFIG or not MMPOSE_CHECKPOINT:
        raise RuntimeError("Pose model paths are not set")

    # ===== CONCURRENCY =====
    MAX_CONCURRENT_TASKS: int = int(os.getenv("MAX_CONCURRENT_TASKS", 2))
    DEVICE: str = os.getenv("DEVICE", "cpu")


settings = Settings()

# ===== CREATE STATIC FOLDERS =====
os.makedirs(os.path.join(settings.STATIC_DIR, "characters"), exist_ok=True)
os.makedirs(os.path.join(settings.STATIC_DIR, "animations"), exist_ok=True)
os.makedirs(os.path.join(settings.STATIC_DIR, "assets"), exist_ok=True)
