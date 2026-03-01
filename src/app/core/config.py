import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Global application settings loaded from environment variables.

    This class acts as a single source of truth for runtime configuration.
    All critical paths and credentials are validated at initialization time
    to ensure the application fails fast if required environment variables
    are missing or misconfigured.

    Attributes:
        GEMINI_API_KEY (str):
            API key used to authenticate requests to Google's Gemini services.

        BASE_DIR (str):
            Root directory for all persistent storage. Must be provided via
            the STORAGE_ROOT environment variable.

        STATIC_DIR (str):
            Directory used to store static resources such as characters,
            animations, and assets.

        ASSETS_DIR (str):
            Subdirectory containing shared asset files.

        MMPOSE_CONFIG (str):
            Filesystem path to the MMPose model configuration file.

        MMPOSE_CHECKPOINT (str):
            Filesystem path to the MMPose model checkpoint file.

        MAX_CONCURRENT_TASKS (int):
            Maximum number of concurrent tasks allowed by the application.

        DEVICE (str):
            Target computation device (e.g., "cpu" or "cuda").

        THIRD_PARTY_WEBSOCKET_URL (str | None):
            Optional WebSocket endpoint used to communicate with external
            third-party services.

        CONNECTEND_WEBSOCKET_CLIENTS (set):
            Runtime registry of currently connected WebSocket clients.
            This is typically populated dynamically during server execution.

        SERVER_IP (str | None):
            IP address on which the server is running.

        SERVER_PORT (str | None):
            Port number exposed by the server.
    """

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

    # ===== WEBSOCKET SETTINGS =====
    THIRD_PARTY_WEBSOCKET_URL = os.getenv("THIRD_PARTY_WEBSOCKET_URL")
    CONNECTEND_WEBSOCKET_CLIENTS = set()

    SERVER_IP = os.getenv("SERVER_IP")
    SERVER_PORT = os.getenv("SERVER_PORT")


settings = Settings()

# ===== CREATE STATIC FOLDERS =====
"""
Ensure required static directories exist at startup.

These directories are created eagerly to prevent runtime failures when
writing character data, animation outputs, or shared assets.
"""
os.makedirs(os.path.join(settings.STATIC_DIR, "characters"), exist_ok=True)
os.makedirs(os.path.join(settings.STATIC_DIR, "animations"), exist_ok=True)
os.makedirs(os.path.join(settings.STATIC_DIR, "assets"), exist_ok=True)
