# main.py
import logging
from contextlib import asynccontextmanager  # Thêm import này

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import router VÀ websocket_service
from src.app.api.endpoints import router, websocket_service
from src.app.core.config import settings
from src.app.core.logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)  # Logger cho main.py

# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "FastAPI app startup: Ensuring initial WebSocket connection to third party...")
    try:
        await websocket_service._ensure_connection()
    except Exception as e:
        logger.warning(f"FastAPI app startup: Initial WebSocket connection to third party failed: {e}. "
                       "Will retry on first data send.")
    yield
    logger.info(
        "FastAPI app shutdown: Closing WebSocket connection to third party...")
    await websocket_service.close_connection()

app = FastAPI(title="AI Character & Animation BE", lifespan=lifespan)

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== STATIC FILES =====
app.mount("/api/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# ===== API =====
app.include_router(router, prefix="/api")
