import logging
from contextlib import asynccontextmanager

import websockets

# Không cần import json ở đây nếu chỉ dùng trong router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.app.api.endpoints import create_api_router
from src.app.core.config import settings
from src.app.core.logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


async def websocket_handler(websocket: websockets.WebSocketClientProtocol):
    logger.info(f"WebSocket client connected: {websocket.remote_address}")
    settings.CONNECTEND_WEBSOCKET_CLIENTS.add(websocket)
    try:
        await websocket.wait_closed()
    except Exception as e:
        logger.error(f"WebSocket error for {websocket.remote_address}: {e}")
    finally:
        if websocket in settings.CONNECTEND_WEBSOCKET_CLIENTS:
            settings.CONNECTEND_WEBSOCKET_CLIENTS.remove(websocket)
        logger.info(
            f"WebSocket client disconnected: {websocket.remote_address}")


websocket_server_instance = None

# --- Lifespan Events ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI app startup: Starting WebSocket server...")
    global websocket_server_instance
    try:
        websocket_server_instance = await websockets.serve(
            websocket_handler,
            host=settings.SERVER_IP,
            port=settings.SERVER_PORT,
            ping_interval=None,
            ping_timeout=None,
        )
        logger.info(
            f"WebSocket server started on ws://{settings.SERVER_IP}:{settings.SERVER_PORT}")
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")

    yield

    logger.info("FastAPI app shutdown: Stopping WebSocket server...")
    if websocket_server_instance:
        websocket_server_instance.close()
        await websocket_server_instance.wait_closed()
        logger.info("WebSocket server gracefully stopped.")


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
try:
    app.mount("/api/static",
              StaticFiles(directory=settings.STATIC_DIR), name="static")
except Exception as e:
    logger.warning(
        f"Could not mount static files directory: {e}. Check settings.STATIC_DIR"
    )

# ===== API =====
app.include_router(
    create_api_router(
        connected_websocket_clients=settings.CONNECTEND_WEBSOCKET_CLIENTS
    ),
    prefix="/api",
)
