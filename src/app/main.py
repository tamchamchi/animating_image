import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.app.api.endpoints import router
from src.app.core.config import settings
from src.app.core.logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Character & Animation BE")

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
