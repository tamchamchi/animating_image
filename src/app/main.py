from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.app.api.endpoints import router
from src.app.core.config import settings

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
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# ===== API =====
app.include_router(router, prefix="/api/v1")