from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.app.api.endpoints import router
from src.app.core.config import settings

app = FastAPI(title="AI Character & Animation BE")

# Mount thư mục static để FE truy cập file kết quả
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn

    # Chạy server
    uvicorn.run("app.main:app", host="0.0.0.0", port=8501, reload=True)
