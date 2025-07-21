from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import init_db 
from app.routers.uploads import router as upload_router
from app.routers.pages import router as pages_router
from app.core import get_model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles ML model initialization and database setup during app startup."""
    print("Application startup: Initializing models...")
    get_model_manager().initialize()
    print("Application startup: Models initialized successfully.")
    
    init_db()
    
    yield
    
    print("Application shutdown.")

app = FastAPI(
    title="OCR Redaction Service",
    description="Upload an image, detect PII with GLiNER+EasyOCR, and receive a redacted image.",
    lifespan=lifespan
)

app.include_router(upload_router, tags=["upload"])
app.include_router(pages_router, tags=["pages"])
