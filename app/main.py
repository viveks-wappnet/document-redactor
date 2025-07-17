from fastapi import FastAPI
from .routers import router

app = FastAPI(
    title="OCR Redaction Service",
    description="Upload an image, detect PII with GLiNER+EasyOCR, and receive a redacted image.")
app.include_router(router, prefix="/api")