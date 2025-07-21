from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app.database import get_db
from app.models import Page, RedactedPage, Upload
from app.redactor import RedactionService
from app.core import get_model_manager

router = APIRouter()

def get_redaction_service() -> RedactionService:
    """
    Provides a RedactionService instance with pre-loaded ML models.

    Ensures EasyOCR reader and GLiNER model are available; raises 503 if not.
    """
    manager = get_model_manager()
    if not manager.easyocr_reader or not manager.gliner:
        raise HTTPException(
            status_code=503, 
            detail="Models are not available or initialized. Please check server logs."
        )
    return RedactionService(
        reader=manager.easyocr_reader, 
        gliner_model=manager.gliner
    )

@router.post("/redact/{upload_id}", summary="Redact all pages in an upload")
def redact_upload(
    upload_id: UUID,
    db: Session = Depends(get_db),
    redactor: RedactionService = Depends(get_redaction_service),
):
    """
    Processes all pages in an upload to identify and redact PII.

    Retrieves pages, applies redaction, saves redacted images, and returns a summary.
    Raises 404 if upload or pages not found.
    """
    upload = db.query(Upload).get(upload_id)
    if not upload:
        raise HTTPException(404, "Upload not found")
    
    pages = db.query(Page).filter(Page.upload_id == upload_id).order_by(Page.page_number).all()
    if not pages:
        raise HTTPException(404, "No pages found for this upload")

    results = []
    total_boxes = 0
    
    for page in pages:
        # The 'redactor' instance is now provided by the Depends() system,
        # fully equipped with the pre-loaded models.
        result = redactor.process_redaction(page.img_bytes)
        total_boxes += len(result["boxes"])
        
        redacted = RedactedPage(
            page_id=page.id,
            redacted_bytes=result["redacted_image"]
        )
        db.merge(redacted)
        
        results.append({
            "page_id": page.id,
            "page_number": page.page_number,
            "boxes_found": len(result["boxes"])
        })
    
    db.commit()

    return {
        "upload_id": str(upload_id),
        "total_pages": len(pages),
        "total_boxes_redacted": total_boxes,
        "pages": results
    }
