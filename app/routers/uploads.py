from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.database import get_db, SessionLocal
from ..models import Upload, Page, UploadStatus
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
from uuid import UUID

router = APIRouter()

def process_pdf_in_background(raw_pdf: bytes, upload_id: UUID, filename: str):
    """
    Processes a PDF in the background: converts pages to images and stores them in the database.

    Updates upload status to 'PROCESSING', then 'DONE' on success, or 'FAILED' on error.
    Each page is converted to a JPEG image and saved.
    """
    # Each background task needs its own database session.
    db = SessionLocal()
    print(f"Background task started for upload_id: {upload_id}")
    
    # Get the specific upload record to update its status
    upload_record = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload_record:
        print(f"Error: Could not find upload_id {upload_id} to update status.")
        db.close()
        return

    try:
        # 1. Set status to PROCESSING to indicate work has started
        upload_record.status = UploadStatus.PROCESSING
        db.commit()
        print(f"Upload status for {upload_id} set to PROCESSING.")

        # 2. Open PDF from bytes using PyMuPDF
        pdf_document = fitz.open(stream=raw_pdf, filetype="pdf")
        
        pages_to_add = []
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=95)
            img_bytes = buf.getvalue()

            page_model = Page(
                upload_id=upload_id,
                page_number=page_number + 1,
                img_bytes=img_bytes
            )
            pages_to_add.append(page_model)

        # 3. Bulk add all created Page objects and commit
        db.add_all(pages_to_add)
        db.commit()

        # 4. Update the upload status to DONE
        upload_record.status = UploadStatus.DONE
        db.commit()
        print(f"Background task finished for upload_id: {upload_id}. Status set to DONE.")

    except Exception as e:
        # On error, rollback and update the status to FAILED
        db.rollback()
        upload_record.status = UploadStatus.FAILED
        db.commit()
        print(f"Error processing {filename} for upload_id {upload_id}: {e}. Status set to FAILED.")
    finally:
        db.close()


@router.post("/upload", summary="Upload a PDF and process it in the background")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="A multi-page PDF"),
    db: Session = Depends(get_db)
):
    """
    Accepts a PDF upload, creates a 'PENDING' record, and queues background processing.

    Validates file type, saves initial upload record, and adds PDF-to-image conversion
    to background tasks for asynchronous processing.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")
    
    raw_pdf = await file.read()

    # Create an Upload record with an initial 'PENDING' status.
    upload = Upload(filename=file.filename, status=UploadStatus.PENDING)
    db.add(upload)
    db.commit()
    db.refresh(upload)

    # Add the heavy processing task to run in the background.
    background_tasks.add_task(process_pdf_in_background, raw_pdf, upload.id, file.filename)

    return {
        "message": "PDF upload accepted and is being processed in the background.",
        "upload_id": str(upload.id),
        "filename": file.filename
    }
