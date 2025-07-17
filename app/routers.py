from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from .redactor import process_redaction

router = APIRouter()

@router.post("/redact")
async def redact_endpoint(
    image: UploadFile = File(...),
    method: str = 'black'
):
    """
    Upload an image and get back a redacted version.
    
    Args:
        image: Image file to process
        method: Redaction method ('black' or 'blur')
    Returns:
        PNG image containing the redacted version
    """
    if method not in ['black', 'blur']:
        raise HTTPException(
            status_code=400,
            detail="Method must be either 'black' or 'blur'"
        )
    
    try:
        # Read image bytes
        contents = await image.read()
        
        # Process redaction
        result = process_redaction(contents, method)
        
        # Return the image as a PNG response
        return Response(
            content=result['redacted_image'],
            media_type="image/png",
            headers={
                "X-PII-Boxes-Found": str(len(result['boxes']))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))