import os
import io
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
import easyocr
from huggingface_hub import login
from dotenv import load_dotenv
from gliner import GLiNER

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Initialize OCR and NER models
easyocr_reader = easyocr.Reader(['en'], gpu=False)
gliner = GLiNER.from_pretrained(os.getenv("GLINER_MODEL_NAME"))

def detect_text_and_boxes(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raw = easyocr_reader.readtext(img_rgb)

    words = []
    for bbox, text, conf in raw:
        x_min = int(min(p[0] for p in bbox))
        y_min = int(min(p[1] for p in bbox))
        x_max = int(max(p[0] for p in bbox))
        y_max = int(max(p[1] for p in bbox))
        words.append({
            "word": text,
            "box": [x_min, y_min, x_max - x_min, y_max - y_min]
        })
    img_pil = Image.fromarray(img_rgb)
    return img_pil, words

def classify_pii(words):
    pii_spans = []
    for idx, w in enumerate(words):
        ents = gliner.predict_entities(
            w["word"],
            labels=["PERSON","EMAIL","PHONE","ID","DATE","ORG"]
        )
        for ent in ents:
            if ent["label"] in {"PERSON","EMAIL","PHONE","ID"}:
                pii_spans.append(w["box"])
    return pii_spans

def redact_image(image: Image.Image, boxes: list, method: str = 'black') -> Image.Image:
    """
    Redact regions on the image. Supports boxes in two formats:
      - [x, y, w, h]
      - polygon list of 4 (x,y) points
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        # Normalize box format
        if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(coord, (int, float)) for coord in box):
            x, y, w, h = box
        elif isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box):
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
        else:
            # unsupported format, skip
            continue

        if method == 'blur':
            region = image.crop((x, y, x + w, y + h))
            image.paste(region.filter(ImageFilter.GaussianBlur(radius=10)), (x, y))
        else:
            draw.rectangle((x, y, x + w, y + h), fill='black')
    return image

def process_redaction(image_bytes: bytes, method: str = 'black') -> dict:
    img, words = detect_text_and_boxes(image_bytes)
    pii_boxes = classify_pii(words)
    redacted = redact_image(img, pii_boxes, method)
    buf = io.BytesIO()
    redacted.save(buf, format='PNG')
    return {
        'boxes': pii_boxes,
        'redacted_image': buf.getvalue()
    }

# if __name__ == "__main__":
#     # Read the test image
#     image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shipping_billing.png")
    
#     with open(image_path, "rb") as f:
#         image_bytes = f.read()
    
#     # import ipdb; ipdb.set_trace()  
#     result = process_redaction(image_bytes, method="black")
    
#     # Save the redacted image
#     output_path = os.path.join(
#         os.path.dirname(image_path),
#         f"redacted_black_{os.path.basename(image_path)}"
#     )
    
#     with open(output_path, "wb") as f:
#         f.write(result["redacted_image"])
    
#     print(f"Redacted image saved to: {output_path}")
#     print(f"Found {len(result['boxes'])} PII regions")