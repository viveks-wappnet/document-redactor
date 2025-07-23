import io
from io import BytesIO
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageDraw
import pytesseract
from pytesseract import Output
from collections import defaultdict

class RedactionService:
    """Service for detecting and redacting PII in images using OCR and GLiNER models."""

    def __init__(self, gliner_model, confidence_threshold: int = 60, box_padding: int = 5):
        """
        Initialize the redaction service.
        
        Args:
            gliner_model: Pre-loaded GLiNER model instance.
            confidence_threshold: Minimum OCR confidence score (default: 60).
            box_padding: Pixels to add around bounding boxes (default: 5).
        """
        if not gliner_model:
            raise ValueError("GLiNER model must be pre-loaded and provided during initialization.")
        self._gliner = gliner_model
        self.confidence_threshold = confidence_threshold
        self.box_padding = box_padding

    def detect_text_and_boxes(self, image_bytes: bytes) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        pil_img = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        try:
            data = pytesseract.image_to_data(
                pil_img,
                config=r'--oem 3 --psm 4',
                output_type=Output.DICT
            )
        except Exception as e:
            raise RuntimeError(f"OCR processing failed: {str(e)}")

        lines = defaultdict(lambda: {'words': [], 'boxes': []})
        # Iterate over the OCR data using zip for better readability
        for block_num, par_num, line_num, word_text, conf_score, left, top, width, height in zip(
            data['block_num'], data['par_num'], data['line_num'], data['text'], data['conf'],
            data['left'], data['top'], data['width'], data['height']
        ):
            word = word_text.strip()
            conf = conf_score
            if isinstance(conf, str):
                conf = int(conf) if conf.isdigit() else -1
            elif not isinstance(conf, int):
                conf = -1

            if not word or conf < self.confidence_threshold:
                continue

            key = (block_num, par_num, line_num)
            lines[key]['words'].append(word)
            x, y, w, h = (left, top, width, height)
            lines[key]['boxes'].append((x, y, w, h))

        result = [
            {'text': ' '.join(lines[key]['words']), 'boxes': lines[key]['boxes']}
            for key in lines
        ]
        return pil_img, result

    def classify_pii(self, lines: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
        """Return bounding boxes of text identified as PII."""
        pii_boxes = []
        for entry in lines:
            text = entry["text"]
            boxes = entry["boxes"]
            try:
                ents = self._gliner.predict_entities(
                    text,
                    labels=["PERSON", "EMAIL", "PHONE", "ADDRESS"]
                )
                if any(e["label"] in {"PERSON", "EMAIL", "PHONE", "ADDRESS"} for e in ents):
                    pii_boxes.extend(boxes)
            except Exception as e:
                print(f"Error classifying PII in '{text}': {e}")
        return pii_boxes

    def redact_image(self, image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
        """Redact specified regions in the image with black rectangles."""
        image = image.convert("RGB")  # Ensure RGB mode for drawing
        draw = ImageDraw.Draw(image)
        for x, y, w, h in boxes:
            draw.rectangle([x, y, x + w, y + h], fill="black")
        return image

    def process_redaction(self, image_bytes: bytes) -> Dict[str, Any]:
        """Process an image to detect and redact PII, returning redacted image and boxes."""
        try:
            img, lines = self.detect_text_and_boxes(image_bytes)
            pii_boxes = self.classify_pii(lines)
            redacted = self.redact_image(img.copy(), pii_boxes)

            buf = io.BytesIO()
            redacted.save(buf, format="PNG")
            return {
                "boxes": pii_boxes,
                "redacted_image": buf.getvalue()
            }
        except Exception as e:
            raise RuntimeError(f"Redaction processing failed: {str(e)}")
