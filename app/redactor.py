import io
from typing import Tuple, List
import cv2
import numpy as np
from PIL import Image, ImageDraw

class RedactionService:
    """Service for detecting and redacting PII in images using OCR and GLiNER models."""

    def __init__(self, reader, gliner_model):
        """
        Args:
            reader: Initialized EasyOCR reader
            gliner_model: Initialized GLiNER model
        """
        if not reader or not gliner_model:
            raise ValueError("Models must be pre-loaded and provided during initialization.")
        self._reader = reader
        self._gliner = gliner_model

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Converts raw image bytes to RGB numpy array."""
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image bytes")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def detect_text_and_boxes(self, image_bytes: bytes) -> Tuple[Image.Image, List[Tuple[str, List[int]]]]:
        """Locates text in image and returns bounding boxes with their content."""
        rgb = self._decode_image(image_bytes)
        detections = self._reader.readtext(rgb)

        boxed_words = []
        for bbox, text, _ in detections:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x, y = int(min(x_coords)), int(min(y_coords))
            w, h = int(max(x_coords) - x), int(max(y_coords) - y)
            boxed_words.append((text, [x, y, w, h]))

        pil_img = Image.fromarray(rgb)
        return pil_img, boxed_words

    def classify_pii(self, boxed_words: List[Tuple[str, List[int]]]) -> List[List[int]]:
        """Returns bounding boxes of text identified as PII."""
        pii_boxes = []
        for word, box in boxed_words:
            ents = self._gliner.predict_entities(word, labels=["PERSON", "EMAIL", "PHONE", "ID"])
            if any(e["label"] in {"PERSON", "EMAIL", "PHONE", "ID"} for e in ents):
                pii_boxes.append(box)
        return pii_boxes

    def redact_image(self, image: Image.Image, boxes: List[List[int]]) -> Image.Image:
        """Blacks out specified regions in the image."""
        draw = ImageDraw.Draw(image)
        for x, y, w, h in boxes:
            draw.rectangle([x, y, x + w, y + h], fill="black")
        return image

    def process_redaction(self, image_bytes: bytes) -> dict:
        """Runs the complete redaction pipeline and returns redacted image with box coordinates."""
        img, boxed_words = self.detect_text_and_boxes(image_bytes)
        pii_boxes = self.classify_pii(boxed_words)
        redacted_img = self.redact_image(img.copy(), pii_boxes)
        
        buf = io.BytesIO()
        redacted_img.save(buf, format="PNG")
        
        return {
            "boxes": pii_boxes,
            "redacted_image": buf.getvalue()
        }