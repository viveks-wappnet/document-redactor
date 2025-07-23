import os
import pytesseract
from huggingface_hub import login
from gliner import GLiNER
from dotenv import load_dotenv

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class ModelManager:
    """Singleton manager for ML models ensuring one-time initialization of GLiNER."""
    _instance = None
    
    def __init__(self):
        if ModelManager._instance is not None:
            raise RuntimeError("Use ModelManager.get_instance() to get the single instance of this class.")
        self.gliner = None
        
    @classmethod
    def get_instance(cls):
        """Returns the singleton instance of ModelManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize(self):
        """Loads GLiNER model if not already initialized."""
        if self.gliner is None:
            print("Initializing GLiNER...")
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
            self.gliner = GLiNER.from_pretrained(os.getenv("GLINER_MODEL_NAME"), load_onnx_model=True, load_tokenizer=True, onnx_model_file="onnx\model.onnx")
            print("GLiNER initialized.")

def get_model_manager():
    """
    Global accessor for the ModelManager instance.
    """
    return ModelManager.get_instance()
