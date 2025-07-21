import os
import easyocr
from huggingface_hub import login
from gliner import GLiNER
from dotenv import load_dotenv
import warnings

# Suppress specific warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, message='.*pin_memory.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*sentencepiece tokenizer.*')
warnings.filterwarnings('ignore', message='Asking to truncate.*')

class ModelManager:
    """Singleton manager for ML models ensuring one-time initialization of EasyOCR and GLiNER."""
    _instance = None
    
    def __init__(self):
        if ModelManager._instance is not None:
            raise RuntimeError("Use ModelManager.get_instance() to get the single instance of this class.")
        self.easyocr_reader = None
        self.gliner = None
        
    @classmethod
    def get_instance(cls):
        """Returns the singleton instance of ModelManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize(self):
        """Loads EasyOCR and GLiNER models if not already initialized."""
        if self.easyocr_reader is None:
            print("Initializing EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("EasyOCR initialized.")
            
        if self.gliner is None:
            print("Initializing GLiNER...")
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
            self.gliner = GLiNER.from_pretrained(os.getenv("GLINER_MODEL_NAME"))
            print("GLiNER initialized.")

def get_model_manager():
    """
    Global accessor for the ModelManager instance.
    """
    return ModelManager.get_instance()
