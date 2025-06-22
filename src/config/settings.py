import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Audio settings
SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a']
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Model settings
WHISPER_MODEL = "base"
TRANSLATION_MODEL = "googletrans"
