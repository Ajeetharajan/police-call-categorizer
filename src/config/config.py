"""
Configuration settings for Police Call Analytics System
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_AUDIO_DIR = DATA_DIR / "sample_audio"
OUTPUT_DIR = DATA_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, SAMPLE_AUDIO_DIR, OUTPUT_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Audio processing settings
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'supported_formats': ['.wav', '.mp3', '.m4a', '.flac'],
    'max_file_size_mb': 100,
    'chunk_duration': 30,  # seconds
}

# Model settings
MODEL_CONFIG = {
    'whisper_model': 'tiny',  # Options: tiny, base, small, medium, large
    'translation_service': 'google',  # Options: google, whisper
    'classification_model': 'distilbert-base-uncased',
    'embedding_model': 'all-MiniLM-L6-v2',
}

# Crime categories and keywords
CRIME_CATEGORIES = {
    'Robbery': [
        'rob', 'robbery', 'robbed', 'steal', 'stolen', 'theft', 'burglar', 
        'break in', 'breaking and entering', 'loot', 'heist', 'mugging'
    ],
    'Assault': [
        'assault', 'attack', 'attacked', 'hit', 'punch', 'kick', 'beat', 
        'violence', 'fight', 'physical', 'battery', 'abuse', 'hurt'
    ],
    'Cybercrime': [
        'hack', 'hacker', 'cyber', 'online', 'internet', 'phishing', 
        'scam', 'fraud', 'identity theft', 'ransomware', 'malware', 'spam'
    ],
    'Domestic Violence': [
        'domestic', 'family', 'spouse', 'partner', 'husband', 'wife',
        'boyfriend', 'girlfriend', 'home violence', 'family dispute'
    ],
    'Drug Related': [
        'drug', 'drugs', 'narcotics', 'cocaine', 'heroin', 'marijuana',
        'dealing', 'possession', 'trafficking', 'overdose', 'substance'
    ],
    'Traffic Violation': [
        'traffic', 'accident', 'car crash', 'speeding', 'drunk driving',
        'hit and run', 'vehicle', 'driving', 'road rage', 'collision'
    ],
    'Fraud': [
        'fraud', 'fraudulent', 'scam', 'cheat', 'deception', 'forgery',
        'embezzlement', 'money laundering', 'counterfeit', 'fake'
    ],
    'Vandalism': [
        'vandalism', 'graffiti', 'damage', 'destroy', 'property damage',
        'broken window', 'spray paint', 'defacement'
    ],
    'Other': []
}

# Insights extraction patterns
INSIGHT_PATTERNS = {
    'location_keywords': [
        'at', 'near', 'street', 'avenue', 'road', 'building', 'store',
        'park', 'school', 'hospital', 'bank', 'mall', 'restaurant'
    ],
    'time_patterns': [
        r'\b\d{1,2}:\d{2}\b',  # Time format like 10:30
        r'\b\d{1,2}\s*(am|pm)\b',  # Time with am/pm
        r'\b(morning|afternoon|evening|night|midnight|noon)\b',
        r'\b(yesterday|today|tonight|last night)\b',
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
    ],
    'urgency_keywords': [
        'emergency', 'urgent', 'immediate', 'help', 'now', 'quickly',
        'serious', 'critical', 'dangerous', 'threat', 'weapon', 'gun', 'knife'
    ],
    'person_indicators': [
        'suspect', 'victim', 'witness', 'perpetrator', 'man', 'woman',
        'person', 'individual', 'male', 'female', 'teen', 'adult'
    ]
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'Police Call Analytics - Crime Insight Extractor',
    'page_icon': '🚔',
    'layout': 'wide',
    'sidebar_width': 300,
}

# API Keys (use environment variables)
API_KEYS = {
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'google_translate_key': os.getenv('GOOGLE_TRANSLATE_KEY'),
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': BASE_DIR / 'logs' / 'app.log'
}
