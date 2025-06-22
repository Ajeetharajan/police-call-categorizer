# src/utils/transcriber.py

import logging
from typing import Dict, Tuple
# Import the actual TranscriptionManager from the models directory
from src.models.transcription import TranscriptionManager

# For direct translation if app.py calls translate_to_english separately
from googletrans import Translator
from langdetect import detect

logger = logging.getLogger(__name__)

class AudioTranscriber:
    """
    A wrapper class to bridge app.py's expectation with the TranscriptionManager.
    This class will be imported by app.py.
    """
    def __init__(self,model_size: str='tiny'):
        # Initialize an instance of TranscriptionManager
        self.manager = TranscriptionManager(model_size=model_size)
        logger.info("AudioTranscriber (wrapper for TranscriptionManager) initialized.")

    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribes the given audio file using the TranscriptionManager's pipeline,
        which includes preprocessing, transcription, and conditional translation.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            Dict: Comprehensive results including 'transcription' and 'translation' data.
        """
        # Call the transcribe_file method of the TranscriptionManager
        # This method handles the full pipeline including conditional translation
        return self.manager.transcribe_file(audio_path, translate_to_english=True)

    def translate_to_english(self, text: str) -> Tuple[str, str]:
        """
        Translates text to English directly using googletrans.
        This method is provided for direct calls from app.py, if any.
        Ideally, the main 'transcribe' method should handle the full flow.
        
        Args:
            text (str): The text string to translate.
            
        Returns:
            Tuple[str, str]: A tuple containing the translated text and the detected source language.
                             Returns original text and 'en' if already English or translation fails.
        """
        try:
            if not text.strip(): # Handle empty string input
                return "", "en"

            # Detect language of the input text
            try:
                source_language = detect(text)
            except:
                source_language = "unknown" # Fallback if detection fails

            if source_language == 'en':
                logger.info("Text is already in English, no translation needed in translate_to_english method.")
                return text, 'en'
            
            logger.info(f"Directly translating from {source_language} to English using googletrans.")
            translator = Translator()
            translated_obj = translator.translate(text, dest='en')
            
            return translated_obj.text, translated_obj.src
        except Exception as e:
            logger.error(f"Error in direct translation within AudioTranscriber: {e}")
            return text, "en" # Fallback to original text if an error occurs
