"""
Transcription models for converting audio to text
"""
# ... existing imports ...
from googletrans import Translator # ADD THIS LINE
from langdetect import detect # ADD THIS LINE (for initial language check)
import whisper
import logging
from typing import Optional, Dict, List
import torch
from pathlib import Path
import numpy as np
from src.config.config import MODEL_CONFIG
from src.utils.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """OpenAI Whisper-based transcription"""
    
    def __init__(self, model_size: str = None):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size (str): Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size or MODEL_CONFIG['whisper_model']
        self.model = None
        self.audio_processor = AudioProcessor()
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, task: str = "transcribe") -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path (str): Path to audio file
            task (str): "transcribe" or "translate" (to English)
            
        Returns:
            Dict: Transcription results with text, language, and segments
        """
        try:
            if not self.model:
                raise ValueError("Model not loaded")
            
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Load and preprocess audio
            audio_data, sr = self.audio_processor.load_audio(audio_path)
            audio_data = self.audio_processor.preprocess_audio(audio_data, sr)
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_data,
                task=task,
                language=None,  # Auto-detect
                fp16=torch.cuda.is_available(),
                verbose=False
            )
            
            # Process results
            transcription_result = {
                'text': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'segments': self._process_segments(result.get('segments', [])),
                'confidence': self._calculate_confidence(result.get('segments', [])),
                'duration': len(audio_data) / sr
            }
            
            logger.info(f"Transcription completed. Language: {transcription_result['language']}")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise
    
    def transcribe_chunks(self, audio_chunks: List[np.ndarray], sr: int) -> Dict:
        """
        Transcribe multiple audio chunks
        
        Args:
            audio_chunks (List[np.ndarray]): List of audio chunks
            sr (int): Sample rate
            
        Returns:
            Dict: Combined transcription results
        """
        try:
            all_text = []
            all_segments = []
            languages = []
            confidences = []
            total_duration = 0
            
            for i, chunk in enumerate(audio_chunks):
                logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}")
                
                # Transcribe chunk
                result = self.model.transcribe(
                    chunk,
                    task="transcribe",
                    language=None,
                    fp16=torch.cuda.is_available(),
                    verbose=False
                )
                
                # Collect results
                all_text.append(result['text'].strip())
                languages.append(result.get('language', 'unknown'))
                
                # Process segments with time offset
                chunk_duration = len(chunk) / sr
                segments = self._process_segments(
                    result.get('segments', []),
                    time_offset=total_duration
                )
                all_segments.extend(segments)
                
                confidences.append(self._calculate_confidence(result.get('segments', [])))
                total_duration += chunk_duration
            
            # Combine results
            combined_result = {
                'text': ' '.join(all_text),
                'language': max(set(languages), key=languages.count),  # Most common language
                'segments': all_segments,
                'confidence': np.mean(confidences) if confidences else 0.0,
                'duration': total_duration,
                'chunks_processed': len(audio_chunks)
            }
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error transcribing chunks: {e}")
            raise
    
    def _process_segments(self, segments: List[Dict], time_offset: float = 0) -> List[Dict]:
        """
        Process transcription segments
        
        Args:
            segments (List[Dict]): Raw segments from Whisper
            time_offset (float): Time offset for chunk processing
            
        Returns:
            List[Dict]: Processed segments
        """
        processed_segments = []
        
        for segment in segments:
            processed_segment = {
                'id': segment.get('id', 0),
                'start': segment.get('start', 0) + time_offset,
                'end': segment.get('end', 0) + time_offset,
                'text': segment.get('text', '').strip(),
                'confidence': self._get_segment_confidence(segment),
                'no_speech_prob': segment.get('no_speech_prob', 0.0)
            }
            processed_segments.append(processed_segment)
        
        return processed_segments
    
    def _get_segment_confidence(self, segment: Dict) -> float:
        """
        Calculate confidence score for a segment
        
        Args:
            segment (Dict): Segment data
            
        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Use average log probability if available
            if 'avg_logprob' in segment:
                # Convert log probability to confidence (approximate)
                return max(0.0, min(1.0, np.exp(segment['avg_logprob'])))
            
            # Use no_speech_prob as fallback
            if 'no_speech_prob' in segment:
                return 1.0 - segment['no_speech_prob']
            
            return 0.5  # Default confidence
            
        except Exception:
            return 0.5
    
    def _calculate_confidence(self, segments: List[Dict]) -> float:
        """
        Calculate overall confidence score
        
        Args:
            segments (List[Dict]): List of segments
            
        Returns:
            float: Overall confidence score
        """
        if not segments:
            return 0.0
        
        try:
            confidences = []
            for segment in segments:
                confidence = self._get_segment_confidence(segment)
                # Weight by segment length
                duration = segment.get('end', 0) - segment.get('start', 0)
                confidences.extend([confidence] * max(1, int(duration)))
            
            return np.mean(confidences) if confidences else 0.0
            
        except Exception:
            return 0.5

class TranscriptionManager:
    """Manage different transcription services"""
    
    def __init__(self,model_size:str=None):
        self.whisper_transcriber = WhisperTranscriber(model_size=model_size)
        self.audio_processor = AudioProcessor()
    
    def transcribe_file(self, audio_path: str, translate_to_english: bool = True) -> Dict:
        """
        Transcribe audio file with automatic language detection and translation
        
        Args:
            audio_path (str): Path to audio file
            translate_to_english (bool): Whether to translate to English
            
        Returns:
            Dict: Complete transcription and translation results
        """
        try:
            logger.info(f"Starting transcription for: {audio_path}")
            
            # Get audio info
            audio_info = self.audio_processor.get_audio_info(audio_path)
            
            # Load and split audio if needed
            audio_data, sr = self.audio_processor.load_audio(audio_path)
            audio_data = self.audio_processor.preprocess_audio(audio_data, sr)
            
            # Split into chunks if audio is long
            chunks = self.audio_processor.split_long_audio(audio_data, sr)
            
            # Transcribe
            if len(chunks) == 1:
                # Single chunk transcription
                temp_path = Path(audio_path).parent / "temp_processed.wav"
                self.audio_processor.save_processed_audio(audio_data, sr, str(temp_path))
                
                transcription_result = self.whisper_transcriber.transcribe_audio(
                    str(temp_path),
                    task="transcribe"
                )
                
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()
                    
            else:
                # Multi-chunk transcription
                transcription_result = self.whisper_transcriber.transcribe_chunks(chunks, sr)
            
            # Translate if needed and not already in English
            translation_result = None
            if translate_to_english and transcription_result['language'] != 'en':
                translation_result = self._translate_to_english(transcription_result)
            
            # Combine results
            final_result = {
                'audio_info': audio_info,
                'transcription': transcription_result,
                'translation': translation_result,
                'processing_info': {
                    'chunks_processed': len(chunks),
                    'model_used': 'whisper',
                    'preprocessing_applied': True
                }
            }
            
            logger.info("Transcription completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in transcription pipeline: {e}")
            raise
    
    # src/models/transcription.py (inside TranscriptionManager class)

    def _translate_to_english(self, transcription_result: Dict) -> Dict:
        """
        Translate transcription to English using Google Translate.
        
        Args:
            transcription_result (Dict): Original transcription results dictionary
                                        (must contain 'text' and 'language' keys).
            
        Returns:
            Dict: Translation results.
        """
        try:
            original_text = transcription_result['text']
            source_language = transcription_result.get('language', 'auto') # Get detected language from Whisper

            if not original_text.strip(): # Handle empty text input
                logger.info("Empty text provided for translation.")
                return {
                    'translated_text': "",
                    'source_language': source_language,
                    'target_language': 'en',
                    'confidence': transcription_result.get('confidence', 1.0),
                    'method': 'n/a (empty text)'
                }

            if source_language == 'en' or source_language.lower() == 'english': # If already English, no need to translate
                logger.info("Content is already in English, no translation needed by Google Translate.")
                return {
                    'translated_text': original_text,
                    'source_language': 'en',
                    'target_language': 'en',
                    'confidence': transcription_result.get('confidence', 1.0),
                    'method': 'n/a (already English)'
                }

            logger.info(f"Translating from {source_language} to English using Google Translate.")
            translator = Translator()
            
            # Perform the translation
            translated_obj = translator.translate(original_text, dest='en')

            translated_text = translated_obj.text
            # Use detected source language from googletrans if it was auto-detected by googletrans
            # otherwise stick with Whisper's detection
            detected_src_lang_by_gt = translated_obj.src if translated_obj.src and translated_obj.src != 'auto' else source_language

            translation_result = {
                'translated_text': translated_text,
                'source_language': detected_src_lang_by_gt,
                'target_language': 'en',
                'confidence': transcription_result.get('confidence', 0.0), # Re-use Whisper's confidence or default
                'method': 'google_translate'
            }
            
            logger.info("Translation completed successfully with Google Translate.")
            return translation_result

        except Exception as e:
            logger.error(f"Error during translation in TranscriptionManager: {e}")
            return {
                'translated_text': original_text, # Fallback to original text on error
                'source_language': transcription_result.get('language', 'unknown'),
                'target_language': 'en',
                'confidence': 0.0, # Indicate low confidence due to error
                'method': 'error_fallback'
            }
