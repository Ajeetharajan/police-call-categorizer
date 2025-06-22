"""
Audio processing utilities for police call analytics
"""
import os
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pathlib import Path
import logging
from typing import Tuple, Optional
from src.config.config import AUDIO_CONFIG

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handle audio file processing and preparation for transcription"""
    
    def __init__(self):
        self.sample_rate = AUDIO_CONFIG['sample_rate']
        self.supported_formats = AUDIO_CONFIG['supported_formats']
        self.max_file_size_mb = AUDIO_CONFIG['max_file_size_mb']
        self.chunk_duration = AUDIO_CONFIG['chunk_duration']
    
    def validate_audio_file(self, file_path: str) -> bool:
        """
        Validate audio file format and size
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            # Check file extension
            if path.suffix.lower() not in self.supported_formats:
                logger.error(f"Unsupported format: {path.suffix}")
                return False
            
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                logger.error(f"File too large: {file_size_mb:.2f}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio file: {e}")
            return False
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to standard format
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        try:
            if not self.validate_audio_file(file_path):
                raise ValueError("Invalid audio file")
            
            # Load audio with librosa
            audio_data, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=True
            )
            
            logger.info(f"Loaded audio: {file_path}, Duration: {len(audio_data)/sr:.2f}s")
            
            return audio_data, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio data for better transcription
        
        Args:
            audio_data (np.ndarray): Raw audio data
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Preprocessed audio data
        """
        try:
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Reduce noise using spectral gating
            audio_data = self._reduce_noise(audio_data, sr)
            
            # Trim silence
            audio_data, _ = librosa.effects.trim(
                audio_data,
                top_db=20,
                frame_length=2048,
                hop_length=512
            )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return audio_data  # Return original if preprocessing fails
    
    def _reduce_noise(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Simple noise reduction using spectral subtraction
        
        Args:
            audio_data (np.ndarray): Audio data
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Noise-reduced audio
        """
        try:
            # Compute STFT
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first 0.5 seconds
            noise_frame_count = int(0.5 * sr / 512)
            noise_spectrum = np.mean(magnitude[:, :noise_frame_count], axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01   # Spectral floor
            
            clean_magnitude = magnitude - alpha * noise_spectrum
            clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
            
            # Reconstruct audio
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft)
            
            return clean_audio
            
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            return audio_data
    
    def split_long_audio(self, audio_data: np.ndarray, sr: int) -> list:
        """
        Split long audio into chunks for processing
        
        Args:
            audio_data (np.ndarray): Audio data
            sr (int): Sample rate
            
        Returns:
            list: List of audio chunks
        """
        try:
            duration = len(audio_data) / sr
            
            if duration <= self.chunk_duration:
                return [audio_data]
            
            chunk_samples = self.chunk_duration * sr
            chunks = []
            
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i:i + chunk_samples]
                if len(chunk) > sr:  # Only include chunks longer than 1 second
                    chunks.append(chunk)
            
            logger.info(f"Split audio into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting audio: {e}")
            return [audio_data]
    
    def save_processed_audio(self, audio_data: np.ndarray, sr: int, output_path: str) -> bool:
        """
        Save processed audio to file
        
        Args:
            audio_data (np.ndarray): Audio data to save
            sr (int): Sample rate
            output_path (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            sf.write(output_path, audio_data, sr)
            logger.info(f"Saved processed audio to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False
    
    def get_audio_info(self, file_path: str) -> dict:
        """
        Get audio file information
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            dict: Audio file information
        """
        try:
            audio_data, sr = self.load_audio(file_path)
            
            info = {
                'duration': len(audio_data) / sr,
                'sample_rate': sr,
                'channels': 1,  # We convert to mono
                'file_size_mb': Path(file_path).stat().st_size / (1024 * 1024),
                'format': Path(file_path).suffix.lower(),
                'samples': len(audio_data)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {}

def convert_to_wav(input_file: str, output_file: str) -> bool:
    """
    Convert audio file to WAV format using pydub
    
    Args:
        input_file (str): Input audio file path
        output_file (str): Output WAV file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        logger.info(f"Converted {input_file} to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return False
