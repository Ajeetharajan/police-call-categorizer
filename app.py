import streamlit as st
import tempfile
import os
from pathlib import Path
import json
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(Path(__file__).parent / 'logs' / 'app.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Import your custom modules
from src.utils.audio_processor import AudioProcessor
from src.utils.transcriber import AudioTranscriber
from models.crime_analyzer import CrimeAnalyzer

# --- Model Initialization (Cached for performance) ---
@st.cache_resource # This tells Streamlit to load these heavy models only once
def load_models():
    """
    Loads and caches the heavy models and processors required by the app.
    This function will run only once across all user sessions.
    """
    logger.info("Loading models and processors...")
    try:
        # Initialize your custom classes (THESE ARE NOW UNCOMMENTED AND ACTIVE)
        audio_processor_instance = AudioProcessor()
        # Using 'tiny' model size for Whisper for better compatibility on free cloud tiers
        transcriber_instance = AudioTranscriber(model_size='tiny')
        analyzer_instance = CrimeAnalyzer()

        logger.info("Models and processors loaded successfully.")
        return {
            'audio_processor': audio_processor_instance,
            'transcriber': transcriber_instance,
            'analyzer': analyzer_instance
        }
    except Exception as e:
        logger.exception("Failed to load models!")
        st.error(f"Application failed to initialize: {e}. Please contact support or check logs for details.")
        st.stop() # Stop the app if critical models can't be loaded

# --- Main Streamlit Application Logic ---
def main():
    st.title("🚔 Police Call Analytics - Crime Insight Extractor")
    st.markdown("Upload police call audio files for automated transcription and crime analysis.")
    
    # Load models and components (this calls the cached function above)
    models = load_models()
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'm4a'], # These formats are specified in your config.py
        help="Supported formats: WAV, MP3, M4A (Max file size 100MB)"
    )
    
    if uploaded_file is not None:
        logger.info(f"Uploaded file: {uploaded_file.name}, Size: {uploaded_file.size} bytes")

        # Create a temporary file to save the uploaded audio on the server
        suffix = Path(uploaded_file.name).suffix
        if suffix.lower() == '.m4a':
            suffix = '.mp3' # pydub often treats m4a as mp3 for conversion

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Display an audio player for the uploaded file
            st.audio(temp_path, format=f'audio/{suffix.lstrip(".")}'.replace('mp3', 'mpeg'))
            
            # --- Audio Processing, Transcription, and Translation ---
            with st.spinner("Processing audio file, transcribing, and translating... This may take a moment."):
                col1, col2 = st.columns(2) # Create two columns for layout
                
                with col1:
                    st.subheader("📄 Transcription")
                    
                    # Call the main transcribe method which handles full pipeline
                    full_processing_result = models['transcriber'].transcribe(temp_path)
                    
                    original_text = full_processing_result['transcription']['text']
                    detected_language = full_processing_result['transcription']['language']
                    
                    st.write(f"**Detected Language:** `{detected_language.upper()}`")
                    st.text_area("Original Transcription:", original_text, height=150, help="The raw text transcribed from the audio.")
                    
                    # Determine the text to be used for analysis: translated English or original English
                    analysis_text = original_text 
                    translation_data = full_processing_result.get('translation')

                    if translation_data and translation_data['translated_text'] and translation_data['translated_text'] != original_text:
                        st.text_area("English Translation:", translation_data['translated_text'], height=150, help="Translation to English if the original language was different.")
                        analysis_text = translation_data['translated_text']
                        logger.info(f"Used translated text for analysis. Original language: {detected_language.upper()}")
                    elif detected_language.lower() != 'en':
                        st.info(f"Audio detected as '{detected_language.upper()}' but translation was not explicitly performed or resulted in the same text as original. Analysis will proceed on original text.")
                        logger.warning(f"Audio was {detected_language.upper()} but translation was not distinct. Analyzing original text.")
                    else:
                        st.info("Audio is already in English. No translation needed.")
                        logger.info("Used original English text for analysis.")

                with col2:
                    st.subheader("🔍 Crime Analysis")
                    
                    # Analyze the text (always in English due to the above logic)
                    analysis_result = models['analyzer'].analyze(analysis_text)
                    
                    # Display key metrics using Streamlit's st.metric
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Crime Category", analysis_result['complaint_category'].replace('_', ' ').title(), help="The primary category of the crime detected.")
                    
                    with metrics_col2:
                        display_confidence = f"{analysis_result['confidence_score']:.2f}" if analysis_result['confidence_score'] is not None else "N/A"
                        st.metric("Category Confidence", display_confidence, help="Confidence score of the detected crime category (0-1).")
                    
                    with metrics_col3:
                        st.metric("Urgency Level", analysis_result['urgency_level'], help="Estimated urgency of the complaint (LOW, MEDIUM, HIGH).")
                
                # --- Detailed Results Section (Tabs) ---
                st.subheader("📊 Detailed Analysis")
                
                tab1, tab2, tab3, tab4 = st.tabs(["Entities", "Contact Info", "Summary", "Raw JSON Output"])
                
                with tab1:
                    entities = analysis_result['extracted_entities']
                    if any(entities.values()): # Check if any entity list is non-empty
                        st.markdown("Identified key entities within the call text:")
                        for entity_type, values in entities.items():
                            if values:
                                st.write(f"**{entity_type.replace('_', ' ').title()}:** {', '.join(set(values))}") # Using set to avoid duplicates
                    else:
                        st.info("No significant entities (persons, locations, organizations, dates, money) detected.")
                
                with tab2:
                    contact = analysis_result['contact_information']
                    if contact['phone_numbers'] or contact['addresses']:
                        st.markdown("Extracted contact details:")
                        if contact['phone_numbers']:
                            st.write(f"**Phone Numbers:** {', '.join(set(contact['phone_numbers']))}")
                        if contact['addresses']:
                            st.write(f"**Addresses:** {', '.join(set(contact['addresses']))}")
                    else:
                        st.info("No phone numbers or addresses detected.")
                
                with tab3:
                    st.write("**Analysis Summary:**")
                    # Ensure full_processing_result contains 'processing_info' and 'audio_info'
                    summary_data = {
                        'Metric': [
                            'Word Count (Analysis Text)', 
                            'Character Count (Analysis Text)', 
                            'Detected Language (Original Audio)', 
                            'Audio Duration', 
                            'Audio Sample Rate',
                            'Audio File Size (MB)'
                        ],
                        'Value': [
                            analysis_result.get('word_count', 'N/A'),
                            analysis_result.get('text_length', 'N/A'),
                            detected_language.upper(),
                            f"{full_processing_result['audio_info'].get('duration', 0):.2f}s",
                            f"{full_processing_result['audio_info'].get('sample_rate', 'N/A')} Hz",
                            f"{full_processing_result['audio_info'].get('file_size_mb', 0):.2f}"
                        ]
                    }
                    st.table(pd.DataFrame(summary_data))
                
                with tab4:
                    st.json(analysis_result) # Display the full JSON result of crime analysis
                    st.json(full_processing_result) # Display the full JSON result of transcription/translation
                
                # --- Download Results ---
                st.subheader("💾 Export Results")
                
                # Prepare combined data for download
                export_data = {
                    'file_name': uploaded_file.name,
                    'original_audio_info': full_processing_result.get('audio_info'),
                    'transcription_details': full_processing_result.get('transcription'),
                    'translation_details': full_processing_result.get('translation'),
                    'crime_analysis_results': analysis_result
                }
                
                st.download_button(
                    label="Download All Analysis Data (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"police_call_analysis_{Path(uploaded_file.name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        except Exception as e:
            logger.exception("An error occurred during file processing:")
            st.error(f"An error occurred during processing: {str(e)}. Please try a different file or contact support.")
        
        finally:
            # Ensure the temporary file is removed
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")

# --- Entry Point for Streamlit App ---
if __name__ == "__main__":
    main()
