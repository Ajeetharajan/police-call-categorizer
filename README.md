---
title: Police Call Analytics
emoji: 🚔
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.34.0 # You can adjust this to your Streamlit version if you know it, otherwise this is a safe default.
app_file: app.py # This tells Hugging Face to run your app.py
python_version: 3.10 # Make sure this matches what you picked when creating the space
---

# Police Call Analytics - Crime Insight Extractor

This Space hosts a Streamlit application for transcribing police call audio and performing crime analysis.

**Features:**
- Audio transcription using OpenAI Whisper.
- Language detection and English translation.
- Extraction of crime categories, urgency levels, and key entities.
- Summary and raw JSON output of analysis.

**To use:**
1. Upload an audio file (WAV, MP3, M4A).
2. View the transcription, translation, and crime analysis results. 
