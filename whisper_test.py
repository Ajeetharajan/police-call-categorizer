# whisper_test.py
import whisper
import sys
import os

print("Attempting to import torch...")
try:
    import torch
    print(f"Torch imported successfully. CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"ERROR: Could not import torch: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Unexpected error importing torch: {e}", file=sys.stderr)
    sys.exit(1)

print("Attempting to load Whisper model 'tiny'...")
try:
    model = whisper.load_model("tiny")
    print("Whisper 'tiny' model loaded successfully!")
    # Optional: You can try a dummy transcription to fully test
    # from librosa import load
    # import numpy as np
    # # Create dummy audio data (1 second of silence)
    # dummy_audio = np.zeros(16000).astype(np.float32)
    # print("Attempting dummy transcription...")
    # result = model.transcribe(dummy_audio)
    # print(f"Dummy transcription result: {result['text']}")
    print("Test completed successfully.")
except Exception as e:
    print(f"ERROR: Failed to load Whisper model: {e}", file=sys.stderr)
    # This will print the full traceback of the error to the console
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print("Script finished.")
