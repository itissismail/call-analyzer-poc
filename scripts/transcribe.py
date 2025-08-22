import whisperx
import os
import sys
import torch
from whisperx.diarize import DiarizationPipeline

def transcribe(audio_path, output_path):
    print(f"Loading model and transcribing: {audio_path}")

    # Load ASR model (CPU on Mac; float32 since float16 is limited)
    model = whisperx.load_model("medium", device="cpu", compute_type="float32")

    # Step 1: Transcribe with known language
    result = model.transcribe(audio_path, language="en")   # <--- specify language
    print("Transcription completed.")

    # Step 2: Align
    model_a, metadata = whisperx.load_align_model(language_code="en", device="cpu")  # <--- specify language
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device="cpu")

    # Step 3: Diarization
    #diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device="cpu")
    #diarize_segments = diarize_model(audio_path)

    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=None,  # or your Hugging Face token if needed
        device="cpu"          # or "cuda" if using NVIDIA GPU
    )
    diarize_segments = diarize_model(audio_path)

    # Step 4: Assign speaker labels
    result_with_speakers = whisperx.assign_speakers(diarize_segments, result_aligned["segments"])

    print("Speaker diarization completed.")
    for seg in result_with_speakers:
        print(f"Speaker {seg['speaker']} [{seg['start']:.2f}-{seg['end']:.2f}]: {seg['text']}")


    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        import json
        #json.dump(result_with_speakers, f, indent=2, ensure_ascii=False)
        json.dump(result_aligned, f, indent=2, ensure_ascii=False)

    print(f"Saved transcription to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/transcribe.py <input_audio.wav> <output.json>")
        sys.exit(1)
    audio_file = sys.argv[1]
    output_file = sys.argv[2]
    transcribe(audio_file, output_file)

