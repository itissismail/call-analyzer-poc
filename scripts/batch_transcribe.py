import os
import json
import whisperx
from whisperx.diarize import DiarizationPipeline

def process_audio(audio_path, output_path):
    """
    Transcribe, align, diarize, and save output as JSON.
    """
    print(f"\nProcessing: {audio_path}")

    # 1. Load model (CPU; use MPS when WhisperX supports it fully)
    model = whisperx.load_model("medium", device="cpu", compute_type="float32")

    # 2. Transcribe (skip language detection by specifying English)
    result = model.transcribe(audio_path, language="en")
    print("Transcription completed.")

    # 3. Align
    align_model, metadata = whisperx.load_align_model(language_code="en", device="cpu")
    result_aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device="cpu")
    print("Alignment completed.")

    # 4. Diarization
    diarize_model = DiarizationPipeline(use_auth_token=None, device="cpu")
    diarize_segments = diarize_model(audio_path)
    print("Diarization completed.")

    # 5. Assign speakers
    result_with_speakers = whisperx.assign_speakers(diarize_segments, result_aligned["segments"])

    # 6. Save to JSON
    with open(output_path, "w") as f:
        json.dump(result_with_speakers, f, indent=2)

    print(f"Output saved to {output_path}")


def batch_process(input_folder, output_folder):
    """
    Process all .wav/.mp3/.m4a files in input_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    audio_exts = (".wav", ".mp3", ".m4a", ".flac")

    for fname in os.listdir(input_folder):
        if fname.lower().endswith(audio_exts):
            audio_path = os.path.join(input_folder, fname)
            json_path = os.path.join(output_folder, os.path.splitext(fname)[0] + ".json")
            process_audio(audio_path, json_path)


if __name__ == "__main__":
    input_folder = "audio"   # replace with your folder containing audio
    output_folder = "results"

    batch_process(input_folder, output_folder)
