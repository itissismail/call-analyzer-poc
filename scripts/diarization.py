import whisperx
from whisperx.diarize import DiarizationPipeline

audio_file = "./audio/sample.mp3"

# Transcribe
model = whisperx.load_model("medium", device="cpu", compute_type="float32")
result = model.transcribe(audio_file, language="en")
print("Transcription completed.")
# Align
align_model, metadata = whisperx.load_align_model(language_code="en", device="cpu")
result_aligned = whisperx.align(result["segments"], align_model, metadata, audio_file, device="cpu")
print("\n Aligh completed.")
# Diarization
diarize_model = DiarizationPipeline(use_auth_token=None, device="cpu")
diarize_segments = diarize_model(audio_file)
print("\n diarize_model completed.")
# Assign speakers
result_with_speakers = whisperx.assign_speakers(diarize_segments, result_aligned["segments"])

for seg in result_with_speakers:
    print(f"{seg['speaker']} [{seg['start']:.2f}-{seg['end']:.2f}]: {seg['text']}")
