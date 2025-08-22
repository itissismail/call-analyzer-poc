# 📞 Call Analyzer POC – Transcription & Diarization

A lightweight proof-of-concept (POC) project to transcribe call center audio recordings and identify different speakers using open-source AI tools. This is the first step toward evaluating call quality against KPIs.

## 📁 Project Structure
#### Call-Analyzer-poc
- audio/ # Input audio files (.wav/.mp3)
- models/ # (Optional) downloaded model files
- results/ # JSON transcription output
- scripts/ # Python scripts for processing
- .venv/ # Python virtual environment (local)
- requirements.txt
- README.md


## ✅ Features

- 🎙️ Transcription using [WhisperX](https://github.com/m-bain/whisperx)
- 🧠 Speaker diarization (who spoke when)
- 💬 Output structured JSON with timestamps and speaker tags
- 📦 Ready for KPI analysis and n8n integration

## 🧰 Prerequisites

- macOS with Python **3.10+**
- [Homebrew](https://brew.sh) installed
- Git and Terminal

## 🚀 Installation Steps

### 1. Clone the repository

```
cd ~/Projects/Test
git clone <your-repo-url> call-analyzer-poc
cd call-analyzer-poc
```

Or manually create the folder:

```
mkdir -p ~/Projects/Test/call-analyzer-poc/{audio,scripts,results,models}
cd ~/Projects/Test/call-analyzer-poc
```

### Set up Python virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
### Install dependencies
```

```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
### Install FFmpeg (for audio processing)

WhisperX integrates diarization via pyannote.audio.
pip install pyannote.audio (Already covered in requirements.txt)
```
```
brew install ffmpeg
```

### 🎤 How to Use
####  1. Add your audio file
Place your .wav or .mp3 file inside the audio/ folder. Example:

bash

audio/sample.wav
####  2. Run transcription + diarization

```
python scripts/transcribe.py audio/sample.wav results/sample.json
or 
python scripts/transcribe.py audio/sample.mp3 results/sample.json
```
The output results/sample.json will look like:

```
[
  {
    "start": 0.5,
    "end": 2.1,
    "text": "Hello, is this Mr. John?",
    "speaker": "SPEAKER_00"
  },
  ...
]
```

###  🧪 Test Example
Download a sample audio file from this GitHub repo or use any .wav file with 1–2 minutes of spoken conversation.

🧹 .gitignore Suggestions
markdown
Copy
Edit
.venv/
__pycache__/
*.pyc
results/*.json
audio/*.wav

###  🛠️ Next Steps
Implement rule-based KPI analysis (script adherence, handoff, etc.)

Integrate with n8n to automate audio upload and processing

Generate summary reports from transcription

Explore using local LLMs (optional)

###  📄 License
This POC is for demo and evaluation purposes only. Open-source tools used here are governed by their respective licenses.