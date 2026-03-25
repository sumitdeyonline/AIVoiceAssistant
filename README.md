# Streaming Voice AI Assistant 🎙️

An ultra-low latency, real-time conversational AI voice assistant. This project implements a fully streaming pipeline optimized for fast Time-to-First-Audio (TTFA) and features a beautiful **Streamlit** web interface for real-time visualization of transcripts and latency milestones.

## Features
- **Voice Activity Detection (VAD)**: Uses `webrtcvad` to detect end-of-speech automatically.
- **Streaming Pipeline**: Asynchronous orchestration (`stt.py` -> `llm.py` -> `tts.py`) that pipelines sentence chunks directly into audio synthesis before the LLM even finishes generating.
- **Latency Tracking**: Granular tracking of STT Time, LLM Time-to-First-Token (TTFT), and TTS Time-to-First-Byte (TFB).
- **Graceful Degradation**: Built-in timeout handling. If cloud APIs (Groq/OpenAI) experience network spikes or if keys are missing, the system instantly degrades securely to local mocks and native macOS local text-to-speech (`say`), preventing the pipeline from hanging.
- **Streamlit GUI**: A threaded frontend that natively renders the conversation and latency budget without blocking the core audio I/O loop.

## Prerequisites
- macOS (for native `say` local fallback)
- Python 3.10+
- `uv` package manager (recommended) or `pip`

You will also need to install `portaudio` for microphone access:
```bash
brew install portaudio
```

## Setup & Installation

1. **Install dependencies**:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
*(Note: If you encounter `pkg_resources` issues with the older `webrtcvad` module, our `src/audio_io.py` script automatically injects a runtime mock to patch it seamlessly without dependency downgrades!)*

2. **Environment Variables**:
Create a `.env` file in the root directory and add your API keys. If you don't add these, the assistant will gracefully fall back to deterministic text mocks and local system TTS.
```env
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Usage

### Run the Streamlit Web App (Recommended)
Launch the unified dashboard:
```bash
uv run streamlit run src/app.py
```
> Click **"🟢 Start Listening"** to begin the background voice loop. You can stop the conversation stream flexibly using the UI buttons.

### Run via Command Line
If you prefer a headless interface, run the raw script:
```bash
uv run src/pipeline.py
```
Press `Ctrl+C` to cleanly shut down the asynchronous event loop.

## Architecture Highlights
- **`src/audio_io.py`**: Handles infinite loops of PyAudio recording, leveraging `webrtcvad` to identify speech segments securely.
- **`src/pipeline.py`**: The `VoiceAssistant` orchestrator. Wraps API operations in timeout limits and pushes granular execution events onto a thread-safe UI Queue.
- **`src/app.py`**: The Streamlit frontend. It executes the VoiceAssistant inside a `threading.Thread` daemon and drains the UI queue iteratively to update the chat interface.
