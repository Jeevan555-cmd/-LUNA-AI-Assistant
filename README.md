# Personal Voice Assistant

Press `1` to talk, press any other key to stop.

    mic -> .wav (local) -> Whisper -> LLM (Ollama) -> Piper -> .mp3 (local) -> played back

## Setup

1. Install system tools: `ffmpeg` (includes `ffplay`) and [Ollama](https://ollama.com).
2. `pip install -r requirements.txt`
3. Pull an LLM: `ollama pull llama3.2`
4. Download a Piper voice (both the `.onnx` and `.onnx.json` files) into `voices/`:
   `python -m piper.download_voices en_US-lessac-medium --data-dir voices`
   (or grab it from the Piper voices page on Hugging Face)
5. Run: `python main.py`

macOS: grant your terminal Accessibility + Microphone + Input Monitoring permission
(System Settings -> Privacy & Security), or the key listener won't work.

## Files

- `config.py`   settings (key, models, paths)
- `recorder.py` mic -> wav
- `stt.py`      Whisper
- `llm.py`      LLM call; loads and saves conversation via `memory.py`
- `memory.py`   SQLite conversation memory (`data/memory.db`)
- `tts.py`      Piper -> mp3 -> playback
- `main.py`     key listener + pipeline

## Memory

Every user message and reply is saved to `data/memory.db`. On startup the last
`MEMORY_MESSAGES` messages (default 20) are loaded, so it remembers the previous
message, even after a restart. Delete `data/memory.db` to wipe it.

Next step for long-term memory: store extracted facts and retrieve relevant ones in
`llm.py` `_build_messages()`.


## Local owner voice verification

Luna uses a local WeSpeaker ONNX speaker-embedding model through `sherpa-onnx`.
The first launch asks for microphone permission for the current session. If no
owner profile exists, Luna asks for explicit enrollment consent and captures
three separate natural speech samples.

Only the averaged speaker embedding is stored in `data/owner_voice.json`.
Temporary enrollment audio is not written to disk. After each wake word, Luna
asks for a fresh speech sample before starting the active conversation.

### Install the additional dependency

Activate the existing virtual environment, then run:

```powershell
python -m pip install sherpa-onnx
```

This adds the speaker-verification runtime without recreating the virtual
environment or reinstalling the rest of the project.

### Download the local speaker model

Run these commands from the `backend` directory:

```powershell
New-Item -ItemType Directory -Force .\data\models | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx" `
  -OutFile ".\data\models\wespeaker_en_voxceleb_resnet34.onnx"
```

The model runs locally. Keep `data/owner_voice.json` private because it
contains biometric-derived data. To remove enrollment and enroll again, stop
Luna and delete `data/owner_voice.json`.

### Threshold tuning and security

`OWNER_VOICE_THRESHOLD` in `config.py` starts at `0.55`. Scores vary with
microphones, room acoustics, distance, and the speaker model. Test with your
own voice and non-owner voices in realistic conditions before relying on it.
A stricter threshold reduces false accepts but can increase false rejections.

Voice matching is not proof of liveness. A recording or synthesized voice may
fool a speaker-verification model. Do not rely on it alone for payments,
password access, deleting data, sending messages, or other high-impact actions.
Use a separate PIN or operating-system authentication for those operations.
