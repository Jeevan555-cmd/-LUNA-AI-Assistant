"""
LUNA AI - CENTRAL CONFIGURATION
"""

from pathlib import Path

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RECORDINGS_DIR = DATA_DIR / "recordings"
RESPONSES_DIR = DATA_DIR / "responses"

for directory in (RECORDINGS_DIR, RESPONSES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# =====================================================
# ASSISTANT IDENTITY
# =====================================================

ASSISTANT_OWNER = "Jeevan Poojary"
ASSISTANT_OWNER_NAME = "Jeevan"
ASSISTANT_NAME = "Luna"
USER_TITLE = "boss"

# =====================================================
# PERSONAL VOCABULARY
# =====================================================

WHISPER_VOCABULARY = [
    "Jeevan", "Jeevan Poojary", "Chetan", "Girish", "Mithun",
    "Shashank", "Sudeep", "Shriharsha", "Yashas", "Raghu",
    "Hemawati", "Suresh",
    "JavaScript", "JavaScript promises", "JavaScript programming",
    "JavaScript asynchronous programming", "JavaScript event loop",
    "async", "await", "Promise", "React", "React hooks", "Next.js",
    "Node.js", "Express.js", "Python", "MongoDB", "Ollama", "Whisper",
    "Piper", "Udupi", "Mangalore", "Mangaluru", "Karnataka",
]

NAME_CORRECTIONS = {
    "gireish": "Girish",
    "miton": "Mithun",
    "shri harsh": "Shriharsha",
}

# =====================================================
# CONTEXT-AWARE STT CORRECTIONS
# =====================================================

STT_CONTEXT_CORRECTIONS = {
    "java screens": {
        "replacement": "JavaScript",
        "context": [
            "promise", "promises", "async", "await", "event loop",
            "programming", "javascript",
        ],
    },
    "tapes of promises": {
        "replacement": "types of promises",
        "context": ["promise", "promises", "javascript"],
    },
    "oupi": {
        "replacement": "Udupi",
        "context": ["weather", "location", "city", "in oupi", "near oupi"],
    },
}

# =====================================================
# WAKE WORD AND CONVERSATION
# =====================================================

WAKE_WORDS = [
    "luna", "loona", "lunar", "luner", "looner", "lunna", "lunah", "lana",
]

ACTIVATE_WORDS = ["active", "activate", "wake up", "turn on"]

DEACTIVATE_WORDS = [
    "deactivate", "go to sleep", "sleep", "standby",
    "stop listening", "turn off",
]

GOODBYE_PHRASES = [
    "goodbye", "good bye", "bye", "that's all", "that is all",
    "go to sleep", "stop listening", "deactivate",
]

WAKE_GREETING = "Hello boss. I'm here. How can I help you?"
SLEEP_REPLY = (
    "Certainly, boss. I'll return to standby mode. "
    "Call me whenever you need me."
)

LISTEN_TIMEOUT = 8
FOLLOWUP_SECONDS = 8
CONFIRM_TIMEOUT = 12

# =====================================================
# MEMORY AND PERSONAL PROFILE
# =====================================================

ASK_BEFORE_SAVING = True
MEMORY_DB = DATA_DIR / "memory.db"
MEMORY_MESSAGES = 10
PROFILE_JSON = DATA_DIR / "profile.json"
PROFILE_VALUE_MAX_LENGTH = 200

# =====================================================
# MICROPHONE AND AUDIO
# =====================================================

SAMPLE_RATE = 16000
BLOCK_MS = 30
INPUT_DEVICE = None
VOICE_SENSITIVITY = 2.8
MIN_VOLUME = 0.006
SILENCE_SECONDS = 1.4
MIN_SPEECH_SECONDS = 0.4
MAX_SPEECH_SECONDS = 60

# =====================================================
# WHISPER - SPEECH TO TEXT
# =====================================================

WAKE_MODEL = "tiny.en"
WHISPER_MODEL = "tiny.en"
WAKE_BEAM = 1
WHISPER_BEAM = 1
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"
CPU_THREADS = 0
WHISPER_LANGUAGE = "en"

# =====================================================
# LLM - OLLAMA
# =====================================================

LLM_PROVIDER = "ollama"
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.2:1b"
LLM_MAX_TOKENS = 160
LLM_CONTEXT = 1024
LLM_TEMPERATURE = 0.3
LLM_KEEP_ALIVE = "30m"
LLM_TIMEOUT = 120

# =====================================================
# LUNA PERSONALITY AND SYSTEM INSTRUCTIONS
# =====================================================

SYSTEM_PROMPT = """
You are Luna, Jeevan's personal AI assistant.

IDENTITY:
Your name is Luna.
Address the user as boss.
Never use the user's personal name in normal replies.

PERSONALITY:
Be intelligent, friendly, natural and professional.
Speak like a real personal assistant.
Be respectful and confident.

RESPONSE RULES:
1. Answer the actual question directly.
2. Keep ordinary voice responses short.
3. Avoid unnecessary introductions.
4. Do not repeat the user's question.
5. Do not repeatedly say hello or thank you.
6. Never add unnecessary closing sentences.
7. Explain technical topics with simple examples.
8. Give longer explanations only when requested.
9. Ask for clarification if the request is unclear.
10. When explaining programming, prioritize technical correctness.
11. Never invent JavaScript methods, properties, APIs or syntax.
12. Include complete, runnable code examples when requested.
13. Explain important limitations and common mistakes.
14. Never claim that an operation succeeded unless it was verified.
15. For long explanations, organize the response into clear sections.

TRANSCRIPTION:
Speech recognition may contain mistakes.
Correct obvious mistakes using conversation context.
Understand the user's intended meaning.
Preserve technical terms and names.
Never invent missing information.

MEMORY:
Use previous conversations when relevant.
Never invent personal information.
Ask permission before saving personal details.

You are Luna. Be concise, intelligent and helpful.
"""

# =====================================================
# VOICE - PIPER TTS
# =====================================================

VOICE_NAME = "en_US-ryan-medium"
PIPER_VOICE = BASE_DIR / "voices" / f"{VOICE_NAME}.onnx"
VOICE_LENGTH_SCALE = 0.95

# =====================================================
# LOCAL OWNER VOICE VERIFICATION
# =====================================================

SPEAKER_MODEL_DIR = DATA_DIR / "models"
SPEAKER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

SPEAKER_MODEL = (
    SPEAKER_MODEL_DIR / "wespeaker_en_voxceleb_resnet34.onnx"
)

OWNER_VOICE_PROFILE = DATA_DIR / "owner_voice.json"
OWNER_VERIFICATION_ENABLED = True
OWNER_ENROLLMENT_SAMPLES = 3
OWNER_MIN_UTTERANCE_SECONDS = 3.0
OWNER_VERIFY_TIMEOUT = 15
OWNER_VOICE_THRESHOLD = 0.55
OWNER_VOICE_NUM_THREADS = 1
