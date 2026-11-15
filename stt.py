
"""
Luna Speech-to-Text Engine

Uses Faster Whisper to:
1. Recognize wake words.
2. Transcribe spoken questions.
3. Improve recognition using personal vocabulary.
4. Correct known names and context-dependent terms.
5. Reject obvious Whisper repetition loops.
"""

import re
import time

import numpy as np

import config
import knowledge


_models = {}


# =====================================================
# WHISPER MODEL LOADING
# =====================================================

def get_model(name):

    if name not in _models:

        from faster_whisper import WhisperModel

        print(f"\nLoading Whisper model: {name}")

        _models[name] = WhisperModel(
            name,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
            cpu_threads=config.CPU_THREADS,
        )

        print(f"Whisper model ready: {name}")

    return _models[name]


def load():

    print("\nLoading Luna speech recognition...")

    silence = np.zeros(
        config.SAMPLE_RATE,
        dtype=np.float32
    )

    for name in set((
        config.WAKE_MODEL,
        config.WHISPER_MODEL
    )):
        transcribe(silence, name)

    print("\nLuna Whisper is ready!")


# =====================================================
# PERSONAL VOCABULARY
# =====================================================

def get_vocabulary():

    vocabulary = getattr(
        config,
        "WHISPER_VOCABULARY",
        []
    )

    return [
        str(word).strip()
        for word in vocabulary
        if str(word).strip()
    ]


# =====================================================
# NAME SPELLING CORRECTION
# =====================================================

def correct_names(text):

    corrections = getattr(
        config,
        "NAME_CORRECTIONS",
        {}
    )

    for wrong, correct in corrections.items():

        pattern = re.compile(
            r"(?<!\w)" +
            re.escape(wrong) +
            r"(?!\w)",
            re.IGNORECASE
        )

        text = pattern.sub(
            lambda match: correct,
            text
        )

    return text


# =====================================================
# REPETITION DETECTION
# =====================================================

def has_repetition_loop(text):
    """
    Detect obvious repeated Whisper hallucinations.

    A phrase repeated three or more times across most
    of a transcript is treated as unreliable.
    """

    words = re.findall(
        r"\b[\w']+\b",
        text.lower()
    )

    if len(words) < 9:
        return False

    # Detect repeated phrases of 2 to 8 words.
    for size in range(2, 9):

        for start in range(len(words) - size * 2 + 1):

            phrase = words[start:start + size]

            repetitions = 1
            position = start + size

            while (
                position + size <= len(words)
                and words[position:position + size] == phrase
            ):
                repetitions += 1
                position += size

            covered_words = repetitions * size

            if (
                repetitions >= 3
                and covered_words >= len(words) * 0.60
            ):
                return True

    # Detect a single word repeated excessively.
    for word in set(words):

        if words.count(word) >= 7:
            return True

    return False


# =====================================================
# CLEAN SMALL REPEATED WORDS
# =====================================================

def clean_adjacent_repetitions(text):
    """
    Collapse immediate accidental duplicates:
    'I I want' -> 'I want'
    'the the book' -> 'the book'

    Does not remove normal repeated words separated
    by other words.
    """

    return re.sub(
        r"\b(\w+)(?:[\s,]+\1\b)+",
        r"\1",
        text,
        flags=re.IGNORECASE
    )


# =====================================================
# CONTEXT-AWARE CORRECTION
# =====================================================

def correct_context_terms(text):

    rules = getattr(
        config,
        "STT_CONTEXT_CORRECTIONS",
        {}
    )

    lowered = text.lower()

    for wrong, rule in rules.items():

        replacement = rule.get("replacement")

        required_context = rule.get(
            "context",
            []
        )

        pattern = re.compile(
            r"(?<!\w)" +
            re.escape(wrong) +
            r"(?!\w)",
            re.IGNORECASE
        )

        if not pattern.search(text):
            continue

        # Only correct when supporting context exists.
        if required_context and not any(
            term.lower() in lowered
            for term in required_context
        ):
            continue

        text = pattern.sub(
            replacement,
            text
        )

        lowered = text.lower()

    return text


# =====================================================
# COMPLETE TRANSCRIPT CORRECTION
# =====================================================

def correct_transcript(text):

    if not text:
        return ""

    original = text.strip()

    # Reject obvious hallucination loops.
    if has_repetition_loop(original):

        print(
            "\n[STT warning: Repeated speech detected. "
            "Transcript rejected.]"
        )

        return ""

    corrected = clean_adjacent_repetitions(
        original
    )

    corrected = correct_names(
        corrected
    )

    corrected = correct_context_terms(
        corrected
    )

    corrected = re.sub(
        r"\s+",
        " ",
        corrected
    ).strip()

    if corrected != original:

        print(
            f"\n[STT corrected: {corrected}]"
        )

    return corrected


# =====================================================
# SPEECH TRANSCRIPTION
# =====================================================

def transcribe(audio, model_name) -> str:

    started = time.perf_counter()

    is_wake = model_name == config.WAKE_MODEL

    assistant_name = config.ASSISTANT_NAME

    hint = (
        f"{assistant_name}, "
        f"hello {assistant_name}. "
        f"{assistant_name}, activate."
    )

    if not is_wake:

        known_words = get_vocabulary()

        if known_words:

            hint += (
                " Names and technical words: "
                + ", ".join(known_words)
                + "."
            )

    try:

        segments, info = get_model(
            model_name
        ).transcribe(
            np.asarray(audio, dtype=np.float32),

            language=getattr(
                config,
                "WHISPER_LANGUAGE",
                "en"
            ),

            beam_size=(
                config.WAKE_BEAM
                if is_wake
                else config.WHISPER_BEAM
            ),

            vad_filter=False,

            condition_on_previous_text=False,

            without_timestamps=True,

            temperature=0.0,

            initial_prompt=hint,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        if not is_wake and text:
            text = correct_transcript(text)

        return text

    finally:

        elapsed = time.perf_counter() - started

        print(
            f"\n[Whisper transcription: {elapsed:.2f}s]"
        )
