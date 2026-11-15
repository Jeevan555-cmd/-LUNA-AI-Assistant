
"""
LUNA AI - MAIN APPLICATION

Voice-first personal AI assistant.

Features:
- Wake word detection
- Activate / deactivate
- Continuous conversations
- Ollama AI responses
- Piper voice output
- Personal memory confirmation
- Profile management
- Google Chrome voice automation
- Graceful error handling

Run:
    python main.py
"""

import json
import re
import threading
import time

import soundfile as sf

import config
import knowledge
import llm
import stt
import chrome_control


# =====================================================
# COMMAND DETECTION
# =====================================================

GOODBYE_PHRASES = getattr(
    config,
    "DEACTIVATE_WORDS",
    config.GOODBYE_PHRASES
)

_SLEEP_COMMANDS = {
    phrase.strip().lower()
    for phrase in GOODBYE_PHRASES
}

_SLEEP_RE = re.compile(
    r"^\s*(?:(?:please|luna|okay|ok|boss)\s+)*"
    r"(" + "|".join(
        re.escape(p)
        for p in sorted(
            _SLEEP_COMMANDS,
            key=len,
            reverse=True
        )
    ) + r")"
    r"(?:\s+please)?[.!?]*\s*$",
    re.IGNORECASE
)

_SHOW_RE = re.compile(
    r"\b(show|display|print|list)\b.*"
    r"\b(data|profile|memory|information|info)\b"
    r"|\bwhat do you (know|remember) about me\b",
    re.IGNORECASE
)

_NO_RE = re.compile(
    r"\b(no|nope|nah|don't|do not|dont|cancel|wrong|incorrect|not)\b",
    re.IGNORECASE
)

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|correct|right|"
    r"please|save|go ahead|do it)\b",
    re.IGNORECASE
)


def yes_no(text):
    """Return True, False or None for unclear answers."""

    text = text.strip().lower()

    if _NO_RE.search(text):
        return False

    if _YES_RE.search(text):
        return True

    return None


def is_sleep_command(text):
    """Only treat a short, direct command as deactivation."""

    return bool(
        _SLEEP_RE.fullmatch(text.strip())
    )


# =====================================================
# RECORDING
# =====================================================

def _save_recording(audio):
    """Save audio in the background."""

    def work():

        try:

            ms = int(time.time() * 1000) % 1000

            filename = (
                f"{time.strftime('%Y%m%d_%H%M%S')}"
                f"_{ms:03d}.wav"
            )

            path = config.RECORDINGS_DIR / filename

            sf.write(
                str(path),
                audio,
                config.SAMPLE_RATE
            )

        except Exception as error:

            print(
                f"\n(Recording error: {error})"
            )

    threading.Thread(
        target=work,
        daemon=True
    ).start()


# DISPLAY

def _print_changes(title, changes):

    print(f"\n--- {title} ---")

    print(
        json.dumps(
            changes,
            indent=2,
            ensure_ascii=False
        )
    )

    print("-" * (len(title) + 8))


# =====================================================
# LUNA ASSISTANT
# =====================================================

class Luna:

    def __init__(
        self,
        listener,
        speaker,
        assistant,
        owner_verifier=None
    ):

        self.listener = listener
        self.speaker = speaker
        self.assistant = assistant
        self.owner_verifier = owner_verifier

        self.running = True

    # =================================================
    # SPEECH
    # =================================================

    def say(self, text):
        """Speak and wait until speech finishes."""

        self.speaker.begin()

        self.speaker.say(text)

        self.speaker.wait()

        time.sleep(0.2)

        self.listener.flush()

    # =================================================
    # MICROPHONE
    # =================================================

    def _heard(self, timeout=None, save=True):
        """Listen and optionally save the recording."""

        try:

            audio = self.listener.listen(
                timeout=timeout
            )

            if audio is None:
                return None, None

            t_end = (
                time.time()
                - config.SILENCE_SECONDS
            )

            if save:
                _save_recording(audio)

            return audio, t_end

        except Exception as error:

            print(
                f"\n(Microphone error: {error})"
            )

            return None, None

    # =================================================
    # SPEECH RECOGNITION
    # =================================================

    def _transcribe(self, audio, model=None):
        """Transcribe speech and print recognition time."""

        started = time.perf_counter()

        try:

            result = stt.transcribe(
                audio,
                model or config.WHISPER_MODEL
            )

            transcript = (
                result.strip()
                if result
                else ""
            )

            if transcript:

                print(
                    f"\n[Transcript: {transcript}]"
                )

            return transcript

        except Exception as error:

            print(
                f"\n(Speech recognition error: {error})"
            )

            return ""

        finally:

            print(
                f"\n[Whisper transcription: "
                f"{time.perf_counter() - started:.2f}s]"
            )

    # =================================================
    # WAIT FOR WAKE WORD
    # =================================================

    def wait_for_wake(self):
        """
        Wait until Luna's name or activation phrase is heard.

        Supports:
            Luna
            Hey Luna
            Luna, how are you?
            Activate
            Luna, activate
        """

        while self.running:

            audio, _ = self._heard(
                save=False
            )

            if audio is None:
                continue

            text = self._transcribe(
                audio,
                config.WAKE_MODEL
            )

            if not text:
                continue

            print(
                f"\n(heard: {text})"
            )

            if not llm.is_addressed(text):
                continue

            rest = llm.strip_wake(text)

            t_end = (
                time.time()
                - config.SILENCE_SECONDS
            )

            return rest, t_end

        return "", None

    # =================================================
    # YES / NO CONFIRMATION
    # =================================================

    def _ask_yes_no(self, sentence):
        """Ask permission and wait for a clear answer."""

        self.say(sentence)

        for attempt in range(2):

            audio, _ = self._heard(
                config.CONFIRM_TIMEOUT
            )

            if audio is None:
                return None

            text = self._transcribe(audio)

            print(
                f"\n(confirmation: {text})"
            )

            verdict = yes_no(text)

            if verdict is not None:
                return verdict

            if attempt == 0:

                self.say(
                    "Sorry boss, please say yes or no."
                )

        return None

    # =================================================
    # OWNER VOICE VERIFICATION
    # =================================================

    def authenticate_owner(self):
        """Verify a fresh speech sample after the wake word."""

        if self.owner_verifier is None:
            return True

        for attempt in range(2):

            self.say(
                "Voice verification, boss. "
                "Please speak naturally for a few seconds."
            )

            audio, _ = self._heard(
                timeout=config.OWNER_VERIFY_TIMEOUT,
                save=False
            )

            if audio is None:

                self.say(
                    "I didn't receive a voice sample. "
                    "Returning to standby."
                )

                return False

            try:

                matched, score = (
                    self.owner_verifier.verify(audio)
                )

                if matched:

                    print(
                        "\n(Owner voice verified.)"
                    )

                    return True

                print(
                    f"\n(Owner voice did not match; "
                    f"similarity {score:.3f}.)"
                )

            except Exception as error:

                print(
                    f"\n(Voice verification error: {error})"
                )

                self.say(
                    "I couldn't verify the voice sample. "
                    "For your privacy, I won't continue this session."
                )

                return False

            if attempt == 0:

                self.say(
                    "Sorry boss, I couldn't confidently "
                    "recognize your voice. Please try once more."
                )

        self.say(
            "Voice verification failed. "
            "I'm returning to standby."
        )

        return False

    # =================================================
    # PERSONAL MEMORY
    # =================================================

    def _offer_to_save(self, question):
        """
        Analyze personal information.

        Never saves without explicit approval.
        """

        try:

            changes = self.assistant.propose_changes(
                question
            )

            if not changes:
                return

            if not config.ASK_BEFORE_SAVING:

                result = knowledge.apply_changes(
                    changes
                )

                if result:

                    _print_changes(
                        "Memory updated",
                        result
                    )

                return

            print(
                "\n--- Proposed profile updates ---"
            )

            print(
                json.dumps(
                    changes,
                    indent=2,
                    ensure_ascii=False
                )
            )

            sentence = (
                "Boss, this information may be useful "
                "for your personal profile. "
                "Would you like me to save it?"
            )

            print(
                f"\n{config.ASSISTANT_NAME}: {sentence}"
            )

            verdict = self._ask_yes_no(sentence)

            if verdict is True:

                result = knowledge.apply_changes(
                    changes
                )

                if result:

                    _print_changes(
                        "Memory updated",
                        result
                    )

                    self.say(
                        "Saved successfully, boss."
                    )

                else:

                    self.say(
                        "Boss, there was nothing new to update."
                    )

            elif verdict is False:

                print(
                    "(Memory not saved.)"
                )

                self.say(
                    "Understood boss. I won't save it."
                )

            else:

                print(
                    "(No confirmation received.)"
                )

                self.say(
                    "I didn't receive a clear confirmation, "
                    "so I haven't saved anything."
                )

        except Exception as error:

            print(
                f"\n(Memory error: {error})"
            )

            self.say(
                "Sorry boss, I couldn't update your profile."
            )

    # =================================================
    # SHOW PROFILE
    # =================================================

    def show_profile(self):

        try:

            data = knowledge.load()

            print(
                "\n========== LUNA PROFILE =========="
            )

            print(
                json.dumps(
                    data,
                    indent=2,
                    ensure_ascii=False
                )
            )

            print(
                "=================================="
            )

            self.say(
                "Here is your personal profile, boss. "
                "You can see it on the screen."
            )

        except Exception as error:

            print(
                f"\n(Profile error: {error})"
            )

            self.say(
                "Sorry boss, I couldn't access your profile."
            )

    # =================================================
    # ANSWER QUESTIONS
    # =================================================

    def respond(self, question, t_end):

        if not question.strip():
            return

        print(
            f"\nYou: {question}"
        )

        # ---------------------------------------------
        # CHROME VOICE AUTOMATION
        # ---------------------------------------------

        try:

            chrome_response = chrome_control.handle_command(
                question
            )

        except Exception as error:

            print(
                f"\n(Chrome automation error: {error})"
            )

            chrome_response = (
                "Sorry boss, I couldn't complete that Chrome command."
            )

        if chrome_response is not None:

            print(
                f"\n{config.ASSISTANT_NAME}: {chrome_response}"
            )

            self.say(chrome_response)

            return

        # ---------------------------------------------
        # DIRECT PROFILE REQUEST
        # ---------------------------------------------

        if _SHOW_RE.search(question):

            self.show_profile()

            return

        # ---------------------------------------------
        # NORMAL AI RESPONSE
        # ---------------------------------------------

        self.speaker.begin()

        print(
            f"\n{config.ASSISTANT_NAME}: ",
            end="",
            flush=True
        )

        try:

            for sentence in self.assistant.stream_sentences(
                question
            ):

                if not sentence.strip():
                    continue

                print(
                    sentence + " ",
                    end="",
                    flush=True
                )

                self.speaker.say(sentence)

        except Exception as error:

            print(
                f"\n(LLM error: {error})"
            )

            self.speaker.say(
                "Sorry boss, I'm having trouble "
                "processing your request."
            )

        print()

        self.speaker.wait()

        time.sleep(0.2)

        self.listener.flush()

        # Performance information.
        first = getattr(
            self.speaker,
            "first_audio_at",
            None
        )

        if first and t_end:

            print(
                f"(Response started in "
                f"{first - t_end:.1f} seconds)"
            )

        # Save spoken reply asynchronously.
        try:

            self.speaker.save_reply_async()

        except Exception as error:

            print(
                f"\n(Audio saving error: {error})"
            )

        # Analyze personal information after answering.
        self._offer_to_save(question)

    # =================================================
    # MAIN CONVERSATION LOOP
    # =================================================

    def run(self):

        print(
            "\nLUNA IS READY."
        )

        print(
            f"Say '{config.ASSISTANT_NAME}' "
            "to wake her."
        )

        while self.running:

            # STANDBY MODE
            question, t_end = self.wait_for_wake()

            if not self.running:
                break

            if config.OWNER_VERIFICATION_ENABLED:

                if not self.authenticate_owner():

                    print(
                        "\n(Access denied. Luna remains in standby.)"
                    )

                    continue

            if not question:

                self.say(
                    config.WAKE_GREETING
                )

            timeout = config.LISTEN_TIMEOUT

            # ACTIVE MODE
            while self.running:

                if not question:

                    audio, t_end = self._heard(
                        timeout
                    )

                    if audio is None:

                        print(
                            "\n(No response. Returning to standby.)"
                        )

                        break

                    question = self._transcribe(
                        audio
                    )

                    if not question:
                        continue

                    # Remove a wake word if repeated.
                    question = llm.strip_wake(
                        question
                    )

                    if not question:
                        continue

                # DEACTIVATION
                if is_sleep_command(question):

                    self.say(
                        config.SLEEP_REPLY
                    )

                    break

                # ANSWER
                self.respond(
                    question,
                    t_end
                )

                question = ""

                timeout = config.FOLLOWUP_SECONDS

            print(
                f"\nStandby. Say "
                f"'{config.ASSISTANT_NAME}' to wake me."
            )

    # =================================================
    # SHUTDOWN
    # =================================================

    def stop(self):

        self.running = False


# =====================================================
# APPLICATION ENTRY POINT
# =====================================================

def _console_yes(prompt):
    """Explicit, session-scoped consent through the local console."""

    answer = input(prompt).strip().lower()

    return answer in {"y", "yes"}


def _enroll_owner_voice(listener, speaker, verifier):
    """Capture temporary samples and store only the averaged voice embedding."""

    required = int(
        config.OWNER_ENROLLMENT_SAMPLES
    )

    samples = []

    print(
        "\nOWNER VOICE ENROLLMENT"
    )

    print(
        f"Luna needs {required} separate samples. "
        "Speak naturally for at least "
        f"{config.OWNER_MIN_UTTERANCE_SECONDS:.0f} seconds each."
    )

    print(
        "Your raw enrollment recordings will not be saved."
    )

    for index in range(required):

        accepted = False

        for attempt in range(2):

            speaker.begin()

            speaker.say(
                f"Sample {index + 1} of {required}. "
                "Please say a few natural sentences now."
            )

            speaker.wait()

            listener.flush()

            try:

                audio = listener.listen(
                    timeout=config.OWNER_VERIFY_TIMEOUT
                )

            except Exception as error:

                print(
                    f"Microphone capture failed: {error}"
                )

                return False

            if audio is None:

                print(
                    "No speech detected. Please try again."
                )

                continue

            duration = (
                len(audio)
                / float(config.SAMPLE_RATE)
            )

            if duration < float(
                config.OWNER_MIN_UTTERANCE_SECONDS
            ):

                print(
                    f"That sample was {duration:.1f} seconds. "
                    "Please speak for longer."
                )

                continue

            samples.append(audio)

            accepted = True

            print(
                f"Captured sample {index + 1}/{required}."
            )

            break

        if not accepted:

            print(
                "Enrollment stopped; no voice profile was created."
            )

            return False

    try:

        verifier.enroll(samples)

    except Exception as error:

        print(
            f"Could not create the owner voice profile: {error}"
        )

        return False

    samples.clear()

    print(
        f"Owner voice profile saved locally: "
        f"{config.OWNER_VOICE_PROFILE}"
    )

    return True


# =====================================================
# MAIN FUNCTION
# =====================================================

def main():

    import tts

    from listener import Listener

    from owner_verification import (
        OwnerVerifier,
        OwnerVoiceError
    )

    print(
        "=" * 45
    )

    print(
        "           LUNA AI ASSISTANT"
    )

    print(
        "=" * 45
    )

    if not _console_yes(
        "\nAllow Luna to use your microphone for this session? [y/N]: "
    ):

        print(
            "Microphone permission was not granted. Luna will exit."
        )

        return

    print(
        "\nLoading speech recognition..."
    )

    stt.load()

    print(
        "Loading Piper voice..."
    )

    speaker = tts.Speaker()

    print(
        "Loading AI assistant..."
    )

    assistant = llm.Assistant()

    assistant.warm_up()

    try:

        owner_verifier = (
            OwnerVerifier()
            if config.OWNER_VERIFICATION_ENABLED
            else None
        )

    except OwnerVoiceError as error:

        print(
            f"\nOwner verification setup error: {error}"
        )

        print(
            "Install the speaker-verification dependency and model, "
            "then start Luna again."
        )

        return

    try:

        changed = knowledge.sync()

        if changed:

            _print_changes(
                "Profile synchronized",
                changed
            )

        print(
            f"\nMemory entries: {len(knowledge.load())}"
        )

        print(
            f"Profile location: {config.PROFILE_JSON}"
        )

    except Exception as error:

        print(
            f"\n(Profile initialization error: {error})"
        )

    listener = Listener(
        mute=speaker.speaking
    )

    listener_started = False

    try:

        listener.start()

        listener_started = True

        if (
            owner_verifier is not None
            and not owner_verifier.enrolled
        ):

            print(
                "\nNo owner voice profile was found."
            )

            if not _console_yes(
                "Enroll your voice locally now? [y/N]: "
            ):

                print(
                    "Owner enrollment is required while voice verification "
                    "is enabled. Luna will exit."
                )

                return

            if not _enroll_owner_voice(
                listener,
                speaker,
                owner_verifier
            ):

                print(
                    "Voice enrollment did not complete. Luna will exit."
                )

                return

        luna = Luna(
            listener,
            speaker,
            assistant,
            owner_verifier=owner_verifier
        )

        print(
            "\nLuna is running."
        )

        print(
            "Say 'Hey Luna' to begin."
        )

        print(
            "Chrome voice automation is enabled."
        )

        print(
            "Owner voice verification is enabled."
        )

        print(
            "Microphone access ends when Luna exits."
        )

        print(
            "Press Ctrl+C to exit."
        )

        luna.run()

    except KeyboardInterrupt:

        print(
            "\n\nLuna is shutting down."
        )

        print(
            "Goodbye boss."
        )

    finally:

        if listener_started:
            listener.stop()


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":
    main()
