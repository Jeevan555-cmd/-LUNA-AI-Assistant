
"""Luna's voice with shared playback reference for echo cancellation."""

import collections
import queue
import re
import subprocess
import threading
import time

import numpy as np
import soundfile as sf

import config


class AudioReference:
    """A thread-safe buffer containing audio played by Luna."""

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self._chunks = collections.deque()
        self._offset = 0
        self._lock = threading.Lock()

    def push(self, pcm, source_rate):
        """Add Piper playback audio, converted to microphone sample rate."""

        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            return

        # Piper provides int16 PCM. Convert to float32 in [-1, 1].
        audio = audio / 32768.0

        # Convert Piper's sample rate to the microphone sample rate.
        if source_rate != self.sample_rate:
            new_length = max(
                1,
                round(len(audio) * self.sample_rate / source_rate)
            )

            old_positions = np.arange(len(audio))
            new_positions = np.linspace(
                0,
                len(audio) - 1,
                new_length
            )

            audio = np.interp(
                new_positions,
                old_positions,
                audio
            ).astype(np.float32)

        with self._lock:
            self._chunks.append(audio)

    def read(self, number_of_samples):
        """Return the next playback-reference samples."""

        output = np.zeros(number_of_samples, dtype=np.float32)
        written = 0

        with self._lock:

            while written < number_of_samples and self._chunks:

                chunk = self._chunks[0]

                available = len(chunk) - self._offset
                needed = number_of_samples - written
                take = min(available, needed)

                output[written:written + take] = (
                    chunk[self._offset:self._offset + take]
                )

                written += take
                self._offset += take

                if self._offset >= len(chunk):
                    self._chunks.popleft()
                    self._offset = 0

        # Any unavailable samples remain zero-filled.
        return output


# Shared by Speaker and Listener.
audio_reference = AudioReference(config.SAMPLE_RATE)


class Speaker:

    def __init__(self):
        from piper import PiperVoice

        if not config.PIPER_VOICE.exists():
            raise FileNotFoundError(
                f"Voice file not found: {config.PIPER_VOICE}\n"
                f"Download it with: python -m piper.download_voices "
                f"{config.VOICE_NAME} --data-dir voices"
            )

        if not config.PIPER_VOICE.with_suffix(".onnx.json").exists():
            raise FileNotFoundError(
                f"The voice's .json file is missing next to "
                f"{config.PIPER_VOICE.name}."
            )

        self.voice = PiperVoice.load(str(config.PIPER_VOICE))
        self.sample_rate = self.voice.config.sample_rate

        self.syn_config = None

        try:
            from piper import SynthesisConfig

            self.syn_config = SynthesisConfig(
                length_scale=config.VOICE_LENGTH_SCALE
            )

        except Exception:
            pass

        self.speaking = threading.Event()
        self.first_audio_at = None

        self._pending = 0
        self._lock = threading.Lock()

        self._text_q = queue.Queue()
        self._audio_q = queue.Queue()

        self._recorded = []

        threading.Thread(
            target=self._synth_loop,
            daemon=True
        ).start()

        threading.Thread(
            target=self._play_loop,
            daemon=True
        ).start()

    def begin(self):
        """Start a new answer."""

        self._recorded = []
        self.first_audio_at = None

    def say(self, text):
        """Queue a sentence for speech."""

        text = re.sub(r"[*_#`]", "", text).strip()

        if not text:
            return

        with self._lock:
            self._pending += 1
            self.speaking.set()

        self._text_q.put(text)

    def wait(self):
        """Wait until all queued sentences have played."""

        self._text_q.join()
        self._audio_q.join()

    def save_reply_async(self):
        """Save the spoken answer as an MP3 in the background."""

        pcm = list(self._recorded)

        if not pcm:
            return

        sr = self.sample_rate

        threading.Thread(
            target=_save_mp3,
            args=(np.concatenate(pcm), sr),
            daemon=True
        ).start()

    def _done_one(self):

        with self._lock:
            self._pending -= 1

            if self._pending <= 0:
                self._pending = 0
                self.speaking.clear()

    def _synthesize(self, text):

        kwargs = (
            {"syn_config": self.syn_config}
            if self.syn_config is not None
            else {}
        )

        parts = []

        for chunk in self.voice.synthesize(text, **kwargs):

            raw = getattr(chunk, "audio_int16_bytes", None)

            if raw is not None:
                parts.append(
                    np.frombuffer(raw, dtype=np.int16)
                )

            else:
                arr = np.clip(
                    chunk.audio_float_array,
                    -1.0,
                    1.0
                )

                parts.append(
                    (arr * 32767).astype(np.int16)
                )

        return np.concatenate(parts) if parts else None

    def _synth_loop(self):

        while True:

            text = self._text_q.get()

            try:
                pcm = self._synthesize(text)

            except Exception as exc:
                print(f"\n(voice synthesis error: {exc})")
                pcm = None

            self._audio_q.put(pcm)
            self._text_q.task_done()

    def _play_loop(self):

        import sounddevice as sd

        stream = None

        while True:

            pcm = self._audio_q.get()

            try:

                if pcm is not None:

                    if stream is None:

                        stream = sd.OutputStream(
                            samplerate=self.sample_rate,
                            channels=1,
                            dtype="int16"
                        )

                        stream.start()

                    if self.first_audio_at is None:
                        self.first_audio_at = time.time()

                    self._recorded.append(pcm)

                    # IMPORTANT:
                    # Give the echo canceller the same audio
                    # that Luna is about to play.
                    audio_reference.push(
                        pcm,
                        self.sample_rate
                    )

                    # Play the audio.
                    stream.write(
                        np.ascontiguousarray(
                            pcm.reshape(-1, 1)
                        )
                    )

            except Exception as exc:
                print(f"\n(playback error: {exc})")

            finally:
                self._done_one()
                self._audio_q.task_done()


def _save_mp3(pcm, sample_rate):

    stamp = time.strftime("%Y%m%d_%H%M%S")

    wav = config.RESPONSES_DIR / f"{stamp}.wav"
    mp3 = config.RESPONSES_DIR / f"{stamp}.mp3"

    try:

        sf.write(str(wav), pcm, sample_rate)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(wav),
                str(mp3)
            ],
            check=True,
            capture_output=True
        )

        wav.unlink(missing_ok=True)

    except Exception as exc:
        print(
            f"\n(could not save mp3: {exc} "
            f"- the .wav is kept in {config.RESPONSES_DIR})"
        )


if __name__ == "__main__":

    print("Loading voice...")

    speaker = Speaker()
    speaker.begin()

    t0 = time.time()

    speaker.say("Hello boss. This is a voice test.")
    speaker.wait()

    print(
        f"Done. First sound came "
        f"{speaker.first_audio_at - t0:.2f}s after the request."
    )
