"""
LUNA AI - MICROPHONE LISTENER

Features:
- Continuous microphone input
- Adaptive background-noise detection
- Speech start detection
- Longer pauses between words
- Audio pre-buffer to preserve first words
- Longer recordings
"""

import collections
import queue
import threading
import time

import numpy as np

import config


class Segmenter:
    """Convert microphone audio blocks into complete speech recordings."""

    def __init__(self, noise_level):

        self.noise = max(float(noise_level), 0.0001)

        self.silence_limit = max(
            1,
            int(
                config.SILENCE_SECONDS * 1000
                / config.BLOCK_MS
            )
        )

        self.max_blocks = max(
            1,
            int(
                config.MAX_SPEECH_SECONDS * 1000
                / config.BLOCK_MS
            )
        )

        self.min_blocks = max(
            1,
            int(
                config.MIN_SPEECH_SECONDS * 1000
                / config.BLOCK_MS
            )
        )

        # Keep half a second before speech starts.
        self.pre = collections.deque(
            maxlen=max(
                1,
                int(500 / config.BLOCK_MS)
            )
        )

        self.reset()

    def reset(self):

        self.speaking = False
        self.chunks = []

        self.silent = 0
        self.loud = 0

        self.pre.clear()

    @property
    def threshold(self):

        return max(
            config.MIN_VOLUME,
            self.noise * config.VOICE_SENSITIVITY
        )

    def feed(self, block):

        block = np.asarray(
            block,
            dtype=np.float32
        )

        rms = float(
            np.sqrt(
                np.mean(block * block)
            )
        )

        start_threshold = self.threshold

        # Use a lower threshold after speech starts.
        # This prevents quiet syllables from ending a recording.
        end_threshold = start_threshold * 0.65

        # =========================================
        # WAIT FOR SPEECH
        # =========================================

        if not self.speaking:

            if rms >= start_threshold:

                self.speaking = True

                self.chunks = (
                    list(self.pre) + [block.copy()]
                )

                self.pre.clear()

                self.loud = 1
                self.silent = 0

            else:

                self.pre.append(block.copy())

                # Slowly adapt to room noise.
                self.noise = (
                    0.97 * self.noise
                    + 0.03 * rms
                )

            return None

        # =========================================
        # RECORD SPEECH
        # =========================================

        self.chunks.append(block.copy())

        if rms >= end_threshold:

            self.loud += 1
            self.silent = 0

        else:

            self.silent += 1

        # =========================================
        # END DETECTION
        # =========================================

        finished = (
            self.silent >= self.silence_limit
            or len(self.chunks) >= self.max_blocks
        )

        if not finished:
            return None

        # Reject clicks and short microphone noises.
        if self.loud < self.min_blocks:

            self.reset()
            return None

        # Keep a short audio tail after the last detected speech.
        trailing_blocks = max(
            1,
            int(240 / config.BLOCK_MS)
        )

        keep = max(
            1,
            len(self.chunks) - self.silent + trailing_blocks
        )

        keep = min(
            keep,
            len(self.chunks)
        )

        audio = np.concatenate(
            self.chunks[:keep]
        ).astype(
            np.float32,
            copy=False
        )

        duration = (
            len(audio) / config.SAMPLE_RATE
        )

        print(
            f"\n[Recording captured: {duration:.1f}s]"
        )

        self.reset()

        return audio


class Listener:

    def __init__(self, mute=None):

        """
        mute:
            A threading.Event.
            Microphone input is ignored while Luna speaks.
        """

        self.mute = mute or threading.Event()

        self.q = queue.Queue()

        self.stream = None
        self.seg = None

        self.block = int(
            config.SAMPLE_RATE
            * config.BLOCK_MS
            / 1000
        )

    def _callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):

        if self.mute.is_set():
            return

        if status:
            pass

        self.q.put(
            indata[:, 0].copy()
        )

    def start(self):

        import sounddevice as sd

        self.stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.block,
            device=config.INPUT_DEVICE,
            callback=self._callback
        )

        self.stream.start()

        print(
            "\nCalibrating microphone. Please stay quiet..."
        )

        levels = []

        end = time.time() + 1.5

        while time.time() < end:

            try:

                audio = self.q.get(
                    timeout=0.5
                )

            except queue.Empty:

                continue

            rms = float(
                np.sqrt(
                    np.mean(audio * audio)
                )
            )

            levels.append(rms)

        noise = (
            float(np.median(levels))
            if levels
            else 0.003
        )

        self.seg = Segmenter(noise)

        print(
            f"\nMicrophone ready."
            f"\nBackground noise: {noise:.5f}"
            f"\nSpeech threshold: {self.seg.threshold:.5f}"
        )

    def stop(self):

        if self.stream is not None:

            self.stream.stop()
            self.stream.close()

            self.stream = None

    def flush(self):

        """Remove old audio after Luna finishes speaking."""

        while True:

            try:

                self.q.get_nowait()

            except queue.Empty:

                break

        if self.seg is not None:
            self.seg.reset()

    def listen(self, timeout=None):

        """
        Wait for speech to finish.

        timeout:
            Maximum time to wait for speech to START.
            Once speech starts, continue until the utterance ends.
        """

        if self.seg is None:

            raise RuntimeError(
                "Listener has not been started. Call start() first."
            )

        self.seg.reset()

        started = time.time()

        while True:

            try:

                block = self.q.get(
                    timeout=0.2
                )

            except queue.Empty:

                block = None

            if block is not None:

                audio = self.seg.feed(block)

                if audio is not None:

                    return audio

            # Do not stop an utterance merely because the
            # initial listening timeout has expired.
            if (
                timeout is not None
                and not self.seg.speaking
                and time.time() - started > timeout
            ):

                return None


if __name__ == "__main__":

    listener = Listener()

    listener.start()

    print(
        "\nSpeak naturally. Press Ctrl+C to stop."
    )

    try:

        while True:

            audio = listener.listen()

            if audio is None:
                continue

            duration = (
                len(audio)
                / config.SAMPLE_RATE
            )

            rms = float(
                np.sqrt(
                    np.mean(audio * audio)
                )
            )

            print(
                f"\nCaptured {duration:.1f} seconds"
                f" | Volume: {rms:.5f}"
            )

    except KeyboardInterrupt:

        print(
            "\nMicrophone test stopped."
        )

    finally:

        listener.stop()