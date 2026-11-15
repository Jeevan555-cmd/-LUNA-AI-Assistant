import time
import numpy as np
import sounddevice as sd
import soundfile as sf

import config


class Recorder:
    """Records mic audio between start() and stop(), saves it as a .wav file."""

    def __init__(self):
        self._chunks = []
        self._stream = None

    @property
    def active(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        self._chunks.append(indata.copy())

    def start(self):
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        """Stop recording. Returns the saved .wav path, or None if too short."""
        if self._stream is None:
            return None
        self._stream.stop()
        self._stream.close()
        self._stream = None

        if not self._chunks:
            return None
        audio = np.concatenate(self._chunks, axis=0)
        if len(audio) / config.SAMPLE_RATE < config.MIN_SECONDS:
            return None

        path = config.RECORDINGS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(path, audio, config.SAMPLE_RATE)
        return path
