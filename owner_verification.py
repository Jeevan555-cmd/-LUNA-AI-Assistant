
"""
LUNA AI - OWNER VOICE VERIFICATION

Uses WeSpeaker ONNX speaker embeddings through sherpa-onnx.

Features:
- Audio quality checks
- Leading/trailing silence trimming
- Consistent embedding normalization
- Cosine similarity verification
- Enrollment consistency diagnostics
- Local embedding storage

Security note:
Speaker verification is probabilistic. It is not liveness detection
and can be fooled by recordings or voice clones.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import config


class OwnerVoiceError(RuntimeError):
    """Raised when owner voice verification cannot be completed."""


class OwnerVerifier:

    def __init__(self):

        self.model_path = Path(config.SPEAKER_MODEL)

        if not self.model_path.is_file():
            raise OwnerVoiceError(
                f"Speaker model is missing: {self.model_path}"
            )

        try:
            import sherpa_onnx

        except ImportError as exc:
            raise OwnerVoiceError(
                "sherpa-onnx is not installed in this virtual environment."
            ) from exc

        extractor_config = (
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.model_path),
                num_threads=int(config.OWNER_VOICE_NUM_THREADS),
                debug=False,
                provider="cpu",
            )
        )

        if not extractor_config.validate():
            raise OwnerVoiceError(
                f"Invalid speaker model configuration: {extractor_config}"
            )

        print("Loading owner voice model...")

        self.extractor = (
            sherpa_onnx.SpeakerEmbeddingExtractor(
                extractor_config
            )
        )

        self.profile_path = Path(config.OWNER_VOICE_PROFILE)

        self.sample_rate = int(config.SAMPLE_RATE)

        self.template = self._load_template()

        print("Owner voice model ready.")

    @property
    def enrolled(self) -> bool:
        return self.template is not None

    # =================================================
    # EMBEDDING NORMALIZATION
    # =================================================

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:

        vector = np.asarray(
            vector,
            dtype=np.float32
        ).reshape(-1)

        if vector.size == 0:
            raise OwnerVoiceError(
                "The speaker model returned an empty embedding."
            )

        if not np.all(np.isfinite(vector)):
            raise OwnerVoiceError(
                "The speaker embedding contains invalid values."
            )

        norm = float(np.linalg.norm(vector))

        if norm < 1e-8:
            raise OwnerVoiceError(
                "The speaker model returned an invalid embedding."
            )

        return vector / norm

    # =================================================
    # LOAD SAVED PROFILE
    # =================================================

    def _load_template(self):

        if not self.profile_path.exists():
            return None

        try:

            payload = json.loads(
                self.profile_path.read_text(
                    encoding="utf-8"
                )
            )

            if payload.get("model") != self.model_path.name:
                raise OwnerVoiceError(
                    "The saved voice profile uses a different speaker model. "
                    "Back up the profile and enroll again."
                )

            vector = self._normalize(
                np.asarray(
                    payload["embedding"],
                    dtype=np.float32
                )
            )

            if vector.size != self.extractor.dim:
                raise OwnerVoiceError(
                    "The saved voice profile has an unexpected "
                    "embedding size. Enroll again."
                )

            print("Saved owner voice profile loaded.")

            return vector

        except OwnerVoiceError:
            raise

        except Exception as exc:
            raise OwnerVoiceError(
                f"Could not read owner voice profile: {exc}"
            ) from exc

    # =================================================
    # AUDIO PREPROCESSING
    # =================================================

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Clean the audio before speaker embedding extraction.

        - Converts audio to mono float32.
        - Removes DC offset.
        - Checks recording quality.
        - Trims leading and trailing silence.
        """

        samples = np.asarray(
            audio,
            dtype=np.float32
        ).reshape(-1)

        if samples.size == 0:
            raise OwnerVoiceError(
                "No audio was received."
            )

        if not np.all(np.isfinite(samples)):
            raise OwnerVoiceError(
                "The microphone recording contains invalid values."
            )

        # Remove DC offset.
        samples = samples - np.mean(samples)

        # Check for clipping.
        peak = float(np.max(np.abs(samples)))

        if peak > 0.999:
            print(
                "(Voice warning: microphone recording may be clipping.)"
            )

        # Initial volume check.
        rms = float(
            np.sqrt(np.mean(samples * samples))
        )

        if rms < 0.002:
            raise OwnerVoiceError(
                "Recording is too quiet. Move closer to the microphone "
                "and speak clearly."
            )

        # Divide audio into 20 ms frames.
        frame_size = int(
            self.sample_rate * 0.020
        )

        if samples.size < frame_size:
            raise OwnerVoiceError(
                "Voice recording is too short."
            )

        frame_count = samples.size // frame_size

        usable = samples[:frame_count * frame_size]

        frames = usable.reshape(
            frame_count,
            frame_size
        )

        frame_rms = np.sqrt(
            np.mean(frames * frames, axis=1)
        )

        maximum_rms = float(
            np.max(frame_rms)
        )

        # Ignore quiet background sections.
        threshold = max(
            0.0025,
            maximum_rms * 0.08
        )

        active = np.where(
            frame_rms >= threshold
        )[0]

        if active.size == 0:
            raise OwnerVoiceError(
                "No clear speech was detected."
            )

        # Preserve 150 ms around the detected speech.
        margin = int(
            0.150 * self.sample_rate / frame_size
        )

        start_frame = max(
            0,
            int(active[0]) - margin
        )

        end_frame = min(
            frame_count,
            int(active[-1]) + margin + 1
        )

        start_sample = start_frame * frame_size
        end_sample = min(
            samples.size,
            end_frame * frame_size
        )

        cleaned = np.ascontiguousarray(
            samples[start_sample:end_sample],
            dtype=np.float32
        )

        duration = (
            len(cleaned) / float(self.sample_rate)
        )

        minimum = float(
            config.OWNER_MIN_UTTERANCE_SECONDS
        )

        

        print(
            f"(Voice sample duration after trimming: "
            f"{duration:.2f}s)"
        )

        if duration < minimum:
            raise OwnerVoiceError(
                f"Only {duration:.1f} seconds of clear speech remained. "
                f"Please provide at least {minimum:.0f} seconds "
                "of natural speech."
            )

        final_rms = float(
            np.sqrt(np.mean(cleaned * cleaned))
        )

        if final_rms < 0.002:
            raise OwnerVoiceError(
                "The cleaned voice recording is too quiet."
            )

        return cleaned

    # =================================================
    # GENERATE SPEAKER EMBEDDING
    # =================================================

    def embedding(self, audio: np.ndarray) -> np.ndarray:

        samples = self._prepare_audio(audio)

        stream = self.extractor.create_stream()

        stream.accept_waveform(
            sample_rate=self.sample_rate,
            waveform=samples
        )

        stream.input_finished()

        if not self.extractor.is_ready(stream):
            raise OwnerVoiceError(
                "The speaker model could not process this recording. "
                "Please record a longer sample."
            )

        vector = np.asarray(
            self.extractor.compute(stream),
            dtype=np.float32
        ).reshape(-1)

        if vector.size != self.extractor.dim:
            raise OwnerVoiceError(
                "The speaker model returned an unexpected "
                "embedding dimension."
            )

        return self._normalize(vector)

    # =================================================
    # ENROLLMENT
    # =================================================

    def enroll(self, audio_samples: list[np.ndarray]) -> None:

        required = int(
            config.OWNER_ENROLLMENT_SAMPLES
        )

        if len(audio_samples) < required:
            raise OwnerVoiceError(
                f"Enrollment requires {required} separate voice samples."
            )

        print("\nGenerating speaker embeddings...")

        vectors = []

        for index, audio in enumerate(
            audio_samples[:required],
            start=1
        ):

            print(
                f"Processing enrollment sample "
                f"{index}/{required}..."
            )

            vectors.append(
                self.embedding(audio)
            )

        matrix = np.stack(vectors)

        # Check consistency between enrollment samples.
        similarities = matrix @ matrix.T

        pair_scores = []

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                pair_scores.append(
                    float(similarities[i, j])
                )

        if pair_scores:
            print(
                "(Enrollment consistency: "
                f"{np.mean(pair_scores):.3f})"
            )

        # Average all normalized embeddings.
        template = self._normalize(
            np.mean(matrix, axis=0)
        )

        payload = {
            "version": 2,
            "model": self.model_path.name,
            "embedding": template.astype(
                float
            ).tolist()
        }

        self.profile_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary = self.profile_path.with_suffix(
            ".tmp"
        )

        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        os.replace(
            temporary,
            self.profile_path
        )

        self.template = template

        print("\nOwner voice enrollment completed.")
        print(
            f"Profile saved: {self.profile_path}"
        )

    # =================================================
    # VOICE VERIFICATION
    # =================================================

    def verify(self, audio: np.ndarray) -> tuple[bool, float]:

        if self.template is None:
            raise OwnerVoiceError(
                "No owner voice has been enrolled."
            )

        print("\nAnalyzing owner voice...")

        candidate = self.embedding(audio)

        # Both embeddings are normalized.
        # Dot product therefore equals cosine similarity.
        score = float(
            np.dot(
                self.template,
                candidate
            )
        )

        score = float(
            np.clip(score, -1.0, 1.0)
        )

        threshold = float(
            config.OWNER_VOICE_THRESHOLD
        )

        matched = score >= threshold

        print(
            f"(Voice similarity: {score:.4f})"
        )

        print(
            f"(Required similarity: {threshold:.4f})"
        )

        print(
            f"(Verification result: "
            f"{'MATCH' if matched else 'NO MATCH'})"
        )

        return matched, score

   # DELETE ENROLLMENT

    def delete_enrollment(self) -> bool:
        """Remove the local voice template."""

        existed = self.profile_path.exists()

        self.profile_path.unlink(
            missing_ok=True
        )

        self.template = None

        return existed
