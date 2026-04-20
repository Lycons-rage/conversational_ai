import numpy as np
import time
from faster_whisper import WhisperModel

class STT:
    def __init__(self):
        print("Loading STT model...")
        self.model = WhisperModel(
            "small",
            device="cuda",
            compute_type="float16",
            download_root="models/stt"
        )

        self.sample_rate = 16000
        self.chunk_size = 1024
        # Buffers
        self.audio_buffer = []
        # Silence detection
        self.silence_counter = 0
        # self.silence_threshold = 8   # ~300–500ms silence
        self.silence_duration_sec = 0.6
        # Sensitivity
        self.silence_energy_threshold = 0.012
        self.speech_streak = 0
        self.min_speech_streak = 3

        # Minimum speech length (~1.2 sec)
        self.min_audio_length = int(self.sample_rate * 2.5)
        self.speech_detected = False
        # turn detection vars
        self.pending_silence = False
        self.pending_silence_time = 0.0
        self.turn_grace_period = 1.5
        self.cooldown_time = 0.5
        self.last_processed_time = 0
        self.turn_locked = False

    def add_audio(self, chunk):
        chunk = np.array(chunk, dtype=np.float32)
        self.audio_buffer.extend(chunk)

        chunk_duration = self.chunk_size / self.sample_rate

        if not self.is_silent(chunk):
            self.speech_streak += 1
            self.silence_counter = 0

            # unlock turn when real speech resumes
            if self.speech_streak >= self.min_speech_streak:
                self.speech_detected = True
                self.turn_locked = False
                self.pending_silence = False
                self.pending_silence_time = 0.0

        else:
            self.speech_streak = 0
            self.silence_counter += 1

            if not self.pending_silence:
                self.pending_silence = True
                self.pending_silence_time = 0.0

            self.pending_silence_time += chunk_duration

    def is_silent(self, chunk):
        energy = np.abs(chunk).mean()
        # slightly relaxed threshold for soft speech
        return energy < (self.silence_energy_threshold * 0.8)

    def should_process(self):
        # prevent rapid duplicate triggering
        if time.time() - self.last_processed_time < self.cooldown_time:
            return False

        # prevent re-processing same silence window
        if self.turn_locked:
            return False

        buffer_ready = len(self.audio_buffer) > self.min_audio_length

        turn_finished = (
            self.pending_silence
            and self.pending_silence_time >= self.turn_grace_period
        )

        return buffer_ready and turn_finished and self.speech_detected
    
    def transcribe(self):
        if not self.audio_buffer:
            return None

        audio_np = np.array(self.audio_buffer, dtype=np.float32)

        segments, _ = self.model.transcribe(
            audio_np,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300
            )
        )

        text = " ".join([seg.text for seg in segments]).strip()

        # lock this turn (prevents duplicate outputs)
        self.turn_locked = True
        self.last_processed_time = time.time()

        # reset state
        self.audio_buffer = []
        self.silence_counter = 0
        self.speech_detected = False
        self.pending_silence = False
        self.pending_silence_time = 0.0
        self.speech_streak = 0

        return text if text else None