import numpy as np
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

        self.audio_buffer = []
        self.sample_rate = 16000
        self.chunk_limit = 80  # number of chunks before processing

    def add_audio(self, chunk):
        """Add incoming audio chunk"""
        self.audio_buffer.extend(chunk)

    def should_process(self):
        return len(self.audio_buffer) > self.chunk_limit * 1024

    def transcribe(self):
        """Run STT on buffered audio"""
        if not self.audio_buffer:
            return None

        audio_np = np.array(self.audio_buffer, dtype=np.float32)

        segments, _ = self.model.transcribe(
            audio_np,
            language="en",
            beam_size=1
        )

        text = " ".join([seg.text for seg in segments]).strip()

        # clear buffer after processing
        self.audio_buffer = []
        return text if text else None