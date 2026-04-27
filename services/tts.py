import numpy as np
import scipy.signal as sps
from piper import PiperVoice

MODEL = None

def load_model(model_path, config_path=None):
    global MODEL

    MODEL = PiperVoice.load(
        model_path=model_path,
        config_path=config_path,
        use_cuda=False  # 🔥 fix CUDA issue
    )

    print("✅ Piper model loaded")


# def text_to_speech(text):
#     global MODEL

#     if MODEL is None:
#         raise ValueError("Model not loaded")

#     audio_chunks = []
#     sample_rate = None

#     # for chunk in MODEL.synthesize(text):
#     #     audio_chunks.append(chunk.audio)
#     #     sample_rate = chunk.sample_rate
#     for chunk in MODEL.synthesize("test"):
#         print(chunk.audio_int16_bytes)
#         print(chunk.sample_rate)
#         print(chunk.sample_channels)
#         print(chunk.sample_width)
#         print(type(chunk))
#         print(dir(chunk))
#         break

#     if not audio_chunks:
#         raise ValueError("No audio generated")
#     audio = np.concatenate(audio_chunks)

#     target_sr = 16000

#     if sample_rate != target_sr:
#         num_samples = int(len(audio) * target_sr / sample_rate)
#         audio = sps.resample(audio, num_samples)

#     audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

#     return audio_int16.tobytes()

def text_to_speech(text):
    global MODEL

    audio_chunks = []
    sample_rate = None

    for chunk in MODEL.synthesize(text):
        # 🔹 convert bytes → int16 numpy
        audio_np = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)

        audio_chunks.append(audio_np)
        sample_rate = chunk.sample_rate

    # 🔹 merge chunks
    audio = np.concatenate(audio_chunks)

    # 🔹 resample
    target_sr = 16000

    if sample_rate != target_sr:
        num_samples = int(len(audio) * target_sr / sample_rate)
        audio = sps.resample(audio.astype(np.float32), num_samples)

    # 🔹 convert back to int16
    audio_int16 = np.clip(audio, -32768, 32767).astype(np.int16)

    return audio_int16.tobytes()

# load_model("/mnt/d/WORK/aios/conversation_with_context/models/tts/lessac/en_US-lessac-medium.onnx", "/mnt/d/WORK/aios/conversation_with_context/models/tts/lessac/en_US-lessac-medium.onnx.json")
# print(text_to_speech("Hello, this is a test of the text-to-speech system."))