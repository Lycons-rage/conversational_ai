import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 5  # seconds

print("🎤 Recording for 5 seconds... speak now")

audio = sd.rec(int(DURATION * SAMPLE_RATE),
               samplerate=SAMPLE_RATE,
               channels=1,
               dtype='float32')

sd.wait()

print("✅ Recording done")

# simple loudness check
volume = np.linalg.norm(audio)
print(f"🔊 Volume level: {volume:.4f}")

if volume > 0.01:
    print("🎉 Mic is working")
else:
    print("⚠️ Mic seems silent / not detected")