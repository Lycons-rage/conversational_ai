import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json
import queue

WS_URL = "ws://localhost:6969/ws"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

audio_input_queue = asyncio.Queue()
audio_output_queue = queue.Queue()


# 🎤 Mic callback → asyncio queue
def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️ Mic:", status)

    chunk = indata.copy().flatten()

    asyncio.run_coroutine_threadsafe(
        audio_input_queue.put(chunk),
        loop
    )


# 🔌 Send audio to server
async def sender(ws):
    while True:
        chunk = await audio_input_queue.get()

        await ws.send(json.dumps({
            "type": "audio_chunk",
            "data": chunk.tolist()
        }))


# 🔌 Receive audio from server
async def receiver(ws):
    while True:
        msg = await ws.recv()
        data = json.loads(msg)

        if data["type"] == "audio_chunk":
            audio = np.array(data["data"], dtype=np.float32)
            audio_output_queue.put(audio)


# 🔊 Continuous speaker stream (FIXED)
def speaker_callback(outdata, frames, time, status):
    if status:
        print("⚠️ Speaker:", status)

    try:
        chunk = audio_output_queue.get_nowait()
    except queue.Empty:
        outdata.fill(0)
        return

    chunk = chunk.reshape(-1, 1)

    if len(chunk) < len(outdata):
        outdata[:len(chunk)] = chunk
        outdata[len(chunk):].fill(0)
    else:
        outdata[:] = chunk[:len(outdata)]


async def speaker():
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=speaker_callback,
        blocksize=CHUNK_SIZE
    )

    with stream:
        print("🔊 Speaker streaming started")
        while True:
            await asyncio.sleep(0.1)


async def main():
    global loop
    loop = asyncio.get_event_loop()

    print("🔌 Connecting to server...")

    async with websockets.connect(WS_URL) as ws:
        print("✅ Connected")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        ):
            print("🎤 Mic streaming started")

            await asyncio.gather(
                sender(ws),
                receiver(ws),
                speaker()
            )


if __name__ == "__main__":
    asyncio.run(main())