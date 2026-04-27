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


# Mic callback → asyncio queue
def audio_callback(indata, frames, time, status):
    if status:
        print("Mic:", status)

    chunk = indata.copy().flatten()
    asyncio.run_coroutine_threadsafe(
        audio_input_queue.put(chunk),
        loop
    )


# Send audio to server
async def sender(ws):
    while True:
        chunk = await audio_input_queue.get()
        await ws.send(json.dumps({
            "type": "audio_chunk",
            "data": chunk.tolist()
        }))


# Receive audio from server
async def receiver(ws):
    while True:
        msg = await ws.recv()

        # 🔹 AUDIO
        if isinstance(msg, bytes):
            audio = np.frombuffer(msg, dtype=np.int16)
            audio = audio.astype(np.float32) / 32767.0

            # 🔥 IMPORTANT: accumulate full message first
            # (ensures proper chunk alignment)
            for i in range(0, len(audio), CHUNK_SIZE):
                chunk = audio[i:i + CHUNK_SIZE]

                # pad last chunk to maintain size consistency
                if len(chunk) < CHUNK_SIZE:
                    padded = np.zeros(CHUNK_SIZE, dtype=np.float32)
                    padded[:len(chunk)] = chunk
                    chunk = padded

                audio_output_queue.put(chunk)

            continue  # keep this OUTSIDE loop

        # 🔹 TEXT
        if isinstance(msg, str):
            msg = msg.strip()
            if not msg:
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                print("\nSkipping invalid JSON:", msg[:50])
                continue

            if data["type"] == "text":
                print("\nSTT:", data["data"])

            elif data["type"] == "llm_token":
                print(data["data"], end="", flush=True)


# Continuous speaker stream (FIXED)
def speaker_callback(outdata, frames, time, status):
    if status:
        print("Speaker:", status)

    output = np.zeros(frames, dtype=np.float32)
    filled = 0

    while filled < frames:
        try:
            chunk = audio_output_queue.get_nowait()
        except queue.Empty:
            break

        remaining = frames - filled
        take = min(len(chunk), remaining)

        output[filled:filled+take] = chunk[:take]

        # 🔥 put back leftover (VERY IMPORTANT)
        if take < len(chunk):
            audio_output_queue.put(chunk[take:])

        filled += take

    outdata[:] = output.reshape(-1, 1)


async def speaker():
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=speaker_callback,
        blocksize=CHUNK_SIZE
    )

    with stream:
        print("Speaker streaming started")
        # wait until buffer has some data
        while audio_output_queue.qsize() < 5:
            await asyncio.sleep(0.01)
        while True:
            await asyncio.sleep(0.1)


async def main():
    global loop
    loop = asyncio.get_event_loop()
    print("Connecting to server...")

    async with websockets.connect(WS_URL) as ws:
        print("Connected")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        ):
            print("Mic streaming started")

            await asyncio.gather(
                sender(ws),
                receiver(ws),
                speaker()
            )


if __name__ == "__main__":
    asyncio.run(main())