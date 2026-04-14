from fastapi import FastAPI, WebSocket
import asyncio
import json

from services.stt import STT

app = FastAPI()

# Buffers
audio_queue = asyncio.Queue()
output_queue = asyncio.Queue()

# STT instance
stt = STT()


@app.get("/health")
def health():
    return {"status": "ok"}


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🔌 Client connected")

    consumer_task = asyncio.create_task(consumer(ws))
    producer_task = asyncio.create_task(producer(ws))

    await asyncio.gather(consumer_task, producer_task)


# Receive audio from client
async def consumer(ws: WebSocket):
    while True:
        msg = await ws.receive_text()
        data = json.loads(msg)

        if data["type"] == "audio_chunk":
            await audio_queue.put(data["data"])


# Send output to client
async def producer(ws: WebSocket):
    while True:
        msg = await output_queue.get()
        await ws.send_text(json.dumps(msg))


# STT processing worker
async def stt_worker():
    while True:
        chunk = await audio_queue.get()

        stt.add_audio(chunk)

        if stt.should_process():
            text = stt.transcribe()

            if text:
                print("STT:", text)

                await output_queue.put({
                    "type": "text",
                    "data": text
                })


# Startup
@app.on_event("startup")
async def startup_event():
    print("Starting STT worker")
    asyncio.create_task(stt_worker())

@app.on_event("shutdown")
async def shutdown_event():
    print("Server shutting down cleanly...")