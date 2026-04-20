from fastapi import FastAPI, WebSocket
import asyncio
import json

from services.stt import STT
from services.llm import LLM

app = FastAPI()

audio_queue = asyncio.Queue()
output_queue = asyncio.Queue()

stt = STT()
llm = LLM()

transcript_buffer = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🔌 Client connected")

    consumer_task = asyncio.create_task(consumer(ws))
    producer_task = asyncio.create_task(producer(ws))

    await asyncio.gather(consumer_task, producer_task)


async def consumer(ws: WebSocket):
    while True:
        msg = await ws.receive_text()
        data = json.loads(msg)

        if data["type"] == "audio_chunk":
            await audio_queue.put(data["data"])


async def producer(ws: WebSocket):
    while True:
        msg = await output_queue.get()
        await ws.send_text(json.dumps(msg))


# 🧠 STT + LLM worker
async def pipeline_worker():
    global transcript_buffer

    while True:
        chunk = await audio_queue.get()

        stt.add_audio(chunk)

        if stt.should_process():
            text = stt.transcribe()

            if text:
                print("📝 STT:", text)

                transcript_buffer += " " + text

                # 🔥 Send transcript update
                await output_queue.put({
                    "type": "text",
                    "data": text
                })

                # 🔥 Call LLM (non-blocking wrapper)
                asyncio.create_task(run_llm(transcript_buffer))


# 🧠 LLM streaming
async def run_llm(prompt):
    loop = asyncio.get_event_loop()

    def blocking_llm():
        for token in llm.stream(prompt):
            asyncio.run_coroutine_threadsafe(
                output_queue.put({
                    "type": "llm_token",
                    "data": token
                }),
                loop
            )

    await asyncio.to_thread(blocking_llm)


@app.on_event("startup")
async def startup_event():
    print("🚀 Starting pipeline worker")
    asyncio.create_task(pipeline_worker())

@app.on_event("shutdown")
async def shutdown_event():
    print("Server shutting down cleanly...")