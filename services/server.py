from fastapi import FastAPI, WebSocket
import asyncio
import json
import re

from services.stt import STT
from services.llm import LLM
from services.tts import text_to_speech, load_model

app = FastAPI()

audio_queue = asyncio.Queue()
output_queue = asyncio.Queue()

stt = STT()
llm = LLM()
load_model("/mnt/d/WORK/aios/conversation_with_context/models/tts/ryan/model.onnx", "/mnt/d/WORK/aios/conversation_with_context/models/tts/ryan/config.json")

transcript_buffer = ""
llm_lock = asyncio.Lock()  # to prevent concurrent LLM calls

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
        if msg["type"] == "audio_chunk":
            await ws.send_bytes(msg["data"])
            print("audio sent")
        else:
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
                # Send transcript update
                await output_queue.put({
                    "type": "text",
                    "data": text
                })
                # Call LLM (non-blocking wrapper)
                async def safe_run_llm(prompt):
                    async with llm_lock:
                        await run_llm(prompt)

                asyncio.create_task(safe_run_llm(transcript_buffer))
                transcript_buffer = ""  # reset buffer after sending to LLM

def split_sentences(buffer):
    sentences = re.split(r'(?<=[.!?]) +', buffer)
    return sentences

# LLM streaming
async def run_llm(prompt):
    loop = asyncio.get_event_loop()
    buffer = ""

    def blocking_llm():
        nonlocal buffer

        for token in llm.stream(prompt):
            buffer += token

            asyncio.run_coroutine_threadsafe(
                output_queue.put({
                    "type": "llm_token",
                    "data": token
                }),
                loop
            )
            if any(p in buffer for p in [".", "?", "!"]):
                sentences = split_sentences(buffer)
                buffer = sentences[-1]

                for sentence in sentences[:-1]:
                    asyncio.run_coroutine_threadsafe(
                        handle_tts(sentence.strip()),
                        loop
                    )

        # CRITICAL FIX: flush remaining buffer
        if buffer.strip():
            asyncio.run_coroutine_threadsafe(
                handle_tts(buffer.strip()),
                loop
            )
    await asyncio.to_thread(blocking_llm)


# TTS handling
async def handle_tts(sentence):
    print("Sent into TTS:", sentence)
    audio_bytes = text_to_speech(sentence)
    await output_queue.put({
        "type": "audio_chunk",
        "data": audio_bytes
    })

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting pipeline worker")
    asyncio.create_task(pipeline_worker())

@app.on_event("shutdown")
async def shutdown_event():
    print("Server shutting down cleanly...")