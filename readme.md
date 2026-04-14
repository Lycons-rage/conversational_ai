# AI OS – Phase 0.1 Architecture

## Goal
Voice-based conversational AI with:
- Multilingual support (Hindi + English + Hinglish)
- Low latency
- Local-only stack
- Efficient VRAM usage
- Future-ready for context memory + fine-tuning

---

## Core Stack

### STT (Speech-to-Text)
- Model: faster-whisper (small)
- Type: GPU accelerated

**VRAM Usage:**
- Best: ~0.8 GB  
- Average: ~1 GB  
- Worst: ~1.5 GB  

---

### LLM (Cognition Layer)
- Model: Phi-3 Mini (Q4 quantized)
- Type: GPU

**VRAM Usage:**
- Best: ~2.5 GB  
- Average: ~3–3.5 GB  
- Worst: ~4 GB  

---

### TTS (Default – Real-time)
- Model: Piper
- Type: CPU

**VRAM Usage:**
- ~0 GB  

---

### TTS (High Quality – Optional)
- Model: XTTS-v2 (Coqui)
- Type: GPU (or CPU fallback)

**VRAM Usage:**
- Best: ~3 GB  
- Average: ~4 GB  
- Worst: ~5–6 GB  

---

## Total VRAM Consumption

### Without XTTS (Normal Mode)

| Scenario | VRAM |
|--------|------|
| Best | ~3.5 GB |
| Average | ~4–5 GB |
| Worst | ~6 GB |

✔ Safe on 12GB GPU

---

### With XTTS (High-Quality Mode)

| Scenario | VRAM |
|--------|------|
| Best | ~6.5 GB |
| Average | ~7–9 GB |
| Worst | ~10–11+ GB |

Near VRAM limit  
Risk of slowdown or OOM

---

## Pipeline Flow

```
Mic Input
   ↓
STT (faster-whisper)
   ↓
Text + Conversation History
   ↓
LLM (Phi-3 Mini)
   ↓
Response Text
   ↓
TTS (Piper / XTTS)
   ↓
Audio Output
```
---

## Known Risks

### 1. Context Overflow
- Too much history → latency + degraded responses  
- Fix later: pruning / summarization  

---

### 2. Pipeline Latency
- Sequential execution → delay  
- Future fix:
  - streaming STT  
  - early LLM triggering  

---

### 3. VRAM Contention
- XTTS + LLM + Whisper together → risk of OOM  

---

## Recommended Strategy

### Default Mode
- Whisper + Phi-3 + Piper  
- Fast, stable, low VRAM  

### High-Quality Mode
- Enable XTTS only when needed  
- Options:
  - unload LLM  
  - or run XTTS on CPU  

---

## Final Summary

- Efficient baseline: ✅  
- Scalable architecture: ✅  
- Hinglish-ready (future tuning): ✅  
- VRAM-safe (if disciplined): ⚠️  

---

## Principle

> Do not run everything at once.  
> Load only what is needed.  
> Optimize flow before upgrading models.