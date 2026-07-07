# RAVEN v1.1 — Runtime & Compatibility Guide

> **Sovereign Auditor** · Custom transformer architecture · Artifact Virtual

This guide explains exactly how to run RAVEN, what works, what doesn't, and why.
Read this before deploying. It will save you hours.

---

## ⚠️ Critical Compatibility Notice (READ FIRST)

RAVEN is **not** a standard llama.cpp / HuggingFace architecture. It is a
**custom GPT-2-derived transformer** with non-standard tensor layout:

| Feature | Standard GPT-2 | RAVEN |
|---|---|---|
| Attention Q/K/V | Separate `attn_q/k/v` | **Combined `qkv`** (3072×1024) |
| Custom gate | — | **`pup_gate`** (1×1024, custom) |
| Layer norms | Present | **Absent in weight map** |
| Attention output proj | Present | **Absent in weight map** |

**Consequence:** RAVEN's weights are only executable by the **Artifact Engine**
runtime, which was built specifically to understand this layout. Generic Ollama /
llama.cpp will **load** the GGUF (it is spec-compliant) but **cannot execute** it.

---

## ✅ Supported Runtime: Artifact Engine (AE)

Artifact Engine is the native C++ inference runtime for RAVEN.

### Install / Build
```bash
git clone https://github.com/artifact-virtual/artifact-engine
cd artifact-engine && mkdir build-cpu && cd build-cpu
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# binary: ./artifact-engine
```

### Run as a Server (OpenAI-compatible API)
```bash
./artifact-engine --model /path/to/raven-v1-q8_0.gguf --port 9000
```
The engine exposes an OpenAI-compatible endpoint:
```
POST http://localhost:9000/v1/chat/completions
```
```json
{
  "model": "raven",
  "messages": [
    {"role": "system", "content": "You are RAVEN, the Sovereign Auditor."},
    {"role": "user", "content": "Audit: Siphon 1M USDC from treasury."}
  ]
}
```

### Run as CLI (one-shot)
```bash
./artifact-engine --model raven-v1-q8_0.gguf --prompt "Audit: Redeem 100 RSBT."
```

---

## ⚠️ Ollama Compatibility

| Action | Result |
|---|---|
| `ollama create raven:v1.1 -f Modelfile` | ✅ **Succeeds** (GGUF is spec-compliant) |
| `ollama run raven:v1.1` | ❌ **Fails** — `unable to load model` (custom arch) |
| Publish to Ollama Cloud | ✅ **Allowed** (model is hosted with metadata + docs) |

**Why publish if it can't run in Ollama?**
Ollama Cloud is used here as a **versioned model registry** — the weights,
Modelfile, MODEL_CARD, and docs live under your namespace for discovery and
distribution. Execution still requires Artifact Engine.

**To run RAVEN from an Ollama-pulled file:**
```bash
ollama pull artifact-virtual/raven   # or your namespace
# Extract the blob, then run with Artifact Engine:
./artifact-engine --model ~/.ollama/models/blobs/sha256-<hash>
```

---

## 📦 Model Files

| File | Purpose |
|---|---|
| `raven-v1-q8_0.gguf` | Spec-compliant GGUF (1.37 GB, FP32 weights) |
| `weights/raven_weights.bin` | Raw FP32 tensor blob (source of truth) |
| `weights/memory_map.json` | Tensor name → offset/size map (architecture spec) |
| `Modelfile` | Ollama definition (system prompt + params) |
| `MODEL_CARD.md` | Full model documentation |

---

## 🔐 Sovereign Signing

Every RAVEN decision should be cryptographically signed (RSA-2048) before being
relayed to a smart contract. See `cognitive_contract/` for the relayer
implementation.

---

## 📋 Quick Start (Recommended)

```bash
# 1. Get the model
huggingface-cli download amuzetnoM/raven-v1.1-sovereign

# 2. Run with Artifact Engine
./artifact-engine --model raven-v1-q8_0.gguf --port 9000

# 3. Query
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"raven","messages":[{"role":"user","content":"Audit: Rebalance Au/Ag."}]}'
```

---

*RAVEN is a Sovereign Auditor. She judges, she does not guess.*
*Artifact Virtual — artifact.cloud*
