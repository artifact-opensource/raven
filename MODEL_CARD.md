# Raven v1.1 — Sovereign AI Model

**Artifact Virtual** · MIT License

---

## What is Raven?

Raven is a 24-layer, 1024-dim transformer (≈1.6B params) built from scratch by
Artifact Virtual. It is a **base language model** — pretrained on raw text, not
instruction-tuned. Think of it as a "raw brain": it has learned language
structure but has not yet been taught to follow commands.

| Property | Value |
|---|---|
| Architecture | Custom Raven (exported as `llama`-compatible) |
| Layers | 24 |
| Embedding dim | 1024 |
| Attention heads | 16 (64-dim each) |
| Context length | 2048 tokens |
| Vocabulary | 259 (256 bytes + 3 specials) |
| Vocab | 259 (256 byte tokens + PAD/BOS/EOS) |
| Feed-forward | 4096 (4× expansion, GELU) |
| Parameters | ~1.6B |
| File size | 1.6 GB (F32) |

---

## Two Versions — Read This

Raven exists in **two forms** with different loaders:

### 1. Ollama version (`artifactvirtual/raven:v1.1-ollama`) ← you are here
- **Architecture tag:** `llama` (re-exported for compatibility)
- **Loader:** Stock **Ollama** / `llama.cpp` — no custom code needed
- **Tokenizer:** Byte-level (256 bytes + specials), registered as `llama` BPE type
- **Custom ops absorbed:** Raven's `pup_gate` (per-layer gating vector) is
  **dropped** from this export — llama.cpp's `llama` arch has no slot for it and
  adding an `ffn_down` bias segfaults the runner (verified). `ffn_gate` is added
  for SwiGLU schema compliance (copied from `ffn_up`, unused by Raven's GELU
  path). **This is a documented fidelity loss** — the native version retains
  `pup_gate`.
- **Known limitation:** Ollama's prefill is fast (~52 tok/s) but **autoregressive
  decode stalls** on this base model (greedy decoding collapses to a degenerate
  loop; the runner hangs on the first decode step). Usable for prompt processing,
  not for full generation in its current export. See `BENCHMARK.md`.
- **Use it:** `ollama run artifactvirtual/raven:v1.1-ollama`

### 2. Native version (`amuzetnoM/raven-v1.1-sovereign` on HuggingFace)
- **Architecture tag:** `raven` (true custom arch)
- **Loader:** **Artifact Engine** (custom runtime) — does NOT run in stock Ollama
- **Full fidelity:** `pup_gate` and all native ops preserved exactly
- **Use it:** Only via the Artifact Virtual inference stack

> **The Ollama version is a faithful re-export.** The weights are identical;
> only the loader-facing structure differs. Both produce the same base-model
> behavior.

---

## How to Use (Ollama)

```bash
# Pull and run
ollama pull artifactvirtual/raven:v1.1-ollama
ollama run artifactvirtual/raven:v1.1-ollama

# Or with the Modelfile
ollama create raven -f Modelfile.ollama
ollama run raven
```

### Python (via ollama SDK)
```python
import ollama
stream = ollama.generate(
    model="artifactvirtual/raven:v1.1-ollama",
    prompt="The future of AI is",
    options={"temperature": 0.8, "num_ctx": 2048},
)
print(stream["response"])
```

---

## Important: This is a Base Model

Raven is **not instruction-tuned**. Out of the box it will:
- Complete text in a statistically-learned way
- Produce repetitive/degenerate output if prompted with a question

This is expected. To get useful chat behavior, fine-tune it (LoRA or full SFT)
on instruction data. The architecture is confirmed loadable and runnable;
quality comes from training.

---

## Loader Notes

| Loader | Works? | Notes |
|---|---|---|
| Ollama (`llama` arch) | ⚠️ Partial | Loads + prefill fast (~52 tok/s); **decode hangs** on base model. `pup_gate` dropped. |
| llama.cpp | ⚠️ Partial | Same GGUF; same decode limitation as Ollama. |
| Artifact Engine | ✅ Yes | `raven-v1-ae.gguf` — full fidelity, retains `pup_gate`. 0.14 tok/s CPU. |
| Raven WASM | ✅ Yes | Dependency-free WASM runtime. 1.65 tok/s CPU. See `wasm` repo. |
| transformers (AutoModel) | ❌ No | Not a HF-standard arch |
| vLLM / TextGen | ⚠️ Partial | GGUF import only, treat as `llama` |

---

## License

MIT — free for commercial and research use. Attribution appreciated.

*Artifact Virtual — building sovereign intelligence.*
