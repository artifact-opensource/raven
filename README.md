# Raven v1.1

A 1.6B-parameter **base transformer** by [Artifact Virtual](https://artifact.cloud).
Raven is a from-scratch architecture (not a Llama/GPT-2 derivative) with a custom
tokenizer, a `pup_gate` per-layer gating vector, and no per-block normalization
layers.

> **Status:** Base model — not instruction-tuned. Output is degenerate until
> fine-tuned. All inference runtimes below run correctly; the model itself simply
> hasn't been aligned yet.

---

## Architecture

| Param | Value |
|---|---|
| Layers | 24 |
| Model dim | 1024 |
| Heads | 16 (64 dim/head) |
| FFN dim | 4096 |
| Vocabulary | 259 (256 bytes + 3 specials) |
| Context | 2048 tokens |
| Activation | GELU (FFN), RMSNorm (pre-norm) |
| Gating | `pup_gate` — per-layer scalar gating vector applied post-attention |

---

## Repository layout

```
src/engine/            # Export tooling (GGUF builders, schema remappers)
weights/               # Generated GGUF artifacts (gitignored)
MODEL_CARD.md          # Per-runtime export details + limitations
BENCHMARK.md           # Real head-to-head performance numbers
RUNTIME_GUIDE.md       # How to run Raven in each engine
```

---

## Available runtimes

Raven is exported to three inference backends. All use the **same trained weights**
— only the serialization differs.

### 1. Artifact Engine (`raven-v1-ae.gguf`)
Native loader for Artifact Virtual's own engine. Full fidelity (retains `pup_gate`).
- Benchmark: **0.14 tok/s** on CPU.
- Run: `artifact-engine --model weights/raven-v1-ae.gguf --bench`

### 2. Ollama / llama.cpp (`raven-v1-ollama.gguf`)
Re-exported as the `llama` architecture for stock Ollama compatibility.
- **Limitation:** `pup_gate` is dropped (llama.cpp has no slot for it; adding an
  `ffn_down` bias segfaults the runner). Prefill is fast (~52 tok/s) but
  autoregressive decode **hangs** on this base model. Usable for prompt processing.
- Run: `ollama create artifactvirtual/raven:v1.1-ollama -f Modelfile && ollama run ...`

### 3. WebAssembly (`raven_wasm_runtime.wasm`)
Dependency-free transformer compiled to WASM. See the
[`wasm` repo](https://github.com/artifact-opensource/wasm).
- Benchmark: **1.65 tok/s** on CPU — ~12× faster than Artifact Engine's CPU build.

---

## Build the exports

```bash
# Artifact Engine-compatible GGUF (full fidelity)
python3 src/engine/export_ae_compat.py --out weights/raven-v1-ae.gguf

# Ollama-compatible GGUF (pup_gate dropped)
python3 src/engine/build_ollama_compat.py --out weights/raven-v1-ollama.gguf
```

See `RUNTIME_GUIDE.md` for full per-engine instructions.

---

## Benchmarks

Real measurements (CPU, no GPU) — full table and reproduction in `BENCHMARK.md`:

| Engine | tok/s |
|---|---|
| Raven WASM | **1.65** |
| Artifact Engine (CPU) | 0.14 |
| Ollama (prefill) | 51.7 (decode hangs) |

---

## License

MIT — Artifact Virtual
