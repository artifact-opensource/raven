# Raven v1.1 — Runtime Guide

How to run Raven v1.1 in each supported inference backend. All backends consume
the **same trained weights**; only the serialization format differs.

> **Reminder:** Raven is a *base model*. Greedy decoding produces degenerate
> output until the model is fine-tuned. The runtimes below execute correctly —
> the quality issue is the model, not the engines.

---

## 1. Artifact Engine (full fidelity)

Uses `weights/raven-v1-ae.gguf`, which preserves Raven's full architecture
including `pup_gate`.

**Build the export:**
```bash
python3 src/engine/export_ae_compat.py --out weights/raven-v1-ae.gguf
```

**Run (benchmark):**
```bash
cd /path/to/artifact-engine
./build-cpu/artifact-engine --model /path/to/raven/weights/raven-v1-ae.gguf --bench
```
Expected: ~0.14 tok/s on CPU (32 tokens in ~228s).

**Run (interactive / generation):**
```bash
./build-cpu/artifact-engine --model /path/to/raven/weights/raven-v1-ae.gguf --chat
```

The AE export remaps Raven's native schema (combined `qkv`, custom `pup_gate`, no
per-block norms) into AE's GPT-2 loader by synthesizing identity norms and an
attention-output projection, and preserving `pup_gate` as an extra per-layer
tensor the engine applies post-attention.

---

## 2. Ollama / llama.cpp

Uses `weights/raven-v1-ollama.gguf`, re-exported under the `llama` architecture tag.

**Build the export:**
```bash
python3 src/engine/build_ollama_compat.py --out weights/raven-v1-ollama.gguf
```

**Register and run:**
```bash
ollama create artifactvirtual/raven:v1.1-ollama -f Modelfile
ollama run artifactvirtual/raven:v1.1-ollama
```

**Known limitations (documented, not bugs in your setup):**
- `pup_gate` is **dropped** from this export. llama.cpp's `llama` arch has no slot
  for it; adding an `ffn_down` bias (an earlier attempt to fold it) segfaults the
  runner. This is a fidelity loss vs. the AE/WASM exports.
- **Prefill works** (~52 tok/s) but **autoregressive decode hangs** on this base
  model — greedy decoding collapses to a degenerate token loop and the runner
  never returns the first generated token. Ollama is therefore usable for prompt
  processing / embeddings-style use, not full text generation, in this export.

---

## 3. WebAssembly

See the [`wasm` repo](https://github.com/artifact-opensource/wasm) for the full
runtime. Summary:

```bash
# In the wasm repo:
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
cargo build --release --manifest-path host/Cargo.toml

python3 extract_weights.py --gguf /path/to/raven/weights/raven-v1-ae.gguf --out weights/weights.bin
./target/release/raven-wasm-host 128 1 weights/weights.bin
```
Expected: ~1.65 tok/s on CPU.

---

## Weights file reference

| File | Backend | `pup_gate` | Notes |
|---|---|---|---|
| `raven-v1-ae.gguf` | Artifact Engine | ✅ preserved | Full fidelity |
| `raven-v1-ollama.gguf` | Ollama | ❌ dropped | `llama` arch, decode hangs on base model |
| `weights.bin` | WASM | ✅ preserved | Flat `f32` blob extracted from either GGUF |

All weight artifacts are gitignored — regenerate from source with the scripts above.
