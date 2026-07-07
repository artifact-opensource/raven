# Raven v1.1 — Inference Benchmark (Real Measurements)

**Date:** 2026-07-07
**Hardware:** CPU only (no GPU), single machine
**Model:** Raven v1.1, ~1.6B params, 24 layers, 1024 dim, 16 heads, 4096 FFN, 259-token byte vocab

> **Honesty note:** All numbers below are from actual runs. The previously
> reported "9ms / 67x faster" figure was **never measured** and is retracted.
> These are the real measurements.

## Head-to-Head (same weights, CPU)

| Engine | Artifact / Build | tok/s | sec/token | Status |
|--------|----------------|-------|-----------|--------|
| **Raven WASM** (Wasmtime 46.0.1) | `raven_wasm_runtime.wasm` | **1.66** | 0.60 | ✅ Works (128 tok in 77.2s, stable 1.6–1.7 across seeds) |
| **Artifact Engine** v0.7.0 (build-cpu) | `raven-v1-ae.gguf` | **0.14** | 7.10 | ✅ Works (`--bench`: 32 tok in 228.37s) |
| **Ollama** (llama.cpp) | `raven-v1-ollama.gguf` | prefill 51.7 / **decode hangs** | — | ⚠️ Prefill OK, decode broken |

## How each was measured

### WASM (Wasmtime)
- `extract_weights.py` → `weights.bin` (flat f32 from `raven-v1-ae.gguf`)
- `host/src/main.rs` stages the blob into WASM linear memory and calls
  `bench_forward(seed, n)`. Host measures wall-clock around the call.
- 128 forward passes = 77.2s → **1.66 tok/s**. Repeated: 1.63–1.68.
- Different seeds produce different token sequences → real inference, not mocked.

### Artifact Engine
- `build-cpu/artifact-engine --model weights/raven-v1-ae.gguf --bench`
- Output: `[BENCH] ... generated 32 tokens in 228.37s (0.1 tok/s)`
- → **0.14 tok/s**.

### Ollama
- `ollama create` + `ollama run` / API.
- **Prefill:** prompt_eval 36 tokens in 696ms = **51.7 tok/s** (fast).
- **Decode:** generation stalls on a degenerate `<0x73>` loop and never
  completes — `eval_duration` is never returned. The model is a **base model**
  (not instruction-tuned), so greedy decoding collapses; llama.cpp's scheduler
  for this custom arch also appears to hang on the first decode step.
- **Conclusion:** Ollama is usable for prompt processing but **not** for
  autoregressive generation with this model in its current export.

## Key engineering findings (real bugs found & fixed)

1. **Ollama segfault** — original export added `ffn_down.bias` (folded
   `pup_gate`); llama.cpp's `llama` arch has no `ffn_down.bias` → runner crash
   in `processBatch`. Fixed by dropping `pup_gate` from the Ollama export.
2. **GGUF corruption** — a manual `_inject_readme` hack shifted tensor data
   offsets, corrupting the file. Replaced with the library's `add_string()`.
3. **WASM was a fake** — the committed `.wasm` was 22 bytes of ASCII
   `"BINARY_DATA_SIMULATED"`. Replaced with a real Rust transformer runtime.
4. **AE couldn't load Raven** — Raven's schema (no per-block norms, custom
   `pup_gate`, no attn output proj) didn't match AE's GPT-2 loader. Fixed with
   `export_ae_compat.py` (identity-norm synthesis + schema remap).

## Reproduce

```bash
# AE
cd /home/adam/workspace/projects/artifact-engine
./build-cpu/artifact-engine --model /home/adam/worxpace/gladius/raven/weights/raven-v1-ae.gguf --bench

# WASM
cd /home/adam/worxpace/gladius/wasm
cargo build --release --target wasm32-unknown-unknown
python3 extract_weights.py   # -> weights.bin
/tmp/wasm_host_build/release/raven-wasm-host 128 1 weights.bin
```
