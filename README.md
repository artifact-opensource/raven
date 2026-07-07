# Raven v1.1

A sovereign 1.6B-parameter base transformer by **Artifact Virtual**.

## Two distributions

| | Ollama (this repo) | HuggingFace (native) |
|---|---|---|
| Registry | `artifactvirtual/raven:v1.1-ollama` | `amuzetnoM/raven-v1.1-sovereign` |
| Arch tag | `llama` (re-exported) | `raven` (native) |
| Loader | Stock Ollama / llama.cpp | Artifact Engine |
| Custom ops | `pup_gate` folded into bias | Fully preserved |
| Fine for | Quick local inference | Full-fidelity research |

## Quick start (Ollama)

```bash
ollama pull artifactvirtual/raven:v1.1-ollama
ollama run artifactvirtual/raven:v1.1-ollama
```

## Build the Ollama-compatible GGUF yourself

```bash
pip install gguf --break-system-packages
python3 src/engine/build_ollama_compat.py
# -> weights/raven-v1-ollama.gguf
ollama create raven -f Modelfile.ollama
ollama run raven
```

## Architecture

24 layers · 1024 dim · 16 heads (64-dim) · 2048 ctx · 259 vocab (byte-level) ·
4096 FFN (GELU) · ~1.6B params · F32.

## Note

Raven is a **base model** — not instruction-tuned. Output is degenerate until
fine-tuned. See `MODEL_CARD.md` for full details.

## License

MIT
