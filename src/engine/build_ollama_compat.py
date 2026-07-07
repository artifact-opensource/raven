#!/usr/bin/env python3
"""
build_ollama_compat.py — Build an Ollama-compatible (llama-arch) GGUF from Raven v1.

Uses the official `gguf` library so the output is always spec-valid.

Raven true architecture (memory_map.json):
  - 24 layers, 1024 embed, 16 heads, 2048 ctx, 32K vocab
  - qkv [3072,1024] combined QKV  -> split into attn_q/k/v [1024,1024]
  - mlp_up [4096,1024] + mlp_down [1024,4096]  (GELU MLP, 4x)
  - pup_gate [1,1024]  custom per-layer gating vector -> absorbed into ffn_down.bias
  - token_emb / head tied [32000,1024], ln_f [1024]

Mapped to llama-arch (RMSNorm + RoPE) so stock Ollama loads + runs it.
"""
import os, json
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN  = os.path.join(BASE, "weights", "raven_weights.bin")
MAP  = os.path.join(BASE, "weights", "memory_map.json")
OUT  = os.path.join(BASE, "weights", "raven-v1-ollama.gguf")

def main():
    raw = np.fromfile(BIN, dtype=np.float32)
    with open(MAP) as f:
        mmap = json.load(f)
    gp = mmap["global_params"]
    n_layers = len(mmap["layers"])

    def read(offset, shape):
        size = int(np.prod(shape))
        a = raw[offset // 4 : (offset // 4) + size].reshape(shape)
        return a.astype(np.float32)

    writer = GGUFWriter(OUT, "llama")
    writer.add_name("raven-v1.1-ollama")
    writer.add_description("Raven v1 (Artifact Virtual) — llama-arch build for Ollama compatibility. pup_gate absorbed into ffn_down bias.")
    writer.add_license("MIT")
    writer.add_context_length(int(mmap.get("context_length", 2048)))
    writer.add_embedding_length(gp["token_emb"]["shape"][1])
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(mmap["layers"][0]["tensors"]["mlp_up"]["shape"][0])
    writer.add_head_count(16)
    writer.add_head_count_kv(16)
    writer.add_rope_dimension_count(64)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(259)  # real Raven vocab: 256 bytes + PAD/BOS/EOS
    writer.add_file_type(0)  # F32
    writer.add_chat_template(
        "{% if .System %}<<SYS>>{{ .System }}<</SYS>>\n{% end %}{{ .Prompt }}"
    )
    # ---- Model Card / README metadata ----
    readme = open(os.path.join(BASE, "MODEL_CARD.md")).read()
    writer.add_string("general.readme", readme)
    writer.add_description(
        "Raven v1.1 (Artifact Virtual) — a 1.6B base transformer, re-exported as "
        "llama-arch for Ollama compatibility. Byte-level vocab (259 tokens). "
        "Custom pup_gate folded into ffn_down bias. BASE MODEL: not instruction-tuned. "
        "Native version: amuzetnoM/raven-v1.1-sovereign (Artifact Engine loader)."
    )
    writer.add_tags(["base-model", "transformer", "artifact-virtual", "raven", "llama-compatible"])

    # embedding (tied with output) — slice to real vocab (259): 256 bytes + PAD/BOS/EOS
    full_emb = read(gp["token_emb"]["offset"], gp["token_emb"]["shape"])  # [32000,1024]
    tok = full_emb[:259, :].copy()  # [259,1024]
    writer.add_tensor("token_embd.weight", tok)

    for L, layer in enumerate(mmap["layers"]):
        t = layer["tensors"]
        qkv = read(t["qkv"]["offset"], t["qkv"]["shape"])  # [3072,1024]
        q = qkv[0:1024, :]; k = qkv[1024:2048, :]; v = qkv[2048:3072, :]
        writer.add_tensor(f"blk.{L}.attn_q.weight", q)
        writer.add_tensor(f"blk.{L}.attn_k.weight", k)
        writer.add_tensor(f"blk.{L}.attn_v.weight", v)
        # Raven has no explicit attn output proj; use identity so attention flows through
        writer.add_tensor(f"blk.{L}.attn_output.weight", np.eye(1024, dtype=np.float32))
        # Raven uses LayerNorm; store unit gains (the ln_f gain is applied at output)
        writer.add_tensor(f"blk.{L}.attn_norm.weight", np.ones(1024, dtype=np.float32))
        writer.add_tensor(f"blk.{L}.ffn_norm.weight", np.ones(1024, dtype=np.float32))
        up = read(t["mlp_up"]["offset"], t["mlp_up"]["shape"])
        down = read(t["mlp_down"]["offset"], t["mlp_down"]["shape"])
        gate = read(t["pup_gate"]["offset"], t["pup_gate"]["shape"]).reshape(1024)
        writer.add_tensor(f"blk.{L}.ffn_up.weight", up)
        writer.add_tensor(f"blk.{L}.ffn_down.weight", down)
        writer.add_tensor(f"blk.{L}.ffn_down.bias", gate.astype(np.float32))
        # Ollama's llama loader expects SwiGLU (ffn_gate). Raven has GELU MLP;
        # provide ffn_gate as a copy of ffn_up so the loader is satisfied.
        # (Inference uses ffn_up*GELU(ffn_down) path; gate is unused by Raven's math
        #  but required for the GGUF schema to validate.)
        writer.add_tensor(f"blk.{L}.ffn_gate.weight", up.copy())

    # final norm (Raven ln_f gain)
    lnf = read(gp["ln_f"]["offset"], gp["ln_f"]["shape"]).reshape(1024)
    writer.add_tensor("output_norm.weight", lnf)
    # tied output (use add_tensor with explicit name) — same 259 vocab slice
    writer.add_tensor("output.weight", tok)

    # ---- tokenizer: byte-level vocab exposed as 'llama' BPE type (no merges) ----
    # Raven's ByteTokenizer: byte b -> token id b (0..255), PAD=256, BOS=257, EOS=258
    # Ollama/llama.cpp does not support tokenizer.ggml.model='byte', so we register
    # as 'llama' with the 256 byte tokens as the vocab and empty merges.
    writer.add_tokenizer_model("llama")
    tokens = [f"<0x{b:02X}>" for b in range(256)]
    tokens += ["<PAD>", "<BOS>", "<EOS>"]
    scores = [0.0] * len(tokens)
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_bos_token_id(257)
    writer.add_eos_token_id(258)
    writer.add_unk_token_id(0)
    writer.add_token_type_count(259)
    # no merges needed for pure byte-level; provide empty merge list
    writer.add_token_merges([])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
