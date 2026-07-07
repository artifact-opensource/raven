#!/usr/bin/env python3
"""
export_ae_compat.py — Re-export Raven as a GPT-2-named GGUF Artifact Engine v0.7.0
can load.

AE's GPT-2 loader REQUIRES per block:
    blk.N.attn_norm.weight   (RMSNorm, [embd])
    blk.N.ffn_norm.weight    (RMSNorm, [embd])
    blk.N.attn_q/k/v.weight  (each [embd, embd])
    blk.N.attn_output.weight ([embd, embd])
    blk.N.ffn_gate.weight    ([ffn, embd])
    blk.N.ffn_up.weight      ([ffn, embd])
    blk.N.ffn_down.weight    ([embd, ffn])
plus global: token_embd.weight [vocab, embd], output_norm.weight [embd], output.weight [vocab, embd]

Raven's native schema (147 tensors) ACTUALLY is:
    token_embd.weight [32000, 1024]   (vocab padded to 32000)
    output_norm.weight [1024]
    output.weight [32000, 1024]
    per block: attn_q/k/v.weight [1024,1024], ffn_up.weight [4096,1024],
               ffn_down.weight [1024,4096], pup_gate.weight [1024]   <-- 1-D gate
    NO per-block attn_norm/ffn_norm, NO attn output proj.

Remapping (documented, APPROXIMATE vs native Raven):
    1. Synthesize identity RMSNorm weights (all ones, [embd]) for attn_norm/ffn_norm.
       RMSNorm with w=1 is near-identity (variance normalize only). Approximation.
    2. attn_q/k/v map directly (shape [embd,embd] matches AE).
    3. Synthesize attn_output.weight as identity [embd,embd] (Raven has none).
    4. ffn_gate = ffn_up.T  ([4096,1024] <- Raven ffn_up [4096,1024])  -> already [ffn,embd] OK
       ffn_up   = zeros [ffn,embd]  (Raven has no second FFN proj; zero = pass-through gating)
       ffn_down = ffn_down.T ([embd,4096] <- Raven [1024,4096])
    5. pup_gate [1024] folded as ffn_down.bias [4096]? No — pup_gate is [embd], not [ffn].
       Instead add as attn_norm.bias? AE has no norm bias. Store pup_gate as an extra
       tensor blk.N.pup_gate.weight [1024] (AE ignores unknown tensors) so it's preserved.
    6. token_embd/output: keep padded 32000 rows (AE reads vocab_size from KV; we set
       vocab_size=259 so AE uses first 259 rows).

The native version (raven-v1.1-sovereign) remains the source of truth. This export
is for AE compatibility and benchmarking only.
"""
import os
import numpy as np
from gguf import GGUFReader, GGUFWriter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(BASE, "weights", "raven-v1-q8_0.gguf")
OUT = os.path.join(BASE, "weights", "raven-v1-ae.gguf")

assert os.path.exists(SRC), f"missing {SRC}"
reader = GGUFReader(SRC)
T = {t.name: t for t in reader.tensors}
def g(n): return np.asarray(T[n].data)

def kvi(key, d=None):
    try: return int(reader.get_value(key))
    except Exception: return d

n_layers = kvi("block_count", 24)
n_embd   = kvi("embedding_length", 1024)
n_head   = kvi("attention.head_count", 16)
n_ffn    = 4096          # measured from file
n_vocab  = kvi("vocab_size", 259)
ctx      = kvi("context_length", 2048)

print(f"layers={n_layers} embd={n_embd} heads={n_head} ffn={n_ffn} vocab={n_vocab} ctx={ctx}")

writer = GGUFWriter(OUT, "gpt2")
writer.add_string("general.architecture", "gpt2")
writer.add_string("general.name", "raven-v1-ae")
# AE's C gguf lib (gguf_extract_arch) reads ARCH-PREFIXED keys: gpt2.block_count etc.
# GGUFWriter.add_* does NOT auto-prefix, so we write the prefixed names explicitly.
writer.add_uint32("gpt2.block_count", n_layers)
writer.add_uint32("gpt2.embedding_length", n_embd)
writer.add_uint32("gpt2.attention.head_count", n_head)
writer.add_uint32("gpt2.feed_forward_length", n_ffn)
writer.add_uint32("gpt2.vocab_size", n_vocab)
writer.add_uint32("gpt2.context_length", ctx)
writer.add_string("general.description",
    "Raven v1.1 re-exported for Artifact Engine (GPT-2 schema). Norms synthesized "
    "as identity; attn_output synthesized as identity; pup_gate preserved as extra "
    "tensor. APPROXIMATE vs native. Source of truth: raven-v1.1-sovereign.")
writer.add_string("general.license", "MIT")
writer.add_string("general.tags", "base-model,transformer,artifact-virtual,raven,artifact-engine")
writer.add_string("general.author", "Artifact Virtual")
writer.add_string("general.basename", "raven-v1-ae")

# global
writer.add_tensor("token_embd.weight", g("token_embd.weight"))
writer.add_tensor("output_norm.weight", g("output_norm.weight"))
writer.add_tensor("output.weight", g("output.weight"))

identity_norm = np.ones((n_embd,), dtype=np.float32)
identity_out  = np.eye(n_embd, dtype=np.float32)      # [embd, embd]
zero_ffn_up   = np.zeros((n_ffn, n_embd), dtype=np.float32)

for i in range(n_layers):
    p = f"blk.{i}."
    writer.add_tensor(p + "attn_norm.weight", identity_norm.copy())
    writer.add_tensor(p + "ffn_norm.weight",  identity_norm.copy())
    writer.add_tensor(p + "attn_q.weight", g(p + "attn_q.weight").astype(np.float32))
    writer.add_tensor(p + "attn_k.weight", g(p + "attn_k.weight").astype(np.float32))
    writer.add_tensor(p + "attn_v.weight", g(p + "attn_v.weight").astype(np.float32))
    writer.add_tensor(p + "attn_output.weight", identity_out.copy())
    # ffn_gate = ffn_up (Raven [4096,1024] == AE [ffn,embd]) OK
    writer.add_tensor(p + "ffn_gate.weight", g(p + "ffn_up.weight").astype(np.float32))
    writer.add_tensor(p + "ffn_up.weight",   zero_ffn_up.copy())
    # ffn_down: Raven [1024,4096] -> AE [embd,ffn] = transpose
    fd = g(p + "ffn_down.weight").astype(np.float32)   # [1024,4096]
    writer.add_tensor(p + "ffn_down.weight", fd.T)     # [4096,1024] == [embd,ffn]
    # preserve pup_gate as extra (AE ignores unknown)
    writer.add_tensor(p + "pup_gate.weight", g(p + "pup_gate.weight").astype(np.float32))

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"Wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
