"""
Spec-compliant GGUF v2 writer for Raven-v1.

Reads the raw FP32 blob (raven_weights.bin) using weights/memory_map.json
and emits a GGUF file that conforms to the llama.cpp GGUF spec:
  - n_tensors, n_kv are uint64
  - tensor name lengths and dimension counts are uint64
  - tensor dims are uint64
  - tensor type is uint32, offset is uint64

NOTE: Raven is a CUSTOM architecture (combined qkv, custom pup_gate, no
layer-norms / no attn_output in the map). This GGUF is spec-compliant so it
parses and can be hosted, but only Artifact Engine can execute it correctly.
"""
import json
import struct
import os
import numpy as np

BASE = "/home/adam/worxpace/gladius/raven/weights"
BIN = os.path.join(BASE, "raven_weights.bin")
MAP = os.path.join(BASE, "memory_map.json")
OUT = os.path.join(BASE, "raven-v1-q8_0.gguf")  # overwrite the broken one

F32 = 0  # GGUF tensor type for float32

def u64(v): return struct.pack('<Q', v)
def u32(v): return struct.pack('<I', v)


def main():
    with open(MAP) as f:
        mm = json.load(f)
    with open(BIN, 'rb') as f:
        blob = f.read()

    # Build the ordered list of (gguf_name, shape, offset, size_bytes)
    tensors = []  # (name, [dims], offset, nbytes)

    # globals
    gp = mm['global_params']
    tensors.append(("token_embd.weight", gp['token_emb']['shape'],
                    gp['token_emb']['offset'], gp['token_emb']['size']))
    tensors.append(("output_norm.weight", gp['ln_f']['shape'],
                    gp['ln_f']['offset'], gp['ln_f']['size']))
    tensors.append(("output.weight", gp['head']['shape'],
                    gp['head']['offset'], gp['head']['size']))

    for layer in mm['layers']:
        i = layer['layer_index']
        t = layer['tensors']
        # qkv [3072,1024] -> split into attn_q/k/v [1024,1024]
        qkv_off = t['qkv']['offset']
        qkv_size = t['qkv']['size']
        per = qkv_size // 3
        for name, off in [("attn_q.weight", qkv_off),
                          ("attn_k.weight", qkv_off + per),
                          ("attn_v.weight", qkv_off + 2 * per)]:
            tensors.append((f"blk.{i}.{name}", [1024, 1024], off, per))
        tensors.append((f"blk.{i}.ffn_up.weight", t['mlp_up']['shape'],
                        t['mlp_up']['offset'], t['mlp_up']['size']))
        tensors.append((f"blk.{i}.ffn_down.weight", t['mlp_down']['shape'],
                        t['mlp_down']['offset'], t['mlp_down']['size']))
        # custom gate: drop the leading 1 dim -> [1024]
        tensors.append((f"blk.{i}.pup_gate.weight", [1024],
                        t['pup_gate']['offset'], t['pup_gate']['size']))

    n_tensors = len(tensors)

    # ---- Metadata KV ----
    kv = []
    def add_str(k, v):
        kv.append((k.encode(), b'gpt2' if False else None, v.encode()))
    # GGUF value types: 8=string, 4=uint32, 5=int32, 6=float32, 7=bool
    meta = []
    def put_str(k, v):
        meta.append((k, 8, v.encode()))
    def put_u32(k, v):
        meta.append((k, 4, struct.pack('<I', v)))
    def put_f32(k, v):
        meta.append((k, 6, struct.pack('<f', v)))

    put_str("general.architecture", "gpt2")
    put_str("general.name", "raven-v1")
    put_str("general.file_type", "0")  # 0 = F32 all
    put_u32("gpt2.block_count", 24)
    put_u32("gpt2.embedding_length", 1024)
    put_u32("gpt2.context_length", 2048)
    put_u32("gpt2.feed_forward_length", 4096)
    put_u32("gpt2.attention.head_count", 16)
    put_u32("gpt2.attention.head_count_kv", 16)
    put_f32("gpt2.attention.layer_norm_epsilon", 1e-5)

    n_kv = len(meta)

    # ---- Header ----
    out = bytearray()
    out += b'GGUF'
    out += u32(2)          # version
    out += u64(n_tensors)
    out += u64(n_kv)

    # metadata
    for k, vtype, vbytes in meta:
        out += u64(len(k))
        out += k.encode()
        out += u32(vtype)
        if vtype == 8:  # string: uint64 len + bytes
            out += u64(len(vbytes))
            out += vbytes
        else:
            out += vbytes

    # tensor info table
    for name, dims, offset, nbytes in tensors:
        out += u64(len(name))
        out += name.encode()
        out += u32(len(dims))      # n_dims (uint32 per spec)
        for d in dims:
            out += u64(d)
        out += u32(F32)            # tensor type
        out += u64(offset)         # offset (relative to data start)

    # pad header to multiple of 32 bytes
    while len(out) % 32 != 0:
        out += b'\x00'

    # append raw tensor data
    out += blob

    with open(OUT, 'wb') as f:
        f.write(out)

    print(f"Wrote spec-compliant GGUF: {OUT}")
    print(f"  tensors={n_tensors} kv={n_kv} bytes={len(out)}")


if __name__ == "__main__":
    main()
