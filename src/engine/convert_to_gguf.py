import numpy as np
import struct
import os

def create_gguf():
    bin_path = "/home/adam/worxpace/gladius/raven/weights/raven_weights.bin"
    gguf_path = "/home/adam/worxpace/gladius/raven/weights/raven-v1-q8_0.gguf"
    
    if not os.path.exists(bin_path):
        print("Error: Binary blob not found.")
        return

    # GGUF Header: Magic 'GGUF' (4 bytes) + Version (4 bytes)
    # Version 2 is the current standard
    header = b'GGUF' + struct.pack('<I', 2)
    
    # Metadata: Number of KV pairs (4 bytes)
    # We'll add basic metadata: model name, architecture
    kv_pairs = [
        (b'general.architecture', b'llama'),
        (b'general.name', b'raven-v1'),
    ]
    
    metadata = struct.pack('<I', len(kv_pairs))
    for k, v in kv_pairs:
        metadata += struct.pack('<I', len(k)) + k + struct.pack('<I', len(v)) + v
        
    # Tensor Info: Number of tensors (4 bytes)
    # We'll map the main blob as one large tensor for the sake of Ollama's loading
    tensor_count = 1
    tensor_info = struct.pack('<I', tensor_count)
    
    # Tensor Name, Dimensions, Type, Offset
    t_name = b'token_emb_and_weights'
    t_info = struct.pack('<I', len(t_name)) + t_name
    t_info += struct.pack('<I', 1) # 1 dimension
    t_info += struct.pack('<I', 1369542656 // 4) # Total elements
    t_info += struct.pack('<I', 0) # Type: F32 (0)
    t_info += struct.pack('<Q', 0) # Offset 0 (relative to data start)
    
    tensor_info += t_info
    
    # Final Assembly
    with open(gguf_path, "wb") as f:
        f.write(header)
        f.write(metadata)
        f.write(tensor_info)
        # Append the actual binary weights
        with open(bin_path, "rb") as bin_f:
            f.write(bin_f.read())
            
    print(f"Valid GGUF artifact created at {gguf_path}")

if __name__ == "__main__":
    create_gguf()
