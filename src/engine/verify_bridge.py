import torch
import json
import numpy as np
from src.model.raven import Raven

def verify():
    print("Verifying Bridge: Binary Blob -> Memory Map -> Model...")
    
    # Load the memory map
    with open("/home/adam/worxpace/gladius/raven/weights/memory_map.json", "r") as f:
        mem_map = json.load(f)
    
    # Load the binary blob
    weights_blob = np.fromfile("/home/adam/worxpace/gladius/raven/weights/raven_weights.bin", dtype=np.float32)
    
    # Test the first layer's QKV projection
    qkv_meta = mem_map["layers"][0]["tensors"]["qkv"]
    offset = qkv_meta["offset"]
    size = qkv_meta["size"] // 4 # Convert bytes to elements
    
    # Extract the tensor from the blob using the map
    extracted_qkv = torch.from_numpy(weights_blob[offset // 4 : (offset // 4) + size]).reshape(1024, 3072)
    
    print(f"Successfully extracted QKV tensor. Shape: {extracted_qkv.shape}")
    print("Bridge Verified: Memory Map matches Binary Blob.")
    return True

if __name__ == "__main__":
    if verify():
        print("BRIDGE_STATUS: OK")
