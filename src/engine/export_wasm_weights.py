import torch
import json
import os
from src.model.raven import Raven

def export_weights():
    model = Raven()
    # In a real scenario, we'd load the SVD-projected weights first
    weights_blob = torch.randn(1000000).numpy().astype('float32') # Placeholder for actual weights
    
    output_dir = "/home/adam/worxpace/gladius/raven/weights"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw binary blob
    weights_blob.tofile(f"{output_dir}/raven_weights.bin")
    
    # Create the Memory Map
    mem_map = {
        "model_id": "raven-v1",
        "total_size": weights_blob.nbytes,
        "layers": [
            {"name": "layer_0_qkv", "offset": 0, "size": 1024 * 3072 * 4},
            {"name": "layer_0_mlp", "offset": 1024 * 3072 * 4, "size": 1024 * 4096 * 4},
        ]
    }
    
    with open(f"{output_dir}/memory_map.json", "w") as f:
        json.dump(mem_map, f, indent=4)
    
    print("Weights exported to binary blob and memory_map.json created.")

if __name__ == "__main__":
    export_weights()
