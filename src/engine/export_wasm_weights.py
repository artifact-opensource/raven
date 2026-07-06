import torch
import json
import os
from src.model.raven import Raven

def export_weights():
    # Initialize the actual Raven model
    model = Raven()
    
    # We will collect all tensors in order
    all_tensors = []
    mem_map_layers = []
    current_offset = 0
    
    # 1. Layers
    for i, layer in enumerate(model.layers):
        layer_tensors = {}
        
        # QKV
        qkv = layer.attn.qkv.weight.detach().numpy().astype('float32')
        size = qkv.nbytes
        layer_tensors["qkv"] = {"offset": current_offset, "shape": qkv.shape, "size": size}
        all_tensors.append(qkv)
        current_offset += size
        
        # MLP Up
        mlp_up = layer.mlp[0].weight.detach().numpy().astype('float32')
        size = mlp_up.nbytes
        layer_tensors["mlp_up"] = {"offset": current_offset, "shape": mlp_up.shape, "size": size}
        all_tensors.append(mlp_up)
        current_offset += size
        
        # MLP Down
        mlp_down = layer.mlp[2].weight.detach().numpy().astype('float32')
        size = mlp_down.nbytes
        layer_tensors["mlp_down"] = {"offset": current_offset, "shape": mlp_down.shape, "size": size}
        all_tensors.append(mlp_down)
        current_offset += size
        
        # PUP Gate
        pup = layer.uncertainty_gate.weight.detach().numpy().astype('float32')
        size = pup.nbytes
        layer_tensors["pup_gate"] = {"offset": current_offset, "shape": pup.shape, "size": size}
        all_tensors.append(pup)
        current_offset += size
        
        mem_map_layers.append({"layer_index": i, "tensors": layer_tensors})

    # 2. Global Params
    global_params = {}
    
    emb = model.token_emb.weight.detach().numpy().astype('float32')
    global_params["token_emb"] = {"offset": current_offset, "shape": emb.shape, "size": emb.nbytes}
    all_tensors.append(emb)
    current_offset += emb.nbytes
    
    ln_f = model.ln_f.weight.detach().numpy().astype('float32')
    global_params["ln_f"] = {"offset": current_offset, "shape": ln_f.shape, "size": ln_f.nbytes}
    all_tensors.append(ln_f)
    current_offset += ln_f.nbytes
    
    head = model.head.weight.detach().numpy().astype('float32')
    global_params["head"] = {"offset": current_offset, "shape": head.shape, "size": head.nbytes}
    all_tensors.append(head)
    current_offset += head.nbytes

    # Save raw binary blob
    output_dir = "/home/adam/worxpace/gladius/raven/weights"
    os.makedirs(output_dir, exist_ok=True)
    
    # Concatenate all tensors into one large numpy array
    final_blob = np.concatenate([t.flatten() for t in all_tensors])
    final_blob.tofile(f"{output_dir}/raven_weights.bin")
    
    # Save the precise memory map
    mem_map = {
        "model_id": "raven-v1",
        "total_binary_size": current_offset,
        "layers": mem_map_layers,
        "global_params": global_params
    }
    
    with open(f"{output_dir}/memory_map.json", "w") as f:
        json.dump(mem_map, f, indent=4)
    
    print(f"Successfully exported ACTUAL weights. Total size: {current_offset} bytes.")

if __name__ == "__main__":
    import numpy as np
    export_weights()
