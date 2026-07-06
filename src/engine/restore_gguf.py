import numpy as np
import os

def restore():
    bin_path = "/home/adam/worxpace/gladius/raven/weights/raven_weights.bin"
    gguf_path = "/home/adam/worxpace/gladius/raven/weights/raven-v1-q8_0.gguf"
    
    if os.path.exists(bin_path):
        print("Restoring GGUF from binary blob...")
        with open(bin_path, "rb") as f:
            data = f.read()
        with open(gguf_path, "wb") as f:
            f.write(data)
        print("GGUF restored.")

if __name__ == "__main__":
    restore()
