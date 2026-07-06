import hashlib

def get_hash():
    path = "/home/adam/worxpace/gladius/raven/weights/raven_weights.bin"
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

if __name__ == "__main__":
    print(f"BINARY_HASH={get_hash()}")
