"""
ByteTokenizer — Native byte-level tokenizer for GLADIUS machine code corpus.

No BPE. No subword. No merges. Raw bytes in, raw bytes out.
256 byte tokens (0x00-0xFF) + 3 specials (PAD=256, BOS=257, EOS=258).

The CPU's native language doesn't need human linguistic scaffolding.
"""

import json
from pathlib import Path
from typing import List, Union


class ByteTokenizer:
    """Byte-level tokenizer: one token per byte value (0-255) + specials."""

    PAD_ID = 256
    BOS_ID = 257
    EOS_ID = 258
    VOCAB_SIZE = 259  # 256 bytes + 3 specials

    def __init__(self):
        self._special_tokens = {
            "<PAD>": self.PAD_ID,
            "<BOS>": self.BOS_ID,
            "<EOS>": self.EOS_ID,
        }
        self._id_to_special = {v: k for k, v in self._special_tokens.items()}

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def encode(self, data: Union[bytes, bytearray], add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode raw bytes to token IDs.
        
        Args:
            data: Raw bytes to encode.
            add_bos: Prepend BOS token.
            add_eos: Append EOS token.
            
        Returns:
            List of integer token IDs.
        """
        tokens = []
        if add_bos:
            tokens.append(self.BOS_ID)
        tokens.extend(int(b) for b in data)
        if add_eos:
            tokens.append(self.EOS_ID)
        return tokens

    def decode(self, token_ids: List[int], strip_special: bool = True) -> bytes:
        """Decode token IDs back to raw bytes.
        
        Args:
            token_ids: List of integer token IDs.
            strip_special: Remove PAD/BOS/EOS from output.
            
        Returns:
            Raw bytes.
        """
        result = bytearray()
        for tid in token_ids:
            if strip_special and tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            if 0 <= tid <= 255:
                result.append(tid)
            # IDs outside 0-258 are silently skipped
        return bytes(result)

    def encode_hex(self, hex_string: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode a hex string (e.g., '4889e548...' or '48 89 e5 48...') to tokens."""
        clean = hex_string.replace(" ", "").replace("\n", "")
        data = bytes.fromhex(clean)
        return self.encode(data, add_bos=add_bos, add_eos=add_eos)

    def decode_hex(self, token_ids: List[int], strip_special: bool = True) -> str:
        """Decode token IDs to hex string."""
        raw = self.decode(token_ids, strip_special=strip_special)
        return raw.hex()

    def pad(self, token_ids: List[int], max_len: int) -> List[int]:
        """Pad or truncate to max_len."""
        if len(token_ids) >= max_len:
            return token_ids[:max_len]
        return token_ids + [self.PAD_ID] * (max_len - len(token_ids))

    def save(self, path: Union[str, Path]):
        """Save tokenizer config as JSON."""
        config = {
            "type": "ByteTokenizer",
            "vocab_size": self.VOCAB_SIZE,
            "byte_range": [0, 255],
            "special_tokens": self._special_tokens,
            "description": "Native byte-level tokenizer for machine code. 256 byte values + PAD/BOS/EOS."
        }
        Path(path).write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ByteTokenizer":
        """Load tokenizer from JSON config (validates, returns fresh instance)."""
        config = json.loads(Path(path).read_text())
        assert config["type"] == "ByteTokenizer", f"Wrong tokenizer type: {config['type']}"
        assert config["vocab_size"] == cls.VOCAB_SIZE, f"Vocab size mismatch: {config['vocab_size']}"
        return cls()

    def __repr__(self):
        return f"ByteTokenizer(vocab_size={self.VOCAB_SIZE}, bytes=0-255, specials=PAD/BOS/EOS)"


# Quick self-test
if __name__ == "__main__":
    tok = ByteTokenizer()
    
    # Test basic encode/decode roundtrip
    test_bytes = bytes([0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x10])  # mov rbp,rsp; sub rsp,0x10
    encoded = tok.encode(test_bytes)
    decoded = tok.decode(encoded)
    assert decoded == test_bytes, f"Roundtrip failed: {decoded.hex()} != {test_bytes.hex()}"
    
    # Test hex encode/decode
    hex_encoded = tok.encode_hex("4889e54883ec10")
    assert tok.decode(hex_encoded) == test_bytes
    
    # Test special tokens
    assert encoded[0] == tok.BOS_ID
    assert encoded[-1] == tok.EOS_ID
    assert len(encoded) == len(test_bytes) + 2
    
    # Test padding
    padded = tok.pad(encoded, 20)
    assert len(padded) == 20
    assert padded[-1] == tok.PAD_ID
    
    # Test save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tok.save(f.name)
        tok2 = ByteTokenizer.load(f.name)
        assert tok2.vocab_size == tok.vocab_size
        os.unlink(f.name)
    
    # Test all 256 byte values roundtrip
    all_bytes = bytes(range(256))
    assert tok.decode(tok.encode(all_bytes)) == all_bytes
    
    print(f"✅ ByteTokenizer: {tok}")
    print(f"   Roundtrip test: {test_bytes.hex()} → {encoded} → {decoded.hex()}")
    print(f"   All 256 byte values roundtrip: PASS")
    print(f"   Vocab: 256 bytes + PAD({tok.PAD_ID}) + BOS({tok.BOS_ID}) + EOS({tok.EOS_ID}) = {tok.VOCAB_SIZE}")
