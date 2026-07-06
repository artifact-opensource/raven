"""
GLADIUS MathTokenizer — Structural tokenizer for mathematical expressions.
128 tokens, purpose-built for mathematical reasoning.
Every token IS the mathematical object — no BPE fragmentation.
"""

from typing import List

class MathTokenizer:
    """
    Structural tokenizer for mathematical expressions.
    128 tokens in range 17000-17127 (but internally remapped to 0-127).
    
    Every token IS the mathematical object â€” no BPE fragmentation.
    Grid tokenizer proved: purpose-built tokens â†’ 91% loss drop.
    """

    # Internal vocab (0-indexed for nn.Embedding)
    VOCAB_SIZE = 128
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    # Original range (for reference/compatibility)
    ORIGINAL_BASE = 17000
    ORIGINAL_RANGE = (17000, 17127)

    def __init__(self):
        # Build token maps (0-indexed)
        self._build_maps()

    def _build_maps(self):
        """Build all token mappings."""
        self.token_to_id = {}
        self.id_to_token = {}
        idx = 0

        # Specials: 0-2
        for name in ['[PAD_MATH]', '[BOS_MATH]', '[EOS_MATH]']:
            self.token_to_id[name] = idx
            self.id_to_token[idx] = name
            idx += 1

        # Digits 0-9: 3-12
        for i in range(10):
            self.token_to_id[str(i)] = idx
            self.id_to_token[idx] = str(i)
            idx += 1

        # Operators: 13-30
        for op in ['+', '-', '*', '/', '=', '^', 'âˆš', '!', '%', '<', '>',
                    'â‰¤', 'â‰¥', 'â‰ ', 'â‰ˆ', 'â†’', 'âˆˆ', '.']:
            self.token_to_id[op] = idx
            self.id_to_token[idx] = op
            idx += 1

        # Variables a-z: 31-56
        for i in range(26):
            ch = chr(ord('a') + i)
            self.token_to_id[ch] = idx
            self.id_to_token[idx] = ch
            idx += 1

        # Greek/constants: 57-69
        for g in ['Ï€', 'e_const', 'Ï†', 'i', 'âˆž', 'Î±', 'Î²', 'Î³', 'Î´', 'Î¸', 'Î»', 'Ïƒ', 'Ï‰']:
            self.token_to_id[g] = idx
            self.id_to_token[idx] = g
            idx += 1

        # Functions: 70-89
        for f in ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'lim', 'Î£', 'âˆ«', 'd/dx',
                   'abs', 'floor', 'ceil', 'max', 'min', 'gcd', 'lcm', 'mod', 'det', 'tr']:
            self.token_to_id[f] = idx
            self.id_to_token[idx] = f
            idx += 1

        # Structure/delimiters: 90-114
        for s in ['(', ')', '[', ']', '{', '}', ',', ';', ':', '_', '|', '\\', ' ', '\n',
                   'â€¦', 'âˆ´', 'âˆµ', 'âŸ¨', 'âŸ©', 'QED', 'GIVEN', 'STEP', 'OUT', 'FILL', 'NEXT']:
            self.token_to_id[s] = idx
            self.id_to_token[idx] = s
            idx += 1

        # Multi-char sorted by length for greedy matching
        self._multi_char = {k: v for k, v in self.token_to_id.items() if len(k) > 1}
        self._sorted_multi = sorted(self._multi_char.keys(), key=len, reverse=True)
        self._single_char = {k: v for k, v in self.token_to_id.items() if len(k) == 1}

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode mathematical expression to token IDs (0-indexed)."""
        tokens = []
        if add_special:
            tokens.append(self.BOS_ID)

        i = 0
        text = text.strip()
        while i < len(text):
            if text[i] == ' ':
                i += 1
                continue

            # Multi-char (longest match first)
            matched = False
            for multi in self._sorted_multi:
                if text[i:i+len(multi)] == multi:
                    tokens.append(self._multi_char[multi])
                    i += len(multi)
                    matched = True
                    break
            if matched:
                continue

            # Single-char
            ch = text[i]
            if ch in self._single_char:
                tokens.append(self._single_char[ch])
                i += 1
                continue

            # R as remainder
            if ch == 'R':
                tokens.append(self.token_to_id.get(',', 96))
                i += 1
                continue

            # Uppercase â†’ lowercase variable
            if ch.isupper() and ch.lower() in self._single_char:
                tokens.append(self._single_char[ch.lower()])
                i += 1
                continue

            i += 1  # skip unknown

        if add_special:
            tokens.append(self.EOS_ID)
        return tokens

    def encode_corpus_line(self, line: str) -> List[int]:
        """Encode a corpus line with D{n}| prefix and |GIVEN:|STEP:|QED:|OUT: segments."""
        import re
        line = line.strip()
        tokens = [self.BOS_ID]

        # Handle D{n}| prefix
        if re.match(r'^D\d\|', line):
            tokens.append(self.token_to_id[line[1]])  # digit
            tokens.append(self.token_to_id['|'])
            line = line[3:]

        segments = line.split('|')
        for seg_idx, segment in enumerate(segments):
            if seg_idx > 0:
                tokens.append(self.token_to_id['|'])

            for label in ['GIVEN:', 'QED:', 'OUT:', 'FILL:', 'NEXT:']:
                if segment.startswith(label):
                    key = label[:-1]  # strip colon
                    tokens.append(self.token_to_id.get(key, 0))
                    tokens.append(self.token_to_id[':'])
                    tokens.extend(self.encode(segment[len(label):], add_special=False))
                    break
            else:
                # Check STEP{n}:
                step_match = re.match(r'^STEP(\d+):', segment)
                if step_match:
                    tokens.append(self.token_to_id['STEP'])
                    for d in step_match.group(1):
                        tokens.append(self.token_to_id[d])
                    tokens.append(self.token_to_id[':'])
                    tokens.extend(self.encode(segment[step_match.end():], add_special=False))
                else:
                    tokens.extend(self.encode(segment, add_special=False))

        tokens.append(self.EOS_ID)
        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to string."""
        parts = []
        for tid in token_ids:
            if skip_special and tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            parts.append(self.id_to_token.get(tid, f'<?{tid}>'))
        return ''.join(parts)


# â”€â”€â”€ Byte Tokenizer (inline â€” no external dependency) â”€â”€â”€â”€â”€â”€â”€â”€
