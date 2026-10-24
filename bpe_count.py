# Pure-Python byte-level BPE token counter (no tiktoken/tokenizers needed).
# Loads tiktoken rank files (*.tiktoken) or HF tokenizer.json (byte-level BPE).
# Pre-tokenizer is approximated: \p{L} -> [^\W\d_], \p{N} -> \d (Python re has no \p{}).
# Run: python3 -I bpe_count.py <tokenizer file> <text file, one segment per line>
import base64, json, re, sys

PRE = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?[^\W\d_]+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

def byte_unicode_map():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

def load(path):
    """Returns ranks: bytes -> rank (lower merges first)."""
    if path.endswith(".tiktoken"):
        ranks = {}
        for line in open(path, "rb"):
            tok, r = line.split()
            ranks[base64.b64decode(tok)] = int(r)
        return ranks
    t = json.load(open(path, encoding="utf-8"))
    u2b = byte_unicode_map()
    tob = lambda s: bytes(u2b[ch] for ch in s)
    ranks = {}
    for b in range(256):
        ranks[bytes([b])] = -1
    for i, m in enumerate(t["model"]["merges"]):
        a, b = m if isinstance(m, list) else m.split(" ")
        ranks.setdefault(tob(a) + tob(b), i)
    # tokens present in vocab but never produced by a merge are reachable only via ignore_merges
    whole = {tob(s) for s in t["model"]["vocab"]} if t["model"].get("ignore_merges") else set()
    return ranks, whole

def bpe(piece, ranks):
    parts = [piece[i:i + 1] for i in range(len(piece))]
    while len(parts) > 1:
        best, idx = None, -1
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best is None or r < best):
                best, idx = r, i
        if idx < 0:
            break
        parts[idx:idx + 2] = [parts[idx] + parts[idx + 1]]
    return len(parts)

def count(text, ranks, whole=frozenset()):
    n = 0
    for p in PRE.findall(text):
        b = p.encode("utf-8")
        n += 1 if b in whole else bpe(b, ranks)
    return n

if __name__ == "__main__":
    loaded = load(sys.argv[1])
    ranks, whole = loaded if isinstance(loaded, tuple) else (loaded, frozenset())
    lines = open(sys.argv[2], encoding="utf-8").read().splitlines()
    print(sum(count(l, ranks, whole) for l in lines))
