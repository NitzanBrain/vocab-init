# Exact pre-tokenizer patterns, rebuilt for Python's `re` (which has no \p{..}).
# Sources: tiktoken_ext/openai_public.py (o200k_base, cl100k_base; fetched 2026-10-20, PRIMARY)
#          DictaLM-3.0 tokenizer.json pre_tokenizer (PRIMARY).
# \p{X} classes are generated from unicodedata, so they follow Python's Unicode version, not Rust's (tiny drift possible).
import re, sys, unicodedata
from functools import lru_cache

@lru_cache(None)
def cls(*cats):
    """Regex class body (no brackets) for all code points whose general category starts with any of cats."""
    out, start, prev = [], None, None
    for cp in range(sys.maxunicode + 1):
        c = unicodedata.category(chr(cp))
        hit = any(c.startswith(k) for k in cats)
        if hit and start is None:
            start = cp
        if not hit and start is not None:
            out.append((start, prev)); start = None
        prev = cp
    if start is not None:
        out.append((start, prev))
    esc = lambda cp: "\\U%08x" % cp
    return "".join(esc(a) if a == b else f"{esc(a)}-{esc(b)}" for a, b in out)

def _sub(p):
    L, N = cls("L"), cls("N")
    p = p.replace(r"[^\r\n\p{L}\p{N}]", f"[^\\r\\n{L}{N}]").replace(r"[^\s\p{L}\p{N}]", f"[^\\s{L}{N}]")
    p = p.replace(r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]", f"[{cls('Lu', 'Lt', 'Lm', 'Lo', 'M')}]")
    p = p.replace(r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]", f"[{cls('Ll', 'Lm', 'Lo', 'M')}]")
    return p.replace(r"\p{L}", f"[{L}]").replace(r"\p{N}", f"[{N}]")

O200K = "|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""", r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""", r"""\s*[\r\n]+""", r"""\s+(?!\S)""", r"""\s+"""])
CL100K = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
DICTALM = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

@lru_cache(None)
def pattern(name):
    """name: a tokenizer file name; picks the matching exact pattern."""
    src = O200K if "o200k" in name else CL100K if "cl100k" in name else DICTALM if "DictaLM" in name else None
    if src is None:
        raise ValueError(f"no exact pattern known for {name}")
    return re.compile(_sub(src))
