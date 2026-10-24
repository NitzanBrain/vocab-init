# Same test as test_hull.py, on a LLaMA-architecture model (SmolLM2-135M: RMSNorm, RoPE, GQA, tied embeddings).
# Question from RESULTS.md caveat: is GPT-2's ≈ −90 logit offset (which makes random init catastrophic) general?
# Run: <venv>/bin/python -I test_hull_llama.py <dir with config.json, model.safetensors, tokenizer.json> <text file>
import sys, json, struct, re, pathlib
import numpy as np

d = pathlib.Path(sys.argv[1])
cfg = json.load(open(d / "config.json"))

def load_bf16(path):
    """safetensors reader that handles BF16 (numpy has no bfloat16): widen to float32 by bit shift."""
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        assert v["dtype"] == "BF16", v["dtype"]
        u = np.frombuffer(raw[base + a:base + b], dtype=np.uint16).astype(np.uint32) << 16
        out[k] = u.view(np.float32).reshape(v["shape"]).astype(np.float64)
    return out

W = load_bf16(str(d / "model.safetensors"))

# --- tokenizer: individual digits, then GPT-2 byte-level regex + BPE merges from tokenizer.json ---
tj = json.load(open(d / "tokenizer.json"))
vocab = tj["model"]["vocab"]
merges = {tuple(m.split(" ") if isinstance(m, str) else m): i for i, m in enumerate(tj["model"]["merges"])}
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
rest = [i for i in range(256) if i not in bs]
b2u = {b: chr(c) for b, c in zip(bs + rest, bs + [256 + n for n in range(len(rest))])}
GPT2 = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")
# NOTE: [^\W\d_] approximates \p{L} (includes some marks); fine for English test text.

def bpe(word):
    parts = list(word)
    while len(parts) > 1:
        cand = [(merges.get((parts[i], parts[i + 1])), i) for i in range(len(parts) - 1)]
        cand = [c for c in cand if c[0] is not None]
        if not cand:
            break
        _, i = min(cand)
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return [vocab[p] for p in parts]

def encode(text):
    ids = []
    for chunk in re.findall(r"\d|\D+", text):
        for w in ([chunk] if chunk.isdigit() else GPT2.findall(chunk)):
            ids += bpe("".join(b2u[b] for b in w.encode("utf-8")))
    return ids

inv = {v: k for k, v in vocab.items()}
u2b = {u: b for b, u in b2u.items()}
decode = lambda ids: bytes(u2b[c] for i in ids for c in inv[i]).decode("utf-8", "replace")

D, nh, nkv, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_hidden_layers"]
hd, eps, theta = D // nh, cfg["rms_norm_eps"], cfg["rope_theta"]
E = W["model.embed_tokens.weight"]

def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

def rope(x, T):  # x: (heads, T, hd), HF non-interleaved (rotate_half)
    inv_f = 1.0 / theta ** (np.arange(0, hd, 2) / hd)
    ang = np.outer(np.arange(T), inv_f); ang = np.concatenate([ang, ang], -1)
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return x * np.cos(ang) + np.concatenate([-x2, x1], -1) * np.sin(ang)

def hidden(ids):
    T = len(ids); x = E[ids]
    mask = np.triu(np.full((T, T), -1e10), 1)
    for l in range(L):
        p = f"model.layers.{l}."
        h = rms(x, W[p + "input_layernorm.weight"])
        q = (h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, nh, hd).transpose(1, 0, 2)
        k = (h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)
        q, k = rope(q, T), rope(k, T)
        k, v = np.repeat(k, nh // nkv, 0), np.repeat(v, nh // nkv, 0)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        x = x + (s @ v).transpose(1, 0, 2).reshape(T, D) @ W[p + "self_attn.o_proj.weight"].T
        h = rms(x, W[p + "post_attention_layernorm.weight"])
        gte = h @ W[p + "mlp.gate_proj.weight"].T
        x = x + ((gte / (1 + np.exp(-gte))) * (h @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
    return rms(x, W["model.norm.weight"])

for prompt in ["The capital of France is", "One, two, three, four,"]:
    ids = encode(prompt)
    for _ in range(5):
        ids.append(int((hidden(ids)[-1] @ E.T).argmax()))
    print("sanity:", repr(prompt), "->", repr(decode(ids[len(encode(prompt)):])))

ids = encode(open(sys.argv[2], encoding="utf-8").read())[:1024]
H = hidden(ids)
old = H @ E.T
lse = np.logaddexp.reduce(old, -1)
nll = -(old[np.arange(len(H) - 1), ids[1:]] - lse[:-1]).mean()
print(f"tokens={len(H)} ppl={np.exp(nll):.1f} logit max mean={old.max(-1).mean():.1f} logit mean={old.mean():.1f} "
      f"|h| mean={np.linalg.norm(H, axis=1).mean():.1f} emb std={E.std():.4f} |mean emb|={np.linalg.norm(E.mean(0)):.3f}")

rng = np.random.default_rng(0)
mu, C = E.mean(0), np.cov(E.T)
Lc = np.linalg.cholesky(C + 1e-9 * np.eye(D))
inits = {
    "mean":            lambda K: np.tile(mu, (K, 1)),
    "multivar x1":     lambda K: mu + rng.standard_normal((K, D)) @ Lc.T,
    "univariate":      lambda K: mu + rng.standard_normal((K, D)) * E.std(0),
    "random N(0,.02)": lambda K: rng.normal(0, 0.02, (K, D)),
    "random N(0,.02)+mean": lambda K: mu + rng.normal(0, 0.02, (K, D)),
}
print(f"{'init':22s} {'K':>6s} {'greedy kept':>12s} {'new mass mean':>14s} {'p90':>8s} {'max':>8s} {'ppl x':>9s}")
for name, f in inits.items():
    for K in (1000, 25000):
        new = H @ f(K).T
        nl = np.logaddexp.reduce(new, -1); tot = np.logaddexp(lse, nl); mass = np.exp(nl - tot)
        kept = (new.max(-1) < old.max(-1)).mean()
        nll2 = -(old[np.arange(len(H) - 1), ids[1:]] - tot[:-1]).mean()
        print(f"{name:22s} {K:6d} {kept:12.3f} {mass.mean():14.4f} {np.quantile(mass, .9):8.4f} {mass.max():8.4f} {np.exp(nll2 - nll):9.3g}")
