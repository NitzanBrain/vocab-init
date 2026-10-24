# Same test as test_hull_llama.py, on a model with an UNTIED LM head (Pythia-160m, GPT-NeoX: LayerNorm,
# parallel residual, partial rotary 25%, fused QKV). Question from RESULTS.md: does an untied head behave like
# GPT-2 (negative logits, random init catastrophic) or like SmolLM2 (positive logits, random init harmless)?
# New tokens are added to the OUTPUT matrix only, so all inits are built from embed_out's statistics.
# Run: <venv>/bin/python -I test_hull_neox.py <dir with config.json, model.safetensors, tokenizer.json> <text file>
import sys, json, struct, re, math, pathlib
import numpy as np

d = pathlib.Path(sys.argv[1])
cfg = json.load(open(d / "config.json"))

def load_f16(path):
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__" or v["dtype"] == "U8":  # U8 = stored causal-mask buffers, not weights
            continue
        a, b = v["data_offsets"]
        dt = {"F16": np.float16, "F32": np.float32}[v["dtype"]]  # main branch is F16, step checkpoints are F32
        out[k] = np.frombuffer(raw[base + a:base + b], dtype=dt).reshape(v["shape"]).astype(np.float64)
    return out

W = load_f16(str(d / "model.safetensors"))

# --- tokenizer: GPT-2 byte-level regex + BPE merges from tokenizer.json (NeoX does not split digits) ---
tj = json.load(open(d / "tokenizer.json"))
vocab = tj["model"]["vocab"]
merges = {tuple(m.split(" ") if isinstance(m, str) else m): i for i, m in enumerate(tj["model"]["merges"])}
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
rest = [i for i in range(256) if i not in bs]
b2u = {b: chr(c) for b, c in zip(bs + rest, bs + [256 + n for n in range(len(rest))])}
GPT2 = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")

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
    for w in GPT2.findall(text):
        ids += bpe("".join(b2u[b] for b in w.encode("utf-8")))
    return ids

inv = {v: k for k, v in vocab.items()}
u2b = {u: b for b, u in b2u.items()}
decode = lambda ids: bytes(u2b[c] for i in ids for c in inv[i]).decode("utf-8", "replace")

D, nh, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
hd, eps, theta = D // nh, cfg["layer_norm_eps"], cfg["rotary_emb_base"]
rd = int(hd * cfg["rotary_pct"])
E, U = W["gpt_neox.embed_in.weight"], W["embed_out.weight"]
erf = np.vectorize(math.erf)

def ln(x, p):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * W[p + ".weight"] + W[p + ".bias"]

def rope(x, T):  # rotate only the first rd dims of each head, HF rotate_half layout
    inv_f = 1.0 / theta ** (np.arange(0, rd, 2) / rd)
    ang = np.outer(np.arange(T), inv_f); ang = np.concatenate([ang, ang], -1)
    r, keep = x[..., :rd], x[..., rd:]
    r1, r2 = r[..., :rd // 2], r[..., rd // 2:]
    return np.concatenate([r * np.cos(ang) + np.concatenate([-r2, r1], -1) * np.sin(ang), keep], -1)

def hidden(ids):
    T = len(ids); x = E[ids]
    mask = np.triu(np.full((T, T), -1e10), 1)
    for l in range(L):
        p = f"gpt_neox.layers.{l}."
        h = ln(x, p + "input_layernorm")
        qkv = (h @ W[p + "attention.query_key_value.weight"].T + W[p + "attention.query_key_value.bias"])
        qkv = qkv.reshape(T, nh, 3 * hd).transpose(1, 0, 2)
        q, k, v = qkv[..., :hd], qkv[..., hd:2 * hd], qkv[..., 2 * hd:]
        q, k = rope(q, T), rope(k, T)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        att = (s @ v).transpose(1, 0, 2).reshape(T, D) @ W[p + "attention.dense.weight"].T + W[p + "attention.dense.bias"]
        h2 = ln(x, p + "post_attention_layernorm")
        f = h2 @ W[p + "mlp.dense_h_to_4h.weight"].T + W[p + "mlp.dense_h_to_4h.bias"]
        f = 0.5 * f * (1 + erf(f / np.sqrt(2)))
        x = x + att + f @ W[p + "mlp.dense_4h_to_h.weight"].T + W[p + "mlp.dense_4h_to_h.bias"]  # parallel residual
    return ln(x, "gpt_neox.final_layer_norm")

for prompt in ["The capital of France is", "One, two, three, four,"]:
    ids = encode(prompt)
    for _ in range(5):
        ids.append(int((hidden(ids)[-1] @ U.T).argmax()))
    print("sanity:", repr(prompt), encode(prompt), "->", repr(decode(ids[len(encode(prompt)):])))

ids = encode(open(sys.argv[2], encoding="utf-8").read())[:1024]
H = hidden(ids)
old = H @ U.T
lse = np.logaddexp.reduce(old, -1)
nll = -(old[np.arange(len(H) - 1), ids[1:]] - lse[:-1]).mean()
print(f"tokens={len(H)} ppl={np.exp(nll):.1f} logit max mean={old.max(-1).mean():.1f} logit mean={old.mean():.1f} "
      f"|h| mean={np.linalg.norm(H, axis=1).mean():.1f} out std={U.std():.4f} |mean out|={np.linalg.norm(U.mean(0)):.3f} "
      f"|mean in|={np.linalg.norm(E.mean(0)):.3f} cos(mean out, mean h)={U.mean(0) @ H.mean(0) / np.linalg.norm(U.mean(0)) / np.linalg.norm(H.mean(0)):.3f}")

if sys.argv[3:] == ["geom"]:  # checkpoint sweep: logit geometry only
    sys.exit()
rng = np.random.default_rng(0)
mu, C = U.mean(0), np.cov(U.T)
Lc = np.linalg.cholesky(C + 1e-9 * np.eye(D))
inits = {
    "mean":            lambda K: np.tile(mu, (K, 1)),
    "multivar x1":     lambda K: mu + rng.standard_normal((K, D)) @ Lc.T,
    "univariate":      lambda K: mu + rng.standard_normal((K, D)) * U.std(0),
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
