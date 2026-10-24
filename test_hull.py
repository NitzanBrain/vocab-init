# Own test of Mundra et al. 2024 (arXiv 2407.05841), Theorem 2:
# new-token embeddings inside the convex hull of the old ones keep the GREEDY output unchanged.
# Question: how much probability do the new tokens take anyway (what sampling sees)?
# GPT-2 small (tied embeddings), English text only, so new tokens matter only through the LM head.
# Run: <venv>/bin/python -I test_hull.py <dir with model.safetensors + r50k.tiktoken> <text file>
import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # bpe_count.py and pretok.py ship in this repo
import numpy as np
from safetensors.numpy import load_file
from bpe_count import load
from pretok import cls

d = pathlib.Path(sys.argv[1])
W = load_file(str(d / "model.safetensors"))
g = lambda k: W[k].astype(np.float64)

L, N = cls("L"), cls("N")
R50K = re.compile(rf"'s|'t|'re|'ve|'m|'ll|'d| ?[{L}]+| ?[{N}]+| ?[^\s{L}{N}]+|\s+(?!\S)|\s+")
ranks = load(str(d / "r50k.tiktoken"))
ranks = ranks[0] if isinstance(ranks, tuple) else ranks

def bpe(b):
    parts = [b[i:i + 1] for i in range(len(b))]
    while len(parts) > 1:
        best = min(((ranks.get(parts[i] + parts[i + 1]), i) for i in range(len(parts) - 1)
                    if parts[i] + parts[i + 1] in ranks), default=None)
        if best is None:
            break
        i = best[1]
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return [ranks[p] for p in parts]

def encode(t):
    return [x for w in R50K.findall(t) for x in bpe(w.encode("utf-8"))]

assert encode("hello world") == [31373, 995], encode("hello world")  # known GPT-2 ids

def ln(x, w, b):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

wte, wpe = g("wte.weight"), g("wpe.weight")

def hidden(ids):
    T = len(ids); x = wte[ids] + wpe[:T]
    mask = np.triu(np.full((T, T), -1e10), 1)
    for l in range(12):
        p = f"h.{l}."
        a = ln(x, g(p + "ln_1.weight"), g(p + "ln_1.bias")) @ g(p + "attn.c_attn.weight") + g(p + "attn.c_attn.bias")
        q, k, v = [z.reshape(T, 12, 64).transpose(1, 0, 2) for z in np.split(a, 3, -1)]
        s = q @ k.transpose(0, 2, 1) / 8 + mask
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(T, 768)
        x = x + o @ g(p + "attn.c_proj.weight") + g(p + "attn.c_proj.bias")
        m = ln(x, g(p + "ln_2.weight"), g(p + "ln_2.bias"))
        x = x + gelu(m @ g(p + "mlp.c_fc.weight") + g(p + "mlp.c_fc.bias")) @ g(p + "mlp.c_proj.weight") + g(p + "mlp.c_proj.bias")
    return ln(x, g("ln_f.weight"), g("ln_f.bias"))

text = open(sys.argv[2], encoding="utf-8").read()
ids = encode(text)
H = np.concatenate([hidden(ids[i:i + 1024]) for i in range(0, min(len(ids), 2048), 1024)])
ids_used = ids[:len(H)]
old = H @ wte.T                                   # (T, 50257) logits
lse = np.logaddexp.reduce(old, -1)
nll = -(old[np.arange(len(H) - 1), ids_used[1:]] - lse[:-1]).mean()
print(f"tokens={len(H)}  GPT-2 ppl on this text={np.exp(nll):.1f}  logit max mean={old.max(-1).mean():.1f}  "
      f"|h| mean={np.linalg.norm(H, axis=1).mean():.1f}  wte std={wte.std():.4f}")

rng = np.random.default_rng(0)
mu, C = wte.mean(0), np.cov(wte.T)
Lc = np.linalg.cholesky(C + 1e-9 * np.eye(768))
inits = {
    "mean":            lambda K: np.tile(mu, (K, 1)),
    "multivar x1e-5":  lambda K: mu + rng.standard_normal((K, 768)) @ (Lc * np.sqrt(1e-5)).T,
    "multivar x1":     lambda K: mu + rng.standard_normal((K, 768)) @ Lc.T,
    "univariate":      lambda K: mu + rng.standard_normal((K, 768)) * wte.std(0),
    "random N(0,.02)": lambda K: rng.normal(0, 0.02, (K, 768)),
    "random N(0,.02)+mean": lambda K: mu + rng.normal(0, 0.02, (K, 768)),
}
print(f"{'init':22s} {'K':>6s} {'greedy kept':>12s} {'new mass mean':>14s} {'p90':>8s} {'max':>8s} {'ppl x':>7s}")
for name, f in inits.items():
    for K in (1000, 25000):
        E = f(K)
        new = H @ E.T
        tot = np.logaddexp(lse, np.logaddexp.reduce(new, -1))
        mass = np.exp(np.logaddexp.reduce(new, -1) - tot)
        kept = (new.max(-1) < old.max(-1)).mean()
        nll2 = -(old[np.arange(len(H) - 1), ids_used[1:]] - tot[:-1]).mean()
        print(f"{name:22s} {K:6d} {kept:12.3f} {mass.mean():14.4f} {np.quantile(mass, .9):8.4f} {mass.max():8.4f} {np.exp(nll2 - nll):7.3f}")
