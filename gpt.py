"""
mygpt - a tiny GPT (transformer language model) written from scratch in pure NumPy.

No PyTorch, no GPU. Everything a real GPT has, just much smaller:
  token + position embeddings -> N transformer blocks
  (LayerNorm -> multi-head causal self-attention -> residual,
   LayerNorm -> MLP with GELU -> residual) -> LayerNorm -> output head
Backpropagation is written by hand, and training uses the Adam optimizer.

Usage:
  python gpt.py train  --data input.txt --steps 3000
  python gpt.py chat                       # type a prompt, it continues it
  python gpt.py generate --prompt "Once upon a time" --length 400
  python gpt.py gradcheck                  # verifies the backprop math
"""
import argparse, json, math, os, sys, time
import numpy as np

DT = np.float32


# ----------------------------------------------------------------------------- tokenizer
class CharTokenizer:
    """Character-level tokenizer: every distinct character is one token."""
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.chars)


# ----------------------------------------------------------------------------- building blocks
def layernorm_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_bwd(dy, cache):
    xhat, rstd, g = cache
    n = xhat.shape[-1]
    dg = (dy * xhat).reshape(-1, n).sum(0)
    db = dy.reshape(-1, n).sum(0)
    dxhat = dy * g
    dx = rstd / n * (n * dxhat - dxhat.sum(-1, keepdims=True)
                     - xhat * (dxhat * xhat).sum(-1, keepdims=True))
    return dx, dg, db


GELU_C = math.sqrt(2.0 / math.pi)


def gelu_fwd(x):
    t = np.tanh(GELU_C * (x + 0.044715 * x ** 3))
    return 0.5 * x * (1.0 + t), (x, t)


def gelu_bwd(dy, cache):
    x, t = cache
    dt = GELU_C * (1.0 + 3 * 0.044715 * x ** 2)
    return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dt)


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


# ----------------------------------------------------------------------------- the model
class GPT:
    def __init__(self, vocab_size, n_layer=3, n_head=4, n_embd=96, block_size=64, seed=0):
        assert n_embd % n_head == 0
        self.cfg = dict(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                        n_embd=n_embd, block_size=block_size)
        rng = np.random.default_rng(seed)
        C, V = n_embd, vocab_size
        std = 0.02
        proj_std = 0.02 / math.sqrt(2 * n_layer)   # GPT-2 style scaled init for residual projections

        def w(*shape, s=std):
            return (rng.standard_normal(shape) * s).astype(DT)

        p = {"wte": w(V, C), "wpe": w(block_size, C)}
        for l in range(n_layer):
            p[f"h{l}.ln1.g"] = np.ones(C, DT);  p[f"h{l}.ln1.b"] = np.zeros(C, DT)
            p[f"h{l}.attn.wqkv"] = w(C, 3 * C); p[f"h{l}.attn.bqkv"] = np.zeros(3 * C, DT)
            p[f"h{l}.attn.wo"] = w(C, C, s=proj_std); p[f"h{l}.attn.bo"] = np.zeros(C, DT)
            p[f"h{l}.ln2.g"] = np.ones(C, DT);  p[f"h{l}.ln2.b"] = np.zeros(C, DT)
            p[f"h{l}.mlp.w1"] = w(C, 4 * C);    p[f"h{l}.mlp.b1"] = np.zeros(4 * C, DT)
            p[f"h{l}.mlp.w2"] = w(4 * C, C, s=proj_std); p[f"h{l}.mlp.b2"] = np.zeros(C, DT)
        p["lnf.g"] = np.ones(C, DT); p["lnf.b"] = np.zeros(C, DT)
        p["head"] = w(C, V)
        self.p = p

    def num_params(self):
        return sum(v.size for v in self.p.values())

    # ---- attention
    def _attn_fwd(self, x, l):
        p, H = self.p, self.cfg["n_head"]
        B, T, C = x.shape
        hs = C // H
        qkv = x @ p[f"h{l}.attn.wqkv"] + p[f"h{l}.attn.bqkv"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(hs)
        s = (q @ k.transpose(0, 1, 3, 2)) * scale
        mask = np.triu(np.ones((T, T), bool), 1)        # causal: can't look at the future
        s = np.where(mask, -1e9, s)
        P = softmax(s).astype(DT)
        y = (P @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        out = y @ p[f"h{l}.attn.wo"] + p[f"h{l}.attn.bo"]
        return out, (x, q, k, v, P, y, scale)

    def _attn_bwd(self, dout, cache, l, g):
        p, H = self.p, self.cfg["n_head"]
        x, q, k, v, P, y, scale = cache
        B, T, C = x.shape
        hs = C // H
        g[f"h{l}.attn.wo"] += y.reshape(-1, C).T @ dout.reshape(-1, C)
        g[f"h{l}.attn.bo"] += dout.reshape(-1, C).sum(0)
        dy = (dout @ p[f"h{l}.attn.wo"].T).reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        dP = dy @ v.transpose(0, 1, 3, 2)
        dv = P.transpose(0, 1, 3, 2) @ dy
        ds = P * (dP - (dP * P).sum(-1, keepdims=True)) * scale
        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q
        dqkv = np.concatenate([t.transpose(0, 2, 1, 3).reshape(B, T, C) for t in (dq, dk, dv)], -1)
        g[f"h{l}.attn.wqkv"] += x.reshape(-1, C).T @ dqkv.reshape(-1, 3 * C)
        g[f"h{l}.attn.bqkv"] += dqkv.reshape(-1, 3 * C).sum(0)
        return dqkv @ p[f"h{l}.attn.wqkv"].T

    # ---- full forward (returns logits and everything needed for backward)
    def forward(self, idx):
        p = self.p
        B, T = idx.shape
        assert T <= self.cfg["block_size"]
        x = p["wte"][idx] + p["wpe"][:T]
        caches = []
        for l in range(self.cfg["n_layer"]):
            h, c_ln1 = layernorm_fwd(x, p[f"h{l}.ln1.g"], p[f"h{l}.ln1.b"])
            a, c_attn = self._attn_fwd(h, l)
            x = x + a
            h2, c_ln2 = layernorm_fwd(x, p[f"h{l}.ln2.g"], p[f"h{l}.ln2.b"])
            z = h2 @ p[f"h{l}.mlp.w1"] + p[f"h{l}.mlp.b1"]
            act, c_gelu = gelu_fwd(z)
            m = act @ p[f"h{l}.mlp.w2"] + p[f"h{l}.mlp.b2"]
            x = x + m
            caches.append((c_ln1, c_attn, c_ln2, h2, c_gelu, act))
        xf, c_lnf = layernorm_fwd(x, p["lnf.g"], p["lnf.b"])
        logits = xf @ p["head"]
        return logits, (idx, caches, xf, c_lnf)

    def loss_and_grads(self, idx, targets):
        p = self.p
        logits, (idx, caches, xf, c_lnf) = self.forward(idx)
        B, T, V = logits.shape
        C = self.cfg["n_embd"]
        probs = softmax(logits.astype(np.float64))
        n = B * T
        flat_t = targets.reshape(-1)
        loss = -np.log(probs.reshape(n, V)[np.arange(n), flat_t] + 1e-12).mean()

        g = {k: np.zeros_like(v) for k, v in p.items()}
        dlogits = probs.reshape(n, V)
        dlogits[np.arange(n), flat_t] -= 1.0
        dlogits = (dlogits / n).astype(DT).reshape(B, T, V)

        g["head"] += xf.reshape(-1, C).T @ dlogits.reshape(-1, V)
        dx = dlogits @ p["head"].T
        dx, g["lnf.g"], g["lnf.b"] = layernorm_bwd(dx, c_lnf)

        for l in reversed(range(self.cfg["n_layer"])):
            c_ln1, c_attn, c_ln2, h2, c_gelu, act = caches[l]
            # MLP branch
            g[f"h{l}.mlp.w2"] += act.reshape(-1, 4 * C).T @ dx.reshape(-1, C)
            g[f"h{l}.mlp.b2"] += dx.reshape(-1, C).sum(0)
            dact = dx @ p[f"h{l}.mlp.w2"].T
            dz = gelu_bwd(dact, c_gelu)
            g[f"h{l}.mlp.w1"] += h2.reshape(-1, C).T @ dz.reshape(-1, 4 * C)
            g[f"h{l}.mlp.b1"] += dz.reshape(-1, 4 * C).sum(0)
            dh2 = dz @ p[f"h{l}.mlp.w1"].T
            d, dg_, db_ = layernorm_bwd(dh2, c_ln2)
            g[f"h{l}.ln2.g"] += dg_; g[f"h{l}.ln2.b"] += db_
            dx = dx + d
            # attention branch
            dh = self._attn_bwd(dx, c_attn, l, g)
            d, dg_, db_ = layernorm_bwd(dh, c_ln1)
            g[f"h{l}.ln1.g"] += dg_; g[f"h{l}.ln1.b"] += db_
            dx = dx + d

        np.add.at(g["wte"], idx, dx)
        g["wpe"][:T] += dx.sum(0)
        return float(loss), g

    def loss(self, idx, targets):
        logits, _ = self.forward(idx)
        B, T, V = logits.shape
        probs = softmax(logits.astype(np.float64)).reshape(-1, V)
        return float(-np.log(probs[np.arange(B * T), targets.reshape(-1)] + 1e-12).mean())

    # ---- text generation
    def generate(self, ids, n_new, temperature=0.8, top_k=None, rng=None):
        rng = rng or np.random.default_rng()
        ids = list(ids) or [0]
        bs = self.cfg["block_size"]
        for _ in range(n_new):
            ctx = np.array([ids[-bs:]])
            logits, _ = self.forward(ctx)
            logits = logits[0, -1].astype(np.float64) / max(temperature, 1e-6)
            if top_k:
                cut = np.sort(logits)[-top_k]
                logits = np.where(logits < cut, -1e9, logits)
            probs = softmax(logits)
            ids.append(int(rng.choice(len(probs), p=probs)))
        return ids

    # ---- save / load
    def save(self, path, tok):
        np.savez(path, **self.p, __cfg__=json.dumps(self.cfg), __chars__=json.dumps(tok.chars))

    @classmethod
    def load(cls, path):
        f = np.load(path)
        cfg = json.loads(str(f["__cfg__"]))
        m = cls(**cfg)
        for k in m.p:
            m.p[k] = f[k]
        return m, CharTokenizer(json.loads(str(f["__chars__"])))


# ----------------------------------------------------------------------------- Adam optimizer
class AdamW:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.99), wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd = params, lr, *betas, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, lr=None):
        lr = lr or self.lr
        self.t += 1
        for k, w in self.p.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            if w.ndim == 2:                       # weight decay on matrices only
                w -= lr * self.wd * w
            w -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(DT)


def clip_grads(g, max_norm=1.0):
    norm = math.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for v in g.values()))
    if norm > max_norm:
        for v in g.values():
            v *= max_norm / norm
    return norm


# ----------------------------------------------------------------------------- commands
def get_batch(data, B, T, rng):
    ix = rng.integers(0, len(data) - T - 1, B)
    x = np.stack([data[i:i + T] for i in ix])
    y = np.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


def cmd_train(a):
    text = open(a.data, encoding="utf-8").read()
    tok = CharTokenizer.from_text(text)
    data = np.array(tok.encode(text), dtype=np.int64)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]
    model = GPT(tok.vocab_size, a.n_layer, a.n_head, a.n_embd, a.block_size, seed=a.seed)
    print(f"text: {len(text):,} chars | vocab: {tok.vocab_size} | params: {model.num_params():,}")
    opt = AdamW(model.p, lr=a.lr)
    rng = np.random.default_rng(a.seed)
    warm = max(1, a.steps // 20)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        # learning rate: linear warmup, then cosine decay to 10%
        if step <= warm:
            lr = a.lr * step / warm
        else:
            pr = (step - warm) / max(1, a.steps - warm)
            lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * pr)))
        x, y = get_batch(train, a.batch_size, a.block_size, rng)
        loss, g = model.loss_and_grads(x, y)
        clip_grads(g, 1.0)
        opt.step(g, lr)
        if step % a.eval_every == 0 or step == 1 or step == a.steps:
            vx, vy = get_batch(val if len(val) > a.block_size + 1 else train, 32, a.block_size, rng)
            print(f"step {step:5d} | train loss {loss:.3f} | val loss {model.loss(vx, vy):.3f} "
                  f"| lr {lr:.5f} | {time.time() - t0:.0f}s", flush=True)
    model.save(a.out, tok)
    print(f"saved model to {a.out}")
    sample = model.generate(tok.encode(text[:1]), 300, rng=rng)
    print("\n--- sample ---\n" + tok.decode(sample))


def cmd_generate(a):
    model, tok = GPT.load(a.model)
    ids = model.generate(tok.encode(a.prompt), a.length, a.temperature, a.top_k)
    print(tok.decode(ids))


def cmd_chat(a):
    model, tok = GPT.load(a.model)
    print("mygpt is ready. Type a beginning and it will continue it. Ctrl+C to quit.\n")
    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print(); break
        ids = model.generate(tok.encode(prompt), a.length, a.temperature, a.top_k)
        print("gpt> " + tok.decode(ids) + "\n")


def cmd_gradcheck(a):
    """Compare hand-written gradients with numerical ones to prove backprop is correct."""
    global DT
    DT = np.float64
    rng = np.random.default_rng(1)
    m = GPT(vocab_size=11, n_layer=2, n_head=2, n_embd=8, block_size=6, seed=1)
    for k in m.p:
        m.p[k] = m.p[k].astype(np.float64) + rng.standard_normal(m.p[k].shape) * 0.1
    x = rng.integers(0, 11, (2, 6)); y = rng.integers(0, 11, (2, 6))
    _, g = m.loss_and_grads(x, y)
    worst = 0.0
    for k, w in m.p.items():
        for _ in range(4):
            i = tuple(rng.integers(0, s) for s in w.shape)
            old = w[i]; eps = 1e-5
            w[i] = old + eps; lp = m.loss(x, y)
            w[i] = old - eps; lm = m.loss(x, y)
            w[i] = old
            num = (lp - lm) / (2 * eps)
            rel = abs(num - g[k][i]) / max(1e-8, abs(num) + abs(g[k][i]))
            worst = max(worst, rel)
    print(f"max relative gradient error: {worst:.2e}  ->  {'PASS' if worst < 1e-4 else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description="mygpt: a tiny GPT in pure NumPy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="input.txt")
    t.add_argument("--out", default="mygpt.npz")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--block_size", type=int, default=64)
    t.add_argument("--n_layer", type=int, default=3)
    t.add_argument("--n_head", type=int, default=4)
    t.add_argument("--n_embd", type=int, default=96)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--eval_every", type=int, default=200)
    t.add_argument("--seed", type=int, default=1337)
    for name in ("generate", "chat"):
        s = sub.add_parser(name)
        s.add_argument("--model", default="mygpt.npz")
        s.add_argument("--prompt", default="")
        s.add_argument("--length", type=int, default=300)
        s.add_argument("--temperature", type=float, default=0.8)
        s.add_argument("--top_k", type=int, default=None)
    sub.add_parser("gradcheck")
    a = ap.parse_args()
    {"train": cmd_train, "generate": cmd_generate, "chat": cmd_chat, "gradcheck": cmd_gradcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
