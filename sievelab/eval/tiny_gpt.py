"""A tiny character-level GPT (decoder-only transformer) trained from scratch.

Used for the fixed-compute raw-vs-clean comparison: both models get the same
architecture, vocabulary, number of optimisation steps and batch size, so the
only variable is the training data. Requires ``torch``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


if torch is not None:

    class Block(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.attn = nn.MultiheadAttention(cfg.n_embd, cfg.n_head, dropout=cfg.dropout, batch_first=True)
            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                                     nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

        def forward(self, x, mask):
            h = self.ln1(x)
            x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
            return x + self.mlp(self.ln2(x))

    class TinyGPT(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.cfg = cfg
            self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            self.head.weight = self.tok.weight

        def forward(self, idx):
            T = idx.size(1)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
            x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
            for b in self.blocks:
                x = b(x, mask)
            return self.head(self.ln(x))


class CharVocab:
    UNK = 0

    def __init__(self, chars: Sequence[str]):
        self.itos = ["<unk>"] + sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, self.UNK) for c in text]

    def __len__(self) -> int:
        return len(self.itos)


def train(
    train_texts: Sequence[str],
    vocab: CharVocab,
    steps: int = 600,
    batch_size: int = 32,
    block_size: int = 128,
    lr: float = 3e-3,
    seed: int = 0,
    n_embd: int = 128,
    n_layer: int = 2,
):
    if torch is None:
        raise ImportError("torch is required for the tiny GPT experiment")
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    data = np.array(vocab.encode("\n\n".join(train_texts)), dtype=np.int64)
    cfg = GPTConfig(vocab_size=len(vocab), block_size=block_size, n_embd=n_embd, n_layer=n_layer)
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    for _ in range(steps):
        ix = rng.randint(0, len(data) - block_size - 1, size=batch_size)
        x = torch.from_numpy(np.stack([data[i : i + block_size] for i in ix]))
        y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + block_size] for i in ix]))
        loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.final_train_loss = float(loss.item())
    return model


def evaluate(model, eval_texts: Sequence[str], vocab: CharVocab) -> Dict[str, float]:
    cfg = model.cfg
    block_size = cfg.block_size
    ev = np.array(vocab.encode("\n\n".join(eval_texts)), dtype=np.int64)
    model.eval()
    nll, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for s in range(0, len(ev) - 1, block_size):
            chunk = ev[s : s + block_size + 1]
            if len(chunk) < 2:
                break
            x = torch.from_numpy(chunk[:-1][None])
            y = torch.from_numpy(chunk[1:][None])
            logits = model(x)
            nll += F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1), reduction="sum").item()
            correct += (logits.argmax(-1) == y).sum().item()
            n += y.numel()
    bpc = nll / n / math.log(2)
    return {"bpc": round(bpc, 4), "char_ppl": round(2**bpc, 4), "top1_acc": round(correct / n, 4), "n_chars": n}
