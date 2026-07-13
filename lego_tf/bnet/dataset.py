"""Tokenize a BrickNet split into training sequences for the native LEGO transformer."""

from __future__ import annotations

import bricknet
import torch
from torch.utils.data import Dataset

from lego_tf.bnet import trees as T
from lego_tf.bnet.tokenizer import Vocab, encode_tree

IGNORE = -100


def tokenize_split(npz_path: str, vocab: Vocab, *, limit: int | None = None,
                   seed: int = 0, collision_free: bool = True) -> list[list[int]]:
    """Each single-component graph -> one token sequence (BOS ... EOS)."""
    graphs = bricknet.load_graphs(npz_path)
    n = len(graphs) if limit is None else min(limit, len(graphs))
    seqs = []
    for i in range(n):
        tree = T.sample_tree(graphs[i], seed=seed, collision_free=collision_free)
        seqs.append(encode_tree(tree, vocab))
    return seqs


class SeqDataset(Dataset):
    def __init__(self, seqs: list[list[int]]):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.tensor(self.seqs[i], dtype=torch.long)


def collate(batch, pad_id: int):
    """Right-pad; next-token targets with padding masked to IGNORE."""
    maxlen = max(len(s) for s in batch)
    inp = torch.full((len(batch), maxlen - 1), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), maxlen - 1), IGNORE, dtype=torch.long)
    for i, s in enumerate(batch):
        inp[i, : len(s) - 1] = s[:-1]
        tgt[i, : len(s) - 1] = s[1:]
    return inp, tgt
