"""Sterile fixed-token datasets for Text-JEPA.

Supports either:
  - `.pt`: a 1-D integer torch.Tensor, or dict with key `tokens`;
  - `.bin` + sidecar JSON: raw int32/int64/int16 token ids.

Each item is a deterministic non-overlapping block of length `block_size`.
Unlike LM training, JEPA does not need `block_size + 1` targets; all targets are
inside the same sequence via masked-span latent prediction.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

VALID_DTYPES = (torch.int16, torch.int32, torch.int64)


def _load_tokens(path: str, *, mmap: bool = True) -> torch.Tensor:
    if path.endswith(".bin"):
        sidecars = [path + ".json", os.path.splitext(path)[0] + ".json"]
        meta_path = next((x for x in sidecars if os.path.exists(x)), None)
        if meta_path is None:
            raise FileNotFoundError(f"missing .bin sidecar; tried {sidecars}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        dtype_map = {"int16": torch.int16, "int32": torch.int32, "int64": torch.int64}
        dtype = dtype_map.get(str(meta.get("dtype")))
        if dtype is None:
            raise TypeError(f"unsupported sidecar dtype={meta.get('dtype')!r}")
        n = int(meta["num_tokens"])
        expected = n * torch.empty((), dtype=dtype).element_size()
        actual = os.path.getsize(path)
        if expected != actual:
            raise ValueError(f".bin size mismatch: expected {expected:,}, got {actual:,}")
        return torch.from_file(path, shared=False, size=n, dtype=dtype)

    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    kwargs["weights_only"] = True
    try:
        obj = torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None); kwargs.pop("weights_only", None)
        obj = torch.load(path, **kwargs)
    if isinstance(obj, dict) and "tokens" in obj:
        obj = obj["tokens"]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"expected tensor in {path}, got {type(obj).__name__}")
    return obj


@dataclass(frozen=True)
class DatasetInfo:
    path: str
    total_tokens: int
    block_size: int
    n_sequences: int
    tokens_used: int
    tokens_dropped: int
    dtype: str
    sample_min: int
    sample_max: int
    sample_fingerprint: str
    expected_vocab_size: Optional[int]

    def to_dict(self):
        return asdict(self)


class FixedBlockDataset(Dataset):
    def __init__(
        self,
        path: str,
        block_size: int,
        *,
        expected_vocab_size: Optional[int] = None,
        validate_full: bool = False,
        mmap: bool = True,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if block_size <= 1:
            raise ValueError("block_size must be > 1")
        self.path = path
        self.block_size = int(block_size)
        self.expected_vocab_size = expected_vocab_size
        self.tokens = _load_tokens(path, mmap=mmap)
        if self.tokens.ndim != 1:
            raise ValueError(f"expected 1-D tensor, got {tuple(self.tokens.shape)}")
        if self.tokens.dtype not in VALID_DTYPES:
            raise TypeError(f"invalid dtype {self.tokens.dtype}; expected {VALID_DTYPES}")
        if self.tokens.device.type != "cpu":
            raise ValueError("tokens must be CPU-backed")
        if len(self.tokens) < self.block_size:
            raise ValueError("dataset shorter than one block")
        self.n_sequences = len(self.tokens) // self.block_size
        self.tokens_used = self.n_sequences * self.block_size
        self.tokens_dropped = len(self.tokens) - self.tokens_used
        mn, mx = self._sample_range()
        if validate_full:
            mn, mx = int(self.tokens.min()), int(self.tokens.max())
        self._validate_range(mn, mx)
        self.info = DatasetInfo(
            path=os.path.abspath(path), total_tokens=len(self.tokens), block_size=self.block_size,
            n_sequences=self.n_sequences, tokens_used=self.tokens_used, tokens_dropped=self.tokens_dropped,
            dtype=str(self.tokens.dtype), sample_min=mn, sample_max=mx,
            sample_fingerprint=self._fingerprint(), expected_vocab_size=expected_vocab_size,
        )

    def _validate_range(self, mn: int, mx: int) -> None:
        if mn < 0:
            raise ValueError(f"negative token id: {mn}")
        if self.expected_vocab_size is not None and mx >= self.expected_vocab_size:
            raise ValueError(f"token id {mx} >= vocab {self.expected_vocab_size}")

    def _sample_range(self) -> tuple[int, int]:
        n = len(self.tokens); w = min(4096, n)
        starts = sorted(set([0, max(0, n // 2 - w // 2), max(0, n - w)]))
        mn, mx = 10**18, -10**18
        for s in starts:
            c = self.tokens.narrow(0, s, min(w, n - s))
            mn = min(mn, int(c.min())); mx = max(mx, int(c.max()))
        return mn, mx

    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        n = len(self.tokens); w = min(8192, n)
        for s in [0, max(0, n // 2 - w // 2), max(0, n - w)]:
            c = self.tokens.narrow(0, s, min(w, n - s)).contiguous()
            h.update(str(s).encode()); h.update(c.numpy().tobytes())
        h.update(str(n).encode()); h.update(str(self.tokens.dtype).encode())
        return h.hexdigest()[:24]

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0:
            idx += self.n_sequences
        if idx < 0 or idx >= self.n_sequences:
            raise IndexError(idx)
        start = idx * self.block_size
        return {"input_ids": self.tokens.narrow(0, start, self.block_size)}


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed); random.seed(seed)


def collate_token_blocks(batch):
    xs = [b["input_ids"] for b in batch]
    return {"input_ids": torch.stack(xs).to(torch.long)}


def make_loader(ds: Dataset, batch_size: int, *, shuffle: bool, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=True,
        num_workers=num_workers, collate_fn=collate_token_blocks,
        worker_init_fn=worker_init_fn, persistent_workers=bool(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
