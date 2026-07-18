# Copyright (c) Text-Span JEPA Authors
# Kaggle dataset loading: WikiText-103 / BookCorpus / C4 small
# Works on Kaggle notebooks with GPU T4/P100

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


class TextDataset(Dataset):
    """Tokenized text dataset for self-supervised pretraining."""

    def __init__(self, token_ids, seq_len=512):
        self.seq_len = seq_len
        self.chunks = []
        for i in range(0, len(token_ids) - seq_len, seq_len):
            self.chunks.append(token_ids[i:i + seq_len])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.chunks[idx], dtype=torch.long)}


def load_wikitext103(tokenizer_name='gpt2', seq_len=512, split='train',
                     data_dir='/kaggle/input/wikitext-103'):
    """Load WikiText-103 dataset for Kaggle."""
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    file_map = {
        'train': 'wiki.train.tokens',
        'valid': 'wiki.valid.tokens',
        'test': 'wiki.test.tokens',
    }

    filepath = None
    for root, dirs, files in os.walk(data_dir):
        target = file_map.get(split)
        if target in files:
            filepath = os.path.join(root, target)
            break

    if filepath is None:
        try:
            from datasets import load_dataset
            ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
            text = '\n'.join(ds['text'])
        except Exception:
            raise FileNotFoundError(
                f"Could not find WikiText-103 {split} data in {data_dir}. "
                f"Add the wikitext-103 dataset to your Kaggle notebook."
            )
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

    token_ids = tokenizer.encode(text)
    print(f"Loaded WikiText-103 {split}: {len(token_ids):,} tokens")
    return TextDataset(token_ids, seq_len=seq_len), tokenizer


def load_bookcorpus(tokenizer_name='gpt2', seq_len=512, data_dir='/kaggle/input/bookcorpus'):
    """Load BookCorpus subset for Kaggle."""
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    try:
        from datasets import load_dataset
        ds = load_dataset('bookcorpus', split='train', streaming=True)
        all_tokens = []
        count = 0
        for item in ds:
            all_tokens.extend(tokenizer.encode(item['text']))
            count += 1
            if count >= 10000:
                break
        print(f"Loaded BookCorpus subset: {len(all_tokens):,} tokens from {count} books")
        return TextDataset(all_tokens, seq_len=seq_len), tokenizer
    except Exception as e:
        print(f"Could not load BookCorpus: {e}")
        print("Falling back to WikiText-103")
        return load_wikitext103(tokenizer_name, seq_len, 'train', data_dir)


def make_dataloader(dataset, batch_size=64, num_workers=2, shuffle=True):
    """Create DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_mask_token_id(tokenizer):
    """Get mask token ID for the tokenizer."""
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    return tokenizer.eos_token_id
