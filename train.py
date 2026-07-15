"""Production-quality single-process trainer for standalone TextSpanJEPA.

This trainer is deliberately simple and auditable. It supports CPU/CUDA/MPS,
fixed-shape batches, deterministic data order, atomic checkpoints, resume,
validation, CSV logging, and collapse/decoder metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np
import torch

from data import FixedBlockDataset, make_loader
from model import TextJEPAConfig, TextSpanJEPA


def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def atomic_save(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device.type == "cpu":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return nullcontext()


def lr_at(step: int, cfg: Dict[str, Any]) -> float:
    warmup = int(cfg.get("warmup", 0))
    total = int(cfg["max_steps"])
    lr = float(cfg["lr"])
    min_lr = float(cfg.get("min_lr", 0.0))
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    p = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * p)) * (lr - min_lr)


def build_optimizer(model: TextSpanJEPA, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    decay, nodecay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(cfg.get("weight_decay", 0.1))},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=float(cfg["lr"]),
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
        eps=float(cfg.get("adam_eps", 1e-8)),
    )


def checkpoint_payload(model: TextSpanJEPA, opt: torch.optim.Optimizer, step: int, best_val: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": int(step), "best_val": float(best_val), "config": cfg}


@torch.no_grad()
def validate(model: TextSpanJEPA, loader, device: torch.device, max_batches: int, use_amp: bool) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    sums: Dict[str, float] = {}
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["input_ids"].to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            out = model(x)
        for k, v in out.items():
            if torch.is_tensor(v) and v.ndim == 0:
                sums[k] = sums.get(k, 0.0) + float(v.detach().float().cpu().item())
        count += 1
    if was_training:
        model.train()
    return {k: v / max(1, count) for k, v in sums.items()} | {"batches": float(count)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    seed_all(int(cfg["seed"]))
    device = get_device(args.device)
    mcfg = TextJEPAConfig(**cfg["model"])

    train_ds = FixedBlockDataset(cfg["train_path"], mcfg.block_size, expected_vocab_size=mcfg.vocab_size)
    val_ds = FixedBlockDataset(cfg["val_path"], mcfg.block_size, expected_vocab_size=mcfg.vocab_size)
    train_loader = make_loader(train_ds, int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg.get("num_workers", 2)))
    val_loader = make_loader(val_ds, int(cfg.get("eval_batch_size", cfg["batch_size"])), shuffle=False, num_workers=int(cfg.get("num_workers", 2)))

    model = TextSpanJEPA(mcfg).to(device)
    opt = build_optimizer(model, cfg)
    step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", best_val))

    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    derived = {
        "tokens_per_update": int(cfg["batch_size"]) * int(mcfg.block_size) * int(cfg.get("grad_accum", 1)),
        "total_tokens": int(cfg["batch_size"]) * int(mcfg.block_size) * int(cfg.get("grad_accum", 1)) * int(cfg["max_steps"]),
        "trainable_params": model.parameter_count()["trainable"],
        "total_stored_params": model.parameter_count()["total_stored"],
    }
    cfg_with_derived = dict(cfg)
    cfg_with_derived["_derived"] = derived
    atomic_write_json(os.path.join(save_dir, "train_config.json"), cfg_with_derived)
    atomic_write_json(os.path.join(save_dir, "dataset_info.json"), {"train": train_ds.info.to_dict(), "val": val_ds.info.to_dict()})
    atomic_write_json(os.path.join(save_dir, "param_report.json"), model.parameter_count())

    log_path = os.path.join(save_dir, f"train_log_{cfg['run_name']}.csv")
    csv_f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    if step == 0:
        writer.writerow([
            "step", "tokens_seen", "lr", "loss", "pred_loss", "future_loss", "decoder_loss", "decoder_acc",
            "var_loss", "cov_loss", "online_std", "target_std", "target_center_norm", "online_target_cosine", "online_target_mse", "ema_tau", "mask_fraction",
            "val_loss", "val_pred_loss", "val_decoder_loss", "val_decoder_acc", "tok_per_sec", "best_val",
        ])

    accum_steps = int(cfg.get("grad_accum", 1))
    block = int(mcfg.block_size)
    batch = int(cfg["batch_size"])
    model.train()
    opt.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    metric_sums: Dict[str, float] = {}
    metric_count = 0
    micro = 0
    t0 = time.time()
    last_step = step

    while step < int(cfg["max_steps"]):
        try:
            batch_obj = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_obj = next(train_iter)
        x = batch_obj["input_ids"].to(device, non_blocking=True)
        lr = lr_at(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.set_training_progress(step, int(cfg["max_steps"]))
        with autocast_context(device, args.amp):
            out = model(x)
            loss = out["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step={step}: {loss}")
        (loss / accum_steps).backward()
        micro += 1
        for k, v in out.items():
            if torch.is_tensor(v) and v.ndim == 0:
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v.detach().float().cpu().item())
        metric_count += 1
        if micro < accum_steps:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        micro = 0
        model.update_target_ema(step, int(cfg["max_steps"]), online_target_cosine=model.last_metrics.get("online_target_cosine"))

        if step % int(cfg.get("log_every", 100)) == 0 or step == 1:
            avg = {k: v / max(1, metric_count) for k, v in metric_sums.items()}
            metric_sums.clear(); metric_count = 0
            val = {"loss": float("nan"), "pred_loss": float("nan"), "decoder_loss": float("nan"), "decoder_accuracy": float("nan")}
            if step % int(cfg.get("val_every", 500)) == 0 or step == 1:
                val = validate(model, val_loader, device, int(cfg.get("val_batches", 20)), args.amp)
                if val.get("loss", float("inf")) < best_val:
                    best_val = val["loss"]
                    atomic_save(os.path.join(save_dir, f"best_val_{cfg['run_name']}.pt"), checkpoint_payload(model, opt, step, best_val, cfg))
            dt = time.time() - t0
            sps = (step - last_step) / max(dt, 1e-9)
            tps = sps * batch * block * accum_steps
            t0 = time.time(); last_step = step
            writer.writerow([
                step, step * batch * block * accum_steps, f"{lr:.8e}",
                avg.get("loss"), avg.get("pred_loss"), avg.get("future_loss"), avg.get("decoder_loss"), avg.get("decoder_accuracy"),
                avg.get("variance_loss"), avg.get("covariance_loss"), avg.get("online_std"), avg.get("target_std"), avg.get("target_center_norm"), avg.get("online_target_cosine"), avg.get("online_target_mse"), avg.get("ema_tau"), avg.get("mask_fraction"),
                val.get("loss"), val.get("pred_loss"), val.get("decoder_loss"), val.get("decoder_accuracy"), tps, best_val,
            ])
            csv_f.flush()
            print(f"step={step} loss={avg.get('loss', float('nan')):.4f} pred={avg.get('pred_loss', float('nan')):.4f} fut={avg.get('future_loss', float('nan')):.4f} dec={avg.get('decoder_loss', float('nan')):.4f} acc={avg.get('decoder_accuracy', float('nan')):.3f} cos={avg.get('online_target_cosine', float('nan')):.3f} tau={avg.get('ema_tau', float('nan')):.6f} val={val.get('loss', float('nan')):.4f} tok/s={tps:.1f}", flush=True)

        if step % int(cfg.get("save_every", 1000)) == 0:
            atomic_save(os.path.join(save_dir, f"latest_{cfg['run_name']}.pt"), checkpoint_payload(model, opt, step, best_val, cfg))

    atomic_save(os.path.join(save_dir, f"final_{cfg['run_name']}.pt"), checkpoint_payload(model, opt, step, best_val, cfg))
    csv_f.close()


if __name__ == "__main__":
    main()
