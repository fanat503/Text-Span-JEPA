# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Main training loop — supports JEPA, MLM, and data2vec baselines
# Training loop patterns from I-JEPA (Assran et al., CVPR 2023):
#   - momentum_scheduler generator (I-JEPA train.py line ~152)
#   - param_groups with WD_exclude (I-JEPA helper.py init_opt)
#   - loss_fn: smooth_l1_loss (I-JEPA train.py loss_fn)
#   - target: layer_norm(h, (h.size(-1),))  (I-JEPA train.py forward_target)
#   - AMP with GradScaler (I-JEPA train.py train_step)
#   - checkpoint saving/loading pattern (I-JEPA train.py save_checkpoint)
#   - AverageMeter, CSVLogger (I-JEPA src/utils/logging.py)

import os
import sys
import csv
import time
import yaml
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.seed import seed_everything, worker_init_fn
from src.utils.logging import CSVLogger, AverageMeter, grad_logger

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# ═══════════════════════════════════════════════════════════════════
#  Model name normalization — handles config suffixes
# ═══════════════════════════════════════════════════════════════════

def _normalize_model_name(raw_name):
    """Normalize model_name from config to canonical form.

    Configs may use suffixed names like 'text_span_jepa_small',
    'mlm_small', 'data2vec_base'. We strip the suffix to get
    the canonical name that create_model() understands.

    Canonical names: text_span_jepa, mlm, data2vec
    """
    name = raw_name.strip().lower()
    if name.startswith('text_span_jepa') or name.startswith('jepa'):
        return 'text_span_jepa'
    if name.startswith('mlm'):
        return 'mlm'
    if name.startswith('data2vec'):
        return 'data2vec'
    return name  # Return as-is, will fail in create_model with clear error


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint save/load — I-JEPA pattern, all model types
# ═══════════════════════════════════════════════════════════════════

def save_checkpoint(path, model, optimizer, scaler, epoch, global_step,
                    ema_step=0, mask_step=0, extra_state=None,
                    model_name='text_span_jepa'):
    """Save complete training state for resumption — all model types.

    Handles JEPA (encoder + predictor + target_encoder + decoder),
    MLM (encoder + mlm_head), and data2vec (encoder + target_encoder + regression_head).
    """
    model_name = _normalize_model_name(model_name)
    state = {
        'model_name': model_name,
        'opt': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'ema_step': ema_step,
        'mask_step': mask_step,
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()

    if model_name == 'text_span_jepa':
        state['encoder'] = model.encoder.state_dict()
        state['predictor'] = model.predictor.state_dict()
        state['target_encoder'] = model.target_encoder.state_dict()
        state['decoder'] = model.decoder.state_dict()
        if hasattr(model, 'target_centering'):
            state['target_centering_center'] = model.target_centering.center.clone()
        # JAWP workspace Q — must be saved for resumption
        if model_name == 'text_span_jepa' and hasattr(model, 'jawp') and model.jawp is not None:
            state['jawp_workspace_Q'] = model.jawp.workspace_Q.data.clone()
            state['jawp_active_k'] = model.jawp.active_k.clone()
        # CGN gate logits — must be saved for resumption
        if model_name == 'text_span_jepa' and hasattr(model, 'cgn') and model.cgn is not None:
            state['cgn_gate_logits_visible'] = model.cgn.gate_logits_visible.data.clone()
            state['cgn_gate_logits_masked'] = model.cgn.gate_logits_masked.data.clone()
            state['cgn_total_steps'] = model.cgn.total_steps.clone()
        # PCR projection Q — must be saved for resumption
        if model_name == 'text_span_jepa' and hasattr(model, 'pcr') and model.pcr is not None:
            state['pcr_workspace_Q'] = model.pcr.workspace_Q.data.clone()
            state['pcr_level_gates'] = [g.data.clone() for g in model.pcr.level_gates]
    elif model_name == 'mlm':
        state['encoder'] = model.encoder.state_dict()
        state['mlm_head'] = model.mlm_head.state_dict()
    elif model_name == 'data2vec':
        state['encoder'] = model.encoder.state_dict()
        state['target_encoder'] = model.target_encoder.state_dict()
        state['regression_head'] = model.regression_head.state_dict()
        if hasattr(model, 'num_updates'):
            state['num_updates'] = model.num_updates

    if extra_state is not None:
        state['extra'] = extra_state
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scaler,
                    model_name='text_span_jepa'):
    """Load checkpoint — I-JEPA helper.py load_checkpoint pattern.

    Handles all model types. Returns (epoch, global_step, ema_step, mask_step, extra_state)
    """
    model_name = _normalize_model_name(model_name)
    try:
        checkpoint = torch.load(path, map_location=torch.device('cpu'),
                                weights_only=False)

        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        ema_step = checkpoint.get('ema_step', 0)
        mask_step = checkpoint.get('mask_step', 0)

        # Determine model type from checkpoint if available
        ckpt_model_name = checkpoint.get('model_name', model_name)
        ckpt_model_name = _normalize_model_name(ckpt_model_name)

        if ckpt_model_name == 'text_span_jepa':
            model.encoder.load_state_dict(checkpoint['encoder'])
            model.predictor.load_state_dict(checkpoint['predictor'])
            model.target_encoder.load_state_dict(checkpoint['target_encoder'])
            model.decoder.load_state_dict(checkpoint['decoder'])
            if 'target_centering_center' in checkpoint and hasattr(model, 'target_centering'):
                model.target_centering.center.copy_(checkpoint['target_centering_center'])
            # JAWP workspace Q restoration
            if 'jawp_workspace_Q' in checkpoint and hasattr(model, 'jawp') and model.jawp is not None:
                model.jawp.workspace_Q.data.copy_(checkpoint['jawp_workspace_Q'])
            if 'jawp_active_k' in checkpoint and hasattr(model, 'jawp') and model.jawp is not None:
                model.jawp.active_k.copy_(checkpoint['jawp_active_k'])
            # CGN gate logits restoration
            if 'cgn_gate_logits_visible' in checkpoint and hasattr(model, 'cgn') and model.cgn is not None:
                model.cgn.gate_logits_visible.data.copy_(checkpoint['cgn_gate_logits_visible'])
            if 'cgn_gate_logits_masked' in checkpoint and hasattr(model, 'cgn') and model.cgn is not None:
                model.cgn.gate_logits_masked.data.copy_(checkpoint['cgn_gate_logits_masked'])
            if 'cgn_total_steps' in checkpoint and hasattr(model, 'cgn') and model.cgn is not None:
                model.cgn.total_steps.copy_(checkpoint['cgn_total_steps'])
            # PCR projection Q restoration
            if 'pcr_workspace_Q' in checkpoint and hasattr(model, 'pcr') and model.pcr is not None:
                model.pcr.workspace_Q.data.copy_(checkpoint['pcr_workspace_Q'])
            if 'pcr_level_gates' in checkpoint and hasattr(model, 'pcr') and model.pcr is not None:
                for i, g in enumerate(checkpoint['pcr_level_gates']):
                    if i < len(model.pcr.level_gates):
                        model.pcr.level_gates[i].data.copy_(g)
        elif ckpt_model_name == 'mlm':
            model.encoder.load_state_dict(checkpoint['encoder'])
            model.mlm_head.load_state_dict(checkpoint['mlm_head'])
        elif ckpt_model_name == 'data2vec':
            model.encoder.load_state_dict(checkpoint['encoder'])
            model.target_encoder.load_state_dict(checkpoint['target_encoder'])
            model.regression_head.load_state_dict(checkpoint['regression_head'])
            if 'num_updates' in checkpoint and hasattr(model, 'num_updates'):
                model.num_updates = checkpoint['num_updates']

        optimizer.load_state_dict(checkpoint['opt'])

        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])

        extra_state = checkpoint.get('extra', None)
        logger.info(f'Loaded checkpoint: epoch={epoch}, step={global_step}, '
                    f'ema_step={ema_step}, mask_step={mask_step}')
        return epoch, global_step, ema_step, mask_step, extra_state

    except Exception as e:
        logger.warning(f'Could not load checkpoint: {e}')
        return 0, 0, 0, 0, None


# ═══════════════════════════════════════════════════════════════════
#  Model creation factory
# ═══════════════════════════════════════════════════════════════════

def create_model(model_name, model_cfg, vocab_size, max_seq_len, device):
    """Create model by type — supports jepa, mlm, data2vec.

    model_name is automatically normalized from config values like
    'text_span_jepa_small' -> 'text_span_jepa'.
    """
    model_name = _normalize_model_name(model_name)

    if model_name == 'text_span_jepa':
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        full_cfg = {**model_cfg, 'vocab_size': vocab_size, 'max_seq_len': max_seq_len}
        config = TextSpanJEPAConfig(**full_cfg)
        config.validate()  # Catch dimension errors early
        model = TextSpanJEPA(config).to(device)
    elif model_name == 'mlm':
        from baselines.mlm_baseline import MLMBaseline
        model = MLMBaseline(
            vocab_size=vocab_size, max_seq_len=max_seq_len,
            embed_dim=model_cfg.get('embed_dim', 768),
            depth=model_cfg.get('encoder_depth', 12),
            num_heads=model_cfg.get('num_heads', 12),
            mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
            drop_rate=model_cfg.get('drop_rate', 0.1),
        ).to(device)
        # Add a .config attribute for compatibility
        model.config = type('Cfg', (), {'lambda_decoder': 0.1,
                                         'lambda_variance': 0.1,
                                         'lambda_covariance': 0.04,
                                         'lambda_span': 1.0,
                                         'lambda_future': 0.5})()
    elif model_name == 'data2vec':
        from baselines.data2vec_baseline import Data2VecTextBaseline
        model = Data2VecTextBaseline(
            vocab_size=vocab_size, max_seq_len=max_seq_len,
            embed_dim=model_cfg.get('embed_dim', 768),
            depth=model_cfg.get('encoder_depth', 12),
            num_heads=model_cfg.get('num_heads', 12),
            mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
            drop_rate=model_cfg.get('drop_rate', 0.0),
            average_top_k_layers=model_cfg.get('average_top_k_layers', 8),
            loss_beta=model_cfg.get('loss_beta', 0.0),
            loss_scale=model_cfg.get('loss_scale', None),
            ema_decay=model_cfg.get('ema_decay', 0.999),
            ema_end_decay=model_cfg.get('ema_end_decay', 0.9999),
            ema_anneal_end_step=model_cfg.get('ema_anneal_end_step', 100000),
            head_layers=model_cfg.get('head_layers', 2),
        ).to(device)
        model.config = type('Cfg', (), {'lambda_decoder': 0.1,
                                         'lambda_variance': 0.1,
                                         'lambda_covariance': 0.04,
                                         'lambda_span': 1.0,
                                         'lambda_future': 0.5})()
    else:
        raise ValueError(f"Unknown model_name: {model_name}. "
                         f"Supported: text_span_jepa, mlm, data2vec")
    return model


def compute_loss(model, masked_input_ids, original_input_ids, mask_positions,
                 current_step=0, total_steps=1):
    """Compute loss for any model type — unified interface.

    Always returns (total_loss, loss_dict, diag_dict) for consistency.
    """
    if hasattr(model, 'compute_loss_with_targets'):
        # JEPA model — returns (loss, loss_dict, diag_dict)
        return model.compute_loss_with_targets(
            masked_input_ids, original_input_ids, mask_positions,
            current_step=current_step, total_steps=total_steps)
    elif hasattr(model, 'forward') and hasattr(model, 'regression_head'):
        # data2vec — returns (loss, info_dict)
        loss, info = model(masked_input_ids, original_input_ids, mask_positions)
        return loss, info, {}
    elif hasattr(model, 'compute_loss'):
        # MLM — returns (loss, info_dict)
        loss, info = model.compute_loss(
            masked_input_ids, original_input_ids, mask_positions)
        return loss, info, {}
    else:
        raise ValueError(f"Model {type(model).__name__} has no supported loss method")


def get_param_groups(model, model_name, wd=0.04):
    """Build optimizer param groups with WD_exclude for bias/norm.

    model_name is automatically normalized.
    """
    model_name = _normalize_model_name(model_name)

    if model_name == 'text_span_jepa':
        return [
            {'params': [p for n, p in model.encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1)]},
            {'params': [p for n, p in model.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1)]},
            {'params': [p for n, p in model.encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)],
             'WD_exclude': True, 'weight_decay': 0},
            {'params': [p for n, p in model.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)],
             'WD_exclude': True, 'weight_decay': 0},
            {'params': list(model.decoder.parameters()),
             'weight_decay': wd},
        ]
    elif model_name == 'mlm':
        # MLM: all encoder params + mlm_head
        return [
            {'params': [p for n, p in model.encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1)]},
            {'params': [p for n, p in model.encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)],
             'WD_exclude': True, 'weight_decay': 0},
            {'params': list(model.mlm_head.parameters()),
             'weight_decay': wd},
        ]
    elif model_name == 'data2vec':
        # data2vec: encoder + regression_head (target encoder is EMA)
        return [
            {'params': [p for n, p in model.encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1)]},
            {'params': [p for n, p in model.encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)],
             'WD_exclude': True, 'weight_decay': 0},
            {'params': list(model.regression_head.parameters()),
             'weight_decay': wd},
        ]
    else:
        return [{'params': list(model.parameters())}]


def do_ema_update(model, model_name, tau=None):
    """Perform EMA update of target encoder — works for all model types.

    model_name is automatically normalized.
    For JEPA: uses scheduled tau from EMATauSchedule.
    For data2vec: uses model's internal get_annealed_decay().
    """
    model_name = _normalize_model_name(model_name)

    if model_name == 'text_span_jepa':
        if tau is not None:
            model.update_target_encoder(tau)
    elif model_name == 'data2vec':
        model.update_target_encoder()
    # MLM has no EMA target — no-op


def _get_all_trainable_params(model):
    """Get all trainable parameters as a single list for global grad clipping."""
    return [p for p in model.parameters() if p.requires_grad]


# ═══════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════

def main(args):
    # ---- Config ----
    seed = args.get('meta', {}).get('seed', 42)
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Normalize model_name once at the start — all downstream functions use it
    raw_model_name = args.get('meta', {}).get('model_name', 'text_span_jepa')
    model_name = _normalize_model_name(raw_model_name)
    logger.info(f'Model type: {raw_model_name} -> {model_name}')

    # ---- Data ----
    data_cfg = args.get('data', {})
    seq_len = data_cfg.get('max_seq_len', 512)

    logger.info('Loading dataset...')
    from src.datasets.kaggle import load_wikitext103, make_dataloader, get_mask_token_id
    dataset, tokenizer = load_wikitext103(
        tokenizer_name=data_cfg.get('tokenizer', 'gpt2'),
        seq_len=seq_len,
        split='train',
        data_dir=data_cfg.get('root_path', '/kaggle/input/wikitext-103'),
    )
    mask_token_id = get_mask_token_id(tokenizer)

    # Validation set
    try:
        val_dataset, _ = load_wikitext103(
            tokenizer_name=data_cfg.get('tokenizer', 'gpt2'),
            seq_len=seq_len,
            split='valid',
            data_dir=data_cfg.get('root_path', '/kaggle/input/wikitext-103'),
        )
        val_dataloader = make_dataloader(
            val_dataset,
            batch_size=data_cfg.get('batch_size', 64),
            num_workers=data_cfg.get('num_workers', 2),
            shuffle=False,
            worker_init_fn=lambda wid: worker_init_fn(wid, seed),
        )
    except Exception:
        logger.warning('No validation set found — training without validation')
        val_dataloader = None

    dataloader = make_dataloader(
        dataset,
        batch_size=data_cfg.get('batch_size', 64),
        num_workers=data_cfg.get('num_workers', 2),
        worker_init_fn=lambda wid: worker_init_fn(wid, seed),
    )

    # ---- Model ----
    model_cfg = args.get('model', {})
    model = create_model(model_name, model_cfg, tokenizer.vocab_size, seq_len, device)

    if hasattr(model, 'get_num_params'):
        num_params = model.get_num_params()
        logger.info(f'Model parameters (non-embedding): {num_params:,}')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {trainable:,}')

    # ---- Mask Collator ----
    # Pass mask curriculum params so mask ratio ramps up during training
    from src.masks.span import SpanMaskCollator
    mask_ratio_start = model_cfg.get('mask_ratio_start', None)
    mask_ratio_end = model_cfg.get('mask_ratio_end', None)
    curriculum_steps = None
    if mask_ratio_start is not None and mask_ratio_end is not None:
        curriculum_steps = 10000  # Default: 10K steps for curriculum

    mask_collator = SpanMaskCollator(
        mask_ratio=data_cfg.get('mask_ratio', 0.35),
        span_length_range=tuple(data_cfg.get('span_length_range', [3, 10])),
        mask_token_id=mask_token_id,
        mask_ratio_start=mask_ratio_start,
        mask_ratio_end=mask_ratio_end,
        curriculum_steps=curriculum_steps or 0,
    )

    # ---- Optimizer + Schedulers ----
    opt_cfg = args.get('optimization', {})
    grad_accum_steps = opt_cfg.get('grad_accum_steps', 1)  # For OOM on small GPUs
    param_groups = get_param_groups(model, model_name,
                                     wd=opt_cfg.get('weight_decay', 0.04))
    optimizer = torch.optim.AdamW(param_groups)

    # AMP: respect device availability
    use_bfloat16 = (args.get('meta', {}).get('use_bfloat16', True)
                    and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_bfloat16)

    ipe = len(dataloader)
    num_epochs = opt_cfg.get('epochs', 50)
    total_steps = int(opt_cfg.get('ipe_scale', 1.0) * num_epochs * ipe)

    from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule, EMATauSchedule
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(opt_cfg.get('warmup', 5) * ipe),
        start_lr=opt_cfg.get('start_lr', 1e-4),
        ref_lr=opt_cfg.get('lr', 1e-3),
        final_lr=opt_cfg.get('final_lr', 1e-5),
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=opt_cfg.get('weight_decay', 0.04),
        final_wd=opt_cfg.get('final_weight_decay', 0.4),
        T_max=total_steps,
    )
    # EMA schedule — only for JEPA models
    if model_name == 'text_span_jepa':
        ema_scheduler = EMATauSchedule(
            tau_start=model_cfg.get('ema_tau_start', 0.996),
            tau_end=model_cfg.get('ema_tau_end', 1.0),
            total_steps=total_steps,
        )
    else:
        # data2vec handles its own EMA internally
        ema_scheduler = None

    # ---- Logging ----
    log_cfg = args.get('logging', {})
    log_dir = log_cfg.get('folder', 'output/')
    os.makedirs(log_dir, exist_ok=True)
    log_freq = log_cfg.get('log_freq', 10)

    # Dump config
    dump_path = os.path.join(log_dir, 'params-text-span-jepa.yaml')
    with open(dump_path, 'w') as f:
        yaml.dump(args, f)

    # CSV loss logger — I-JEPA pattern
    csv_path = os.path.join(log_dir, 'train_log.csv')
    csv_logger = CSVLogger(
        csv_path,
        ('%f', 'loss'), ('%f', 'lr'), ('%f', 'wd'),
        ('%f', 'loss_span'), ('%f', 'loss_future'),
        ('%f', 'loss_decoder'), ('%f', 'loss_variance'),
        ('%f', 'loss_covariance'),
        ('%f', 'effective_rank'), ('%f', 'collapsed_dim_ratio'),
        ('%f', 'mask_fraction'), ('%f', 'decoder_accuracy'),
    )

    # ---- Resume from checkpoint ----
    start_epoch = 0
    global_step = 0
    ema_step = 0
    mask_step = 0
    best_val_loss = float('inf')

    r_file = args.get('meta', {}).get('read_checkpoint', None)
    load_model = args.get('meta', {}).get('load_checkpoint', False)
    latest_path = os.path.join(log_dir, 'checkpoint-latest.pth.tar')

    if load_model:
        load_path = os.path.join(log_dir, r_file) if r_file else latest_path
        if os.path.exists(load_path):
            start_epoch, global_step, ema_step, mask_step, extra = load_checkpoint(
                load_path, model, optimizer, scaler, model_name=model_name)
            if extra and 'best_val_loss' in extra:
                best_val_loss = extra['best_val_loss']
            # Advance schedulers to correct step
            for _ in range(global_step):
                scheduler.step()
                wd_scheduler.step()
                if ema_scheduler is not None:
                    ema_scheduler.step()
            # Advance mask curriculum
            for _ in range(mask_step):
                mask_collator.step()
            logger.info(f'Resumed: epoch={start_epoch}, step={global_step}')

    # ---- Training Loop ----
    logger.info(f'Starting training: {num_epochs} epochs, {total_steps} total steps, '
                f'model={model_name}, grad_accum={grad_accum_steps}')
    logger.info(f'Mask curriculum: start={mask_ratio_start}, end={mask_ratio_end}, '
                f'curriculum_steps={curriculum_steps}')

    for epoch in range(start_epoch, num_epochs):
        loss_meter = AverageMeter()
        model.train()
        epoch_start = time.time()

        for itr, batch in enumerate(dataloader):
            # Collate with masking
            collated = mask_collator(
                [{'input_ids': batch['input_ids'][i]}
                 for i in range(batch['input_ids'].size(0))]
            )
            masked_input_ids = collated['masked_input_ids'].to(device)
            original_input_ids = collated['original_input_ids'].to(device)
            mask_positions = collated['mask_positions'].to(device)

            # LR + WD step
            new_lr = scheduler.step()
            new_wd = wd_scheduler.step()

            # Forward + backward
            autocast_device = device.type if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(autocast_device,
                                     enabled=use_bfloat16,
                                     dtype=torch.bfloat16 if use_bfloat16 else torch.float32):
                total_loss, loss_dict, diag_dict = compute_loss(
                    model, masked_input_ids, original_input_ids, mask_positions,
                    current_step=global_step, total_steps=total_steps)
                # Scale loss for gradient accumulation
                scaled_loss = total_loss / grad_accum_steps

            if use_bfloat16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Only update weights every grad_accum_steps
            if (itr + 1) % grad_accum_steps == 0:
                # Global gradient clipping (I-JEPA pattern: single clip_grad_norm)
                all_trainable = _get_all_trainable_params(model)
                if all_trainable:
                    torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)

                if use_bfloat16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # JAWP Stiefel manifold retraction — MUST be called
                # after every optimizer.step() to keep Q orthonormal.
                # Ref: Absil, Mahony & Sepulchre (2008), §4.1.
                if model_name == 'text_span_jepa' and hasattr(model, 'jawp') and model.jawp is not None:
                    model.jawp.stiefel_retract()
                # PCR Stiefel retraction — keeps cascade projection Q orthonormal
                if model_name == 'text_span_jepa' and hasattr(model, 'pcr') and model.pcr is not None:
                    model.pcr.stiefel_retract()

            # EMA update
            if ema_scheduler is not None:
                tau = ema_scheduler.step()
                do_ema_update(model, model_name, tau)
                ema_step += 1
            elif model_name == 'data2vec':
                do_ema_update(model, model_name)

            mask_collator.step()
            mask_step += 1
            global_step += 1

            loss_val = total_loss.item()  # Unscaled for logging
            loss_meter.update(loss_val)

            # Logging
            if itr % log_freq == 0 or np.isnan(loss_val) or np.isinf(loss_val):
                mem = (torch.cuda.max_memory_allocated() / 1024.**2
                       if device.type == 'cuda' else 0)
                logger.info(
                    f'[{epoch+1}, {itr:5d}] loss={loss_meter.avg:.3f} '
                    f'lr={new_lr:.2e} wd={new_wd:.2e} mem={mem:.0f}MB')
                # Log individual loss components
                logger.info(
                    f'[{epoch+1}, {itr:5d}] losses: '
                    f'span={loss_dict.get("loss_span", 0):.4f} '
                    f'future={loss_dict.get("loss_future", 0):.4f} '
                    f'decoder={loss_dict.get("loss_decoder", 0):.4f} '
                    f'var={loss_dict.get("loss_variance", 0):.4f} '
                    f'cov={loss_dict.get("loss_covariance", 0):.4f} '
                    f'dec_acc={loss_dict.get("decoder_accuracy", 0):.3f}')
                if diag_dict:
                    logger.info(
                        f'[{epoch+1}, {itr:5d}] diag: '
                        f'eff_rank={diag_dict.get("effective_rank_online",0):.1f} '
                        f'collapsed={diag_dict.get("collapsed_dim_ratio_online",0):.3f} '
                        f'mask_frac={diag_dict.get("mask_fraction",0):.2f} '
                        f'target_center_norm={diag_dict.get("target_center_norm",0):.2f} '
                        f'ws_quality={diag_dict.get("workspace_quality",0):.3f}')
                    # JAWP-specific diagnostics
                    if 'jawk_k' in loss_dict:
                        logger.info(
                            f'[{epoch+1}, {itr:5d}] jawp: '
                            f'k={loss_dict.get("jawk_k",0)} '
                            f'ws_util={loss_dict.get("jawk_workspace_utilization",0):.3f} '
                            f'ws_cos={loss_dict.get("jawk_workspace_cosine",0):.3f} '
                            f'ortho={loss_dict.get("jawk_ortho_score",0):.3f} '
                            f'pca_align={loss_dict.get("jawk_pca_alignment",0):.3f}')
                    # CGN-specific diagnostics
                    if 'cgn_tau' in loss_dict:
                        logger.info(
                            f'[{epoch+1}, {itr:5d}] cgn: '
                            f'tau={loss_dict.get("cgn_tau",0):.3f} '
                            f'gate_diff={loss_dict.get("cgn_gate_diff",0):.3f} '
                            f'routing_gap={loss_dict.get("cgn_routing_gap",0):.3f} '
                            f'sparsity={loss_dict.get("cgn_sparsity",0):.3f}')

                # CSV logging
                csv_logger.log(
                    loss_val, new_lr, new_wd,
                    loss_dict.get('loss_span', 0),
                    loss_dict.get('loss_future', 0),
                    loss_dict.get('loss_decoder', 0),
                    loss_dict.get('loss_variance', 0),
                    loss_dict.get('loss_covariance', 0),
                    diag_dict.get('effective_rank_online', 0),
                    diag_dict.get('collapsed_dim_ratio_online', 0),
                    diag_dict.get('mask_fraction', 0),
                    loss_dict.get('decoder_accuracy', 0),
                )

            if np.isnan(loss_val):
                # Save emergency checkpoint before crashing
                logger.error('NaN loss detected! Saving emergency checkpoint...')
                save_checkpoint(
                    os.path.join(log_dir, 'checkpoint-nan.pth.tar'),
                    model, optimizer, scaler, epoch, global_step, ema_step, mask_step,
                    extra_state={'best_val_loss': best_val_loss},
                    model_name=model_name)
                raise RuntimeError(f'Loss is NaN at epoch {epoch+1}, step {global_step}')

        # ---- End of epoch ----
        epoch_time = time.time() - epoch_start
        logger.info(f'Epoch {epoch+1} avg loss: {loss_meter.avg:.4f} '
                    f'time: {epoch_time:.0f}s')

        # ---- Validation ----
        val_loss = None
        if val_dataloader is not None:
            val_loss = _validate(model, val_dataloader, mask_collator, device,
                                  model_name, max_batches=50)
            logger.info(f'  Validation loss: {val_loss:.4f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(log_dir, 'best.pt')
                save_checkpoint(best_path, model, optimizer, scaler,
                               epoch+1, global_step, ema_step, mask_step,
                               extra_state={'best_val_loss': best_val_loss},
                               model_name=model_name)
                logger.info(f'  New best model! val_loss={best_val_loss:.4f}')

        # ---- Checkpoint ----
        save_checkpoint(
            latest_path, model, optimizer, scaler,
            epoch+1, global_step, ema_step, mask_step,
            extra_state={'best_val_loss': best_val_loss},
            model_name=model_name)
        epoch_path = os.path.join(log_dir, f'checkpoint-ep{epoch+1}.pth.tar')
        save_checkpoint(epoch_path, model, optimizer, scaler,
                       epoch+1, global_step, ema_step, mask_step,
                       extra_state={'best_val_loss': best_val_loss},
                       model_name=model_name)
        logger.info(f'Saved checkpoint: {epoch_path}')

    logger.info(f'Training complete! Best val loss: {best_val_loss:.4f}')


def _validate(model, val_dataloader, mask_collator, device, model_name,
              max_batches=50):
    """Run validation and return average loss."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_batches:
                break
            collated = mask_collator(
                [{'input_ids': batch['input_ids'][j]}
                 for j in range(batch['input_ids'].size(0))]
            )
            masked = collated['masked_input_ids'].to(device)
            original = collated['original_input_ids'].to(device)
            mask = collated['mask_positions'].to(device)

            total_loss, _, _ = compute_loss(model, masked, original, mask)
            val_losses.append(total_loss.item())
    model.train()
    return np.mean(val_losses) if val_losses else float('inf')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str,
                        default='config/wikitext/textspanjepa_wikitext_small.yaml')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()
    with open(args.fname, 'r') as f:
        config = yaml.safe_load(f)
    if args.output_dir is not None:
        config.setdefault('logging', {})['folder'] = args.output_dir
    main(config)
