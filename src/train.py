# Copyright (c) Text-Span JEPA Authors
# Main training loop for Kaggle (single GPU)
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
import yaml
import logging
import numpy as np

import torch
import torch.nn.functional as F

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.masks.span import SpanMaskCollator
from src.datasets.kaggle import load_wikitext103, make_dataloader, get_mask_token_id
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule, EMATauSchedule
from src.utils.logging import CSVLogger, AverageMeter, grad_logger

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(r_path, encoder, predictor, target_encoder, decoder, opt, scaler):
    """Load checkpoint — I-JEPA helper.py load_checkpoint pattern."""
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)

        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'Loaded encoder from epoch {epoch}: {msg}')

        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'Loaded predictor from epoch {epoch}: {msg}')

        pretrained_dict = checkpoint['target_encoder']
        msg = target_encoder.load_state_dict(pretrained_dict)
        logger.info(f'Loaded target_encoder from epoch {epoch}: {msg}')

        pretrained_dict = checkpoint['decoder']
        msg = decoder.load_state_dict(pretrained_dict)
        logger.info(f'Loaded decoder from epoch {epoch}: {msg}')

        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'Loaded optimizers from epoch {epoch}')
        logger.info(f'Read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint: {e}')
        epoch = 0
        global_step = 0

    return encoder, predictor, target_encoder, decoder, opt, scaler, epoch, global_step


def main(args):
    # ---- Config ----
    seed = args.get('meta', {}).get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # ---- Data ----
    data_cfg = args.get('data', {})
    seq_len = data_cfg.get('max_seq_len', 512)

    logger.info('Loading dataset...')
    dataset, tokenizer = load_wikitext103(
        tokenizer_name=data_cfg.get('tokenizer', 'gpt2'),
        seq_len=seq_len,
        split='train',
        data_dir=data_cfg.get('root_path', '/kaggle/input/wikitext-103'),
    )
    mask_token_id = get_mask_token_id(tokenizer)

    dataloader = make_dataloader(
        dataset,
        batch_size=data_cfg.get('batch_size', 64),
        num_workers=data_cfg.get('num_workers', 2),
    )

    # ---- Model ----
    model_cfg = {**args.get('model', {}), 'vocab_size': tokenizer.vocab_size, 'max_seq_len': seq_len}
    config = TextSpanJEPAConfig(**model_cfg)
    config.vocab_size = tokenizer.vocab_size
    model = TextSpanJEPA(config).to(device)

    num_params = model.get_num_params()
    logger.info(f'Model parameters: {num_params:,}')

    # ---- Mask Collator ----
    mask_collator = SpanMaskCollator(
        mask_ratio=data_cfg.get('mask_ratio', 0.35),
        span_length_range=tuple(data_cfg.get('span_length_range', [3, 10])),
        mask_token_id=mask_token_id,
        mask_ratio_start=config.mask_ratio_start,
        mask_ratio_end=config.mask_ratio_end,
        curriculum_steps=args.get('optimization', {}).get('epochs', 50) * len(dataloader),
    )

    # ---- Optimizer + Schedulers — I-JEPA init_opt pattern ----
    opt_cfg = args.get('optimization', {})
    # I-JEPA param_groups: separate WD for encoder/predictor, exclude bias/norm
    param_groups = [
        {'params': (p for n, p in model.encoder.named_parameters()
                    if ('bias' not in n) and (len(p.shape) != 1))},
        {'params': (p for n, p in model.predictor.named_parameters()
                    if ('bias' not in n) and (len(p.shape) != 1))},
        {'params': (p for n, p in model.encoder.named_parameters()
                    if ('bias' in n) or (len(p.shape) == 1)),
         'WD_exclude': True, 'weight_decay': 0},
        {'params': (p for n, p in model.predictor.named_parameters()
                    if ('bias' in n) or (len(p.shape) == 1)),
         'WD_exclude': True, 'weight_decay': 0},
        {'params': model.decoder.parameters()},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    use_bfloat16 = args.get('meta', {}).get('use_bfloat16', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_bfloat16)

    ipe = len(dataloader)
    num_epochs = opt_cfg.get('epochs', 50)
    total_steps = int(opt_cfg.get('ipe_scale', 1.0) * num_epochs * ipe)

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
    ema_scheduler = EMATauSchedule(
        tau_start=config.ema_tau_start,
        tau_end=config.ema_tau_end,
        total_steps=total_steps,
    )

    # ---- Logging ----
    log_cfg = args.get('logging', {})
    log_dir = log_cfg.get('folder', 'output/')
    os.makedirs(log_dir, exist_ok=True)
    log_freq = log_cfg.get('log_freq', 10)

    # Dump config (I-JEPA pattern)
    dump_path = os.path.join(log_dir, 'params-text-span-jepa.yaml')
    with open(dump_path, 'w') as f:
        yaml.dump(args, f)

    # ---- Resume from checkpoint if available ----
    start_epoch = 0
    global_step = 0
    r_file = args.get('meta', {}).get('read_checkpoint', None)
    load_model = args.get('meta', {}).get('load_checkpoint', False)
    latest_path = os.path.join(log_dir, 'checkpoint-latest.pth.tar')
    if load_model:
        load_path = os.path.join(log_dir, r_file) if r_file else latest_path
        if os.path.exists(load_path):
            model.encoder, model.predictor, model.target_encoder, model.decoder, \
                optimizer, scaler, start_epoch, global_step = load_checkpoint(
                    load_path, model.encoder, model.predictor,
                    model.target_encoder, model.decoder, optimizer, scaler)
            # Advance schedulers to correct step
            for _ in range(global_step):
                scheduler.step()
                wd_scheduler.step()
                ema_scheduler.step()
                mask_collator.step()
            logger.info(f'Resumed from epoch {start_epoch}, step {global_step}')

    # ---- Training Loop — I-JEPA pattern ----
    logger.info(f'Starting training: {num_epochs} epochs, {total_steps} total steps')

    for epoch in range(start_epoch, num_epochs):
        loss_meter = AverageMeter()

        for itr, batch in enumerate(dataloader):
            # Collate with span masking
            collated = mask_collator(
                [{'input_ids': batch['input_ids'][i]}
                 for i in range(batch['input_ids'].size(0))]
            )
            masked_input_ids = collated['masked_input_ids'].to(device)
            original_input_ids = collated['original_input_ids'].to(device)
            mask_positions = collated['mask_positions'].to(device)

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # I-JEPA: forward with autocast
                with torch.amp.autocast('cuda', enabled=use_bfloat16, dtype=torch.bfloat16):
                    total_loss, loss_dict, diag_dict = model.compute_loss_with_targets(
                        masked_input_ids, original_input_ids, mask_positions,
                        current_step=global_step, total_steps=total_steps,
                    )

                # I-JEPA: backward with scaler
                if use_bfloat16:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

                # I-JEPA: momentum update of target encoder
                with torch.no_grad():
                    m = ema_scheduler.step()
                    model.update_target_encoder(m)

                mask_collator.step()
                grad_stats = grad_logger(model.encoder.named_parameters())

                return float(total_loss), _new_lr, _new_wd, grad_stats

            loss_val, new_lr, new_wd, grad_stats = train_step()
            loss_meter.update(loss_val)
            global_step += 1

            # I-JEPA logging pattern
            if itr % log_freq == 0 or np.isnan(loss_val) or np.isinf(loss_val):
                logger.info(
                    '[%d, %5d] loss: %.3f '
                    '[wd: %.2e] [lr: %.2e] '
                    '[mem: %.2e] '
                    % (epoch + 1, itr,
                       loss_meter.avg,
                       new_wd, new_lr,
                       torch.cuda.max_memory_allocated() / 1024.**2
                       if device.type == 'cuda' else 0)
                )
                if grad_stats is not None:
                    logger.info(
                        '[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                        % (epoch + 1, itr,
                           grad_stats.first_layer, grad_stats.last_layer,
                           grad_stats.min, grad_stats.max)
                    )

            assert not np.isnan(loss_val), 'loss is nan'

        logger.info('avg. loss %.3f' % loss_meter.avg)

        # I-JEPA checkpoint pattern — save all state for resumption
        ckpt_path = os.path.join(log_dir, f'checkpoint-ep{epoch + 1}.pth.tar')
        save_dict = {
            'encoder': model.encoder.state_dict(),
            'predictor': model.predictor.state_dict(),
            'target_encoder': model.target_encoder.state_dict(),
            'decoder': model.decoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'epoch': epoch + 1,
            'global_step': global_step,
            'loss': loss_meter.avg,
        }
        torch.save(save_dict, latest_path)
        torch.save(save_dict, ckpt_path)
        logger.info(f'Saved checkpoint: {ckpt_path}')

    logger.info('Training complete!')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    with open(args.fname, 'r') as f:
        config = yaml.safe_load(f)
    main(config)
