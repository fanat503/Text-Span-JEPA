"""
Standalone Text-Span JEPA.

A full JEPA-style representation learner for token sequences:
  online encoder sees a masked/corrupted sequence;
  EMA target encoder sees the original sequence;
  predictor predicts target latents at masked positions;
  optional tied token decoder grounds predicted latents lexically;
  VICReg-style variance/covariance losses reduce collapse risk.

No dependency on hla-v4. No autoregressive LM head. This is a representation
learning model, not a generative LLM.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextJEPAConfig:
    # Token/model shape
    vocab_size: int = 50257
    padded_vocab_size: int = 50304
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-5

    # Position/attention
    use_rope: bool = True
    rope_theta: float = 10000.0
    attention_backend: str = "manual"  # "manual" | "sdpa"
    bidirectional_context: bool = True  # JEPA sees visible tokens on both sides by default

    # MLP
    ffn_hidden_multiple_of: int = 64
    fused_swiglu: bool = True

    # Masking
    mask_token_id: Optional[int] = None
    mask_ratio: float = 0.25
    span_length: int = 8
    min_masked_tokens: int = 1
    mask_seed: int = 1234
    mask_ratio_start: Optional[float] = None
    mask_curriculum_steps: int = 0

    # Target view. "full" is standard JEPA teacher view. "target_only" and
    # "local_window" are stricter controls to test whether the target encoder is
    # leaking too much surrounding context into the target representation.
    target_view_mode: str = "full"  # "full" | "target_only" | "local_window"
    target_local_window: int = 32

    # JEPA objective
    predictor_depth: int = 2
    predictor_mult: int = 2
    predictor_refine_steps: int = 1  # latent "thinking" iterations inside predictor
    predictor_residual_scale: float = 0.5
    position_aware_predictor: bool = True
    separate_future_predictor: bool = True
    normalize_targets: bool = True
    center_targets: bool = True
    target_center_momentum: float = 0.9
    pred_loss: str = "mse"  # "mse" | "cosine"

    # Optional NextLat-style future latent prediction. This reuses the same
    # online/target forward pass, so it adds only predictor/loss compute.
    use_future_prediction: bool = True
    future_weight: float = 0.25
    future_offsets: tuple[int, ...] = (1, 4, 16)

    # EMA target
    ema_tau_start: float = 0.996
    ema_tau_end: float = 0.9995
    adaptive_ema_tau: bool = True
    ema_adapt_strength: float = 0.001
    ema_cosine_target: float = 0.25
    ema_tau_min: float = 0.990

    # Collapse guards
    variance_weight: float = 0.01
    covariance_weight: float = 0.001
    variance_target_std: float = 1.0

    # Optional decoder grounding
    use_decoder: bool = True
    decoder_weight: float = 0.1
    decoder_hidden_mult: int = 1
    tie_decoder_to_embeddings: bool = True

    # Init
    init_std: float = 0.02
    residual_init_scale: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.padded_vocab_size < self.vocab_size:
            raise ValueError("padded_vocab_size must be >= vocab_size")
        if self.mask_token_id is None:
            self.mask_token_id = self.padded_vocab_size - 1
        if not (self.vocab_size <= self.mask_token_id < self.padded_vocab_size):
            raise ValueError("mask_token_id must be in padded-only range [vocab_size, padded_vocab_size)")
        if self.block_size <= 1:
            raise ValueError("block_size must be > 1")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be 'manual' or 'sdpa'")
        if not (0.0 < self.mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1)")
        if self.span_length <= 0:
            raise ValueError("span_length must be positive")
        if self.min_masked_tokens <= 0:
            raise ValueError("min_masked_tokens must be positive")
        if self.mask_ratio_start is not None and not (0.0 < self.mask_ratio_start < 1.0):
            raise ValueError("mask_ratio_start must be in (0,1) when provided")
        if self.mask_curriculum_steps < 0:
            raise ValueError("mask_curriculum_steps must be non-negative")
        if self.target_view_mode not in {"full", "target_only", "local_window"}:
            raise ValueError("target_view_mode must be full, target_only, or local_window")
        if self.target_local_window < 0:
            raise ValueError("target_local_window must be non-negative")
        if self.predictor_depth <= 0 or self.predictor_mult <= 0:
            raise ValueError("predictor_depth and predictor_mult must be positive")
        if self.predictor_refine_steps <= 0:
            raise ValueError("predictor_refine_steps must be positive")
        if self.predictor_residual_scale < 0:
            raise ValueError("predictor_residual_scale must be non-negative")
        if self.pred_loss not in {"mse", "cosine"}:
            raise ValueError("pred_loss must be 'mse' or 'cosine'")
        if self.future_weight < 0:
            raise ValueError("future_weight must be non-negative")
        if any(int(x) <= 0 for x in self.future_offsets):
            raise ValueError("future_offsets must be positive")
        if self.future_offsets and max(self.future_offsets) >= self.block_size:
            raise ValueError("max(future_offsets) must be < block_size")
        if not (0.0 <= self.target_center_momentum < 1.0):
            raise ValueError("target_center_momentum must be in [0,1)")
        if not (0.0 <= self.ema_tau_start < 1.0 and 0.0 <= self.ema_tau_end < 1.0):
            raise ValueError("EMA tau values must be in [0, 1)")
        if self.ema_tau_end < self.ema_tau_start:
            raise ValueError("ema_tau_end must be >= ema_tau_start")
        if not (0.0 <= self.ema_tau_min < 1.0):
            raise ValueError("ema_tau_min must be in [0,1)")
        if self.ema_adapt_strength < 0:
            raise ValueError("ema_adapt_strength must be non-negative")
        if not (-1.0 <= self.ema_cosine_target <= 1.0):
            raise ValueError("ema_cosine_target must be in [-1,1]")
        for name in ["variance_weight", "covariance_weight", "decoder_weight"]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.decoder_hidden_mult <= 0:
            raise ValueError("decoder_hidden_mult must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, cfg: TextJEPAConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = ((hidden + cfg.ffn_hidden_multiple_of - 1) // cfg.ffn_hidden_multiple_of) * cfg.ffn_hidden_multiple_of
        self.hidden_dim = hidden
        self.fused = cfg.fused_swiglu
        if self.fused:
            self.w12 = nn.Linear(cfg.n_embd, 2 * hidden, bias=cfg.bias)
        else:
            self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.w2 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.w3.RESIDUAL_SCALE_INIT = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            a, b = torch.split(self.w12(x), self.hidden_dim, dim=-1)
            h = F.silu(a) * b
        else:
            h = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(h)


class SelfAttention(nn.Module):
    def __init__(self, cfg: TextJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.out.RESIDUAL_SCALE_INIT = True
        self.drop = nn.Dropout(cfg.dropout)
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        self.register_buffer("rope_inv_freq", inv, persistent=False)

    @staticmethod
    def rotate_pairwise(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        xr, xi = x.chunk(2, dim=-1)
        return torch.cat((xr * cos - xi * sin, xr * sin + xi * cos), dim=-1)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.use_rope:
            return q, k
        pos = torch.arange(T, device=q.device, dtype=self.rope_inv_freq.dtype)
        ang = torch.einsum("t,d->td", pos, self.rope_inv_freq)
        cos, sin = torch.cos(ang)[None, None].to(q.dtype), torch.sin(ang)[None, None].to(q.dtype)
        return self.rotate_pairwise(q, cos, sin), self.rotate_pairwise(k, cos, sin)

    def forward(self, x: torch.Tensor, key_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.apply_rope(q, k, T)
        if self.cfg.attention_backend == "sdpa" and key_valid_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=not self.cfg.bidirectional_context)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if not self.cfg.bidirectional_context:
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                att = att.masked_fill(~causal[None, None], torch.finfo(att.dtype).min)
            if key_valid_mask is not None:
                att = att.masked_fill(~key_valid_mask[:, None, None, :], torch.finfo(att.dtype).min)
            att = F.softmax(att.float() - att.float().amax(dim=-1, keepdim=True), dim=-1).to(q.dtype)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.out(y))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: TextJEPAConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, key_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_valid_mask=key_valid_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TokenEncoder(nn.Module):
    def __init__(self, cfg: TextJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.padded_vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = self.cfg.init_std
            if self.cfg.residual_init_scale and hasattr(module, "RESIDUAL_SCALE_INIT"):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(module.weight, 0.0, std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, self.cfg.init_std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, ids: torch.Tensor, key_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = ids.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=ids.device, dtype=torch.long)
        x = self.wte(ids) + self.wpe(pos)[None]
        for block in self.blocks:
            x = block(x, key_valid_mask=key_valid_mask)
        return self.ln_f(x)


class Predictor(nn.Module):
    """Latent predictor with optional iterative refinement.

    The refinement loop is the model's lightweight latent "thinking" mechanism:
    after producing an initial predicted target latent, it can repeatedly update
    that latent using a small residual MLP. This adds compute only in the small
    predictor, not in the expensive encoder/target encoder.
    """

    def __init__(self, cfg: TextJEPAConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.n_embd
        hidden = cfg.predictor_mult * d
        layers = []
        cur = d
        for _ in range(cfg.predictor_depth - 1):
            layers += [nn.Linear(cur, hidden, bias=False), nn.SiLU()]
            cur = hidden
        layers.append(nn.Linear(cur, d, bias=False))
        self.init_net = nn.Sequential(*layers)
        self.refiner = nn.Sequential(
            RMSNorm(d, cfg.norm_eps),
            nn.Linear(d, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, d, bias=False),
        ) if cfg.predictor_refine_steps > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.init_net(x)
        if self.refiner is not None:
            for _ in range(self.cfg.predictor_refine_steps - 1):
                z = z + float(self.cfg.predictor_residual_scale) * self.refiner(z)
        return z


class LatentTokenDecoder(nn.Module):
    def __init__(self, cfg: TextJEPAConfig, tied_embedding: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.tied_embedding = tied_embedding
        hidden = cfg.decoder_hidden_mult * cfg.n_embd
        self.proj = nn.Identity() if hidden == cfg.n_embd else nn.Sequential(
            nn.Linear(cfg.n_embd, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, cfg.n_embd, bias=False)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = F.linear(self.proj(z), self.tied_embedding.weight)
        if self.cfg.padded_vocab_size != self.cfg.vocab_size:
            logits = logits.float()
            logits[..., self.cfg.vocab_size :] = torch.finfo(logits.dtype).min
        return logits


class TextSpanJEPA(nn.Module):
    def __init__(self, cfg: TextJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.online = TokenEncoder(cfg)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.span_predictor = Predictor(cfg)
        self.span_predictor.apply(self.online.init_weights)
        if cfg.separate_future_predictor:
            self.future_predictor = Predictor(cfg)
            self.future_predictor.apply(self.online.init_weights)
        else:
            self.future_predictor = self.span_predictor
        # Backward-compatible alias for older scripts/checkpoints.
        self.predictor = self.span_predictor

        if cfg.position_aware_predictor:
            self.span_query = nn.Parameter(torch.zeros(cfg.n_embd))
            self.future_query = nn.Parameter(torch.zeros(len(cfg.future_offsets), cfg.n_embd))
        else:
            self.span_query = None
            self.future_query = None

        self.decoder = LatentTokenDecoder(cfg, self.online.wte) if cfg.use_decoder and cfg.decoder_weight > 0 else None
        self.last_metrics: Dict[str, torch.Tensor] = {}
        self.current_mask_ratio = float(cfg.mask_ratio_start if cfg.mask_ratio_start is not None else cfg.mask_ratio)
        self.register_buffer("target_center", torch.zeros(cfg.n_embd), persistent=True)
        self.register_buffer("last_ema_tau", torch.tensor(float(cfg.ema_tau_start)), persistent=True)
        self.register_buffer("last_online_target_cosine", torch.tensor(0.0), persistent=True)
        self.register_buffer("last_online_target_mse", torch.tensor(0.0), persistent=True)

    @torch.no_grad()
    def update_target_center(self, target: torch.Tensor) -> None:
        if not self.cfg.center_targets:
            return
        mean = target.detach().float().mean(dim=0)
        self.target_center.mul_(self.cfg.target_center_momentum).add_(mean, alpha=1.0 - self.cfg.target_center_momentum)

    @torch.no_grad()
    def scheduled_tau(self, step: int, total_steps: int) -> float:
        """Cosine EMA schedule before small health-based adaptation."""
        progress = min(max(step / max(1, total_steps), 0.0), 1.0)
        return float(
            self.cfg.ema_tau_end
            - (self.cfg.ema_tau_end - self.cfg.ema_tau_start) * (0.5 * (1.0 + math.cos(math.pi * progress)))
        )

    @torch.no_grad()
    def effective_tau(self, step: int, total_steps: int, online_target_cosine: Optional[torch.Tensor] = None) -> float:
        """Mild adaptive EMA tau.

        If online and target representations are poorly aligned, the target should
        follow the online encoder a little faster (smaller tau). The adaptation is
        deliberately tiny to preserve stability and auditability.
        """
        tau = self.scheduled_tau(step, total_steps)
        if self.cfg.adaptive_ema_tau and online_target_cosine is not None:
            cos = float(online_target_cosine.detach().float().cpu().item())
            deficit = max(0.0, float(self.cfg.ema_cosine_target) - cos)
            tau = tau - float(self.cfg.ema_adapt_strength) * deficit
        tau = max(float(self.cfg.ema_tau_min), min(float(self.cfg.ema_tau_end), tau))
        return float(tau)

    @torch.no_grad()
    def update_target_ema(self, step: int, total_steps: int, online_target_cosine: Optional[torch.Tensor] = None) -> float:
        tau = self.effective_tau(step, total_steps, online_target_cosine=online_target_cosine)
        for pt, po in zip(self.target.parameters(), self.online.parameters()):
            pt.data.mul_(tau).add_(po.data, alpha=1.0 - tau)
        self.last_ema_tau.fill_(tau)
        return tau

    def set_training_progress(self, step: int, total_steps: int) -> None:
        if self.cfg.mask_ratio_start is None or self.cfg.mask_curriculum_steps <= 0:
            self.current_mask_ratio = float(self.cfg.mask_ratio)
            return
        p = min(max(step / max(1, self.cfg.mask_curriculum_steps), 0.0), 1.0)
        self.current_mask_ratio = float(self.cfg.mask_ratio_start + p * (self.cfg.mask_ratio - self.cfg.mask_ratio_start))

    def build_target_input(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.target_view_mode == "full":
            return ids
        mask_id = torch.full_like(ids, int(self.cfg.mask_token_id))
        if self.cfg.target_view_mode == "target_only":
            return torch.where(mask, ids, mask_id)
        # local_window: keep target tokens plus local neighborhood around targets.
        B, T = ids.shape
        keep = mask.clone()
        w = int(self.cfg.target_local_window)
        if w > 0:
            for shift in range(1, w + 1):
                keep[:, shift:] |= mask[:, :-shift]
                keep[:, :-shift] |= mask[:, shift:]
        return torch.where(keep, ids, mask_id)

    @torch.no_grad()
    def make_span_mask(self, ids: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        B, T = ids.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=ids.device)
        target_n = max(int(round(T * float(getattr(self, "current_mask_ratio", self.cfg.mask_ratio)))), self.cfg.min_masked_tokens)
        n_spans = max(1, math.ceil(target_n / self.cfg.span_length))
        for b in range(B):
            starts = torch.randint(0, max(1, T - self.cfg.span_length + 1), (n_spans,), device=ids.device, generator=generator)
            for st in starts.tolist():
                mask[b, st : min(T, st + self.cfg.span_length)] = True
            if int(mask[b].sum().item()) < target_n:
                deficit = target_n - int(mask[b].sum().item())
                fill = (~mask[b]).nonzero(as_tuple=False).flatten()[:deficit]
                mask[b, fill] = True
        return mask

    def corrupt(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, torch.full_like(ids, int(self.cfg.mask_token_id)), ids)

    def representation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cfg.normalize_targets:
            pred = F.normalize(pred.float(), dim=-1)
            target = F.normalize(target.float(), dim=-1)
        else:
            pred, target = pred.float(), target.float()
        if self.cfg.pred_loss == "mse":
            return F.mse_loss(pred, target)
        return (1.0 - (pred * target).sum(dim=-1)).mean()

    def future_latent_loss(self, online_h: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_future_prediction or self.cfg.future_weight <= 0:
            return online_h.new_zeros(())
        losses = []
        for i, off in enumerate(self.cfg.future_offsets):
            src = online_h[:, :-off, :]
            if self.future_query is not None:
                src = src + self.future_query[i].to(src.dtype)[None, None, :]
            pred = self.future_predictor(src)
            tgt = target_h.detach()[:, off:, :]
            losses.append(self.representation_loss(pred, tgt))
        return torch.stack(losses).mean() if losses else online_h.new_zeros(())

    def variance_loss(self, h: torch.Tensor) -> torch.Tensor:
        if self.cfg.variance_weight <= 0:
            return h.new_zeros(())
        z = h.float().reshape(-1, h.size(-1))
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(self.cfg.variance_target_std - std).mean()

    def covariance_loss(self, h: torch.Tensor) -> torch.Tensor:
        if self.cfg.covariance_weight <= 0:
            return h.new_zeros(())
        z = h.float().reshape(-1, h.size(-1))
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / max(1, z.size(0) - 1)
        off = cov - torch.diag(torch.diag(cov))
        return off.pow(2).sum() / z.size(-1)

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            mask = self.make_span_mask(ids)
        if not mask.any():
            raise RuntimeError("mask contains no target positions")
        context_ids = self.corrupt(ids, mask)
        online_h = self.online(context_ids)
        with torch.no_grad():
            target_ids = self.build_target_input(ids, mask)
        target_h = self.target(target_ids)
        span_input = online_h + self.span_query.to(online_h.dtype)[None, None, :] if self.span_query is not None else online_h
        pred_h = self.span_predictor(span_input)
        pred = pred_h[mask]
        target = target_h.detach()[mask]
        self.update_target_center(target)
        if self.cfg.center_targets:
            target_for_loss = target - self.target_center.to(target.device, dtype=target.dtype)
        else:
            target_for_loss = target
        pred_loss = self.representation_loss(pred, target_for_loss)
        future_loss = self.future_latent_loss(pred_h, target_h)
        with torch.no_grad():
            online_masked = online_h[mask].detach().float()
            target_masked = target.detach().float()
            online_norm = F.normalize(online_masked, dim=-1)
            target_norm = F.normalize(target_masked, dim=-1)
            online_target_cosine = (online_norm * target_norm).sum(dim=-1).mean()
            online_target_mse = F.mse_loss(online_norm, target_norm)
            self.last_online_target_cosine.copy_(online_target_cosine.detach())
            self.last_online_target_mse.copy_(online_target_mse.detach())
        dec_loss = ids.new_tensor(0.0, dtype=torch.float32)
        dec_acc = ids.new_tensor(0.0, dtype=torch.float32)
        if self.decoder is not None and self.cfg.decoder_weight > 0:
            logits = self.decoder(pred)
            true = ids[mask]
            dec_loss = F.cross_entropy(logits, true)
            with torch.no_grad():
                dec_acc = (logits.argmax(dim=-1) == true).float().mean()
        vloss = self.variance_loss(online_h)
        closs = self.covariance_loss(online_h)
        loss = (
            pred_loss
            + self.cfg.future_weight * future_loss
            + self.cfg.decoder_weight * dec_loss
            + self.cfg.variance_weight * vloss
            + self.cfg.covariance_weight * closs
        )
        with torch.no_grad():
            online_std = online_h.float().reshape(-1, online_h.size(-1)).std(dim=0, unbiased=False).mean()
            target_std = target_h.float().reshape(-1, target_h.size(-1)).std(dim=0, unbiased=False).mean()
            mask_fraction = mask.float().mean()
        self.last_metrics = {
            "loss": loss.detach(),
            "pred_loss": pred_loss.detach(),
            "future_loss": future_loss.detach(),
            "decoder_loss": dec_loss.detach(),
            "decoder_accuracy": dec_acc.detach(),
            "variance_loss": vloss.detach(),
            "covariance_loss": closs.detach(),
            "online_std": online_std.detach(),
            "target_std": target_std.detach(),
            "target_center_norm": self.target_center.detach().float().norm(),
            "online_target_cosine": online_target_cosine.detach(),
            "online_target_mse": online_target_mse.detach(),
            "ema_tau": self.last_ema_tau.detach(),
            "mask_fraction": mask_fraction.detach(),
        }
        return {"loss": loss, **self.last_metrics}

    def parameter_count(self) -> Dict[str, int]:
        return {
            "online": sum(p.numel() for p in self.online.parameters()),
            "target_frozen": sum(p.numel() for p in self.target.parameters()),
            "predictor": sum(p.numel() for p in self.predictor.parameters()),
            "decoder": 0 if self.decoder is None else sum(p.numel() for p in self.decoder.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_stored": sum(p.numel() for p in self.parameters()),
        }


__all__ = ["TextJEPAConfig", "TextSpanJEPA", "TokenEncoder", "SelfAttention", "RMSNorm", "SwiGLU"]
