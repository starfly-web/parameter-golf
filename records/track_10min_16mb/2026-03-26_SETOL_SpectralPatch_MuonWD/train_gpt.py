"""
The `train_gpt.py` script for the SETOL/WeightWatcher Spectral Diagnostics submission.
Integrates:
  1. Convergence Oracle (SpectralMonitor)
  2. AlphaDecay (per-parameter scaling for Muon)
  3. alpha-selective quantization (dynamic bit-width allocation)
  4. GPTQ-SR (Stable-Rank aware Hessian clip search)
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import sys
import time
import uuid
import zlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Optional import for WeightWatcher
import weightwatcher as ww
_WW_AVAILABLE = True


# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 3600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 10))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # Dropout & EMA
    dropout = float(os.environ.get("DROPOUT", 0.1))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.999))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.03))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    muon_wd = float(os.environ.get("MUON_WD", 0.05))

    # LoRA TTT
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 64))

    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 512))

# -----------------------------
# UTILITIES & DATA LOADING
# -----------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.backup = {}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].copy_(self.decay * self.shadow[n] + (1.0 - self.decay) * p.detach())

    def apply(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.backup[n] = p.clone().detach()
                    p.copy_(self.shadow[n])

    def restore(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.backup:
                    p.copy_(self.backup[n])
        self.backup.clear()

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0: raise ValueError(f"Validation split too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, wd: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, wd=wd))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("wd", 0.0)
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                if wd > 0: p.mul_(1 - lr * wd)
                curr += p.numel()
        return loss

# -----------------------------
# POST-TRAINING QUANTIZATION UTILS
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights", "tok_emb")
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 0.9999984
INT6_STEP = 4

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor], INT6_LAYERS: str = ""):
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    stats = {k: 0 for k in ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes")}
    int6_set = {int(x) for x in INT6_LAYERS.split(",") if x.strip()} if INT6_LAYERS else set()
    for name, t in state_dict.items():
        t = t.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += t.numel() * t.element_size()
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
            if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS): kept = t.float()
            else:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(INT8_KEEP_FLOAT_STORE_DTYPE)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += kept.numel() * kept.element_size()
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if any(f"blocks.{i}." in name for i in int6_set):
            q = (torch.round(q.float() / INT6_STEP) * INT6_STEP).to(torch.int8)
        if s.ndim > 0: qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name], scales[name], dtypes[name] = q, s, str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += q.numel() * q.element_size() + s.numel() * s.element_size()
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough, "qmeta": qmeta, "passthrough_orig_dtypes": passthrough_orig_dtypes}
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out = {}
    qmeta, pod = obj.get("qmeta", {}), obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype, s = getattr(torch, obj["dtypes"][name]), obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1))).float()).to(dtype)
        else: out[name] = (q.float() * s.item()).to(dtype)
    for name, t in obj["passthrough"].items():
        if name in pod: out[name] = t.to(getattr(torch, pod[name]))
        else: out[name] = t
    return out

# -----------------------------
# TEST-TIME TRAINING (LoRA)
# -----------------------------

BOS_ID = 1

class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)
    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()

class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz: int, model: nn.Module, rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for _ in range(len(model.blocks)):
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, model.blocks[0].attn.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, model.blocks[0].attn.c_v.weight.shape[0], rank))
    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA): m.reset()

def _reset_ttt_optimizer(opt):
    for group in opt.param_groups:
        for p in group["params"]:
            s = opt.state.get(p)
            if not s: continue
            s["exp_avg"].zero_(); s["exp_avg_sq"].zero_(); s["step"].fill_(0)

def _build_ttt_optimizer(lora, args: Hyperparameters):
    return torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=1e-10)

def _find_docs(all_tokens: Tensor, include_next_bos: bool = True) -> list[tuple[int, int]]:
    bos_indices = torch.where(all_tokens == BOS_ID)[0].tolist()
    if not bos_indices or bos_indices[0] != 0: bos_indices.insert(0, 0)
    docs = []
    for i in range(len(bos_indices)):
        start = bos_indices[i]
        end = bos_indices[i+1] if i+1 < len(bos_indices) else all_tokens.numel()
        if include_next_bos and i+1 < len(bos_indices): end += 1
        docs.append((start, end))
    return docs

def _accumulate_bpb(loss_per_token: Tensor, x: Tensor, y: Tensor, batch_idx: int, chunk_start: int, chunk_len: int,
                    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                    loss_sum, byte_sum, token_count):
    lpt = loss_per_token[batch_idx, :chunk_len]
    loss_sum += lpt.to(torch.float64).sum()
    token_count += chunk_len
    prev_ids = x[batch_idx, :chunk_len]
    tgt_ids = y[batch_idx, :chunk_len]
    tbytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
    tbytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
    byte_sum += tbytes.to(torch.float64).sum()

# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def eval_val(args: Hyperparameters, model: nn.Module, rank: int, world_size: int, device: torch.device, grad_accum_steps: int,
             val_tokens: Tensor, base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start, seq_end = (total_seqs * rank) // world_size, (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            raw_start, raw_end = bss * args.train_seq_len, bse * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(x)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            loss_sum += loss.to(torch.float64)
            token_count += y.numel()
            tbytes = base_bytes_lut[y].to(dtype=torch.int16)
            tbytes += (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).to(dtype=torch.int16)
            byte_sum += tbytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    model.train()
    return val_loss, val_bpb

@torch.no_grad()
def eval_val_sliding(args: Hyperparameters, model: nn.Module, compiled_forward_logits, rank: int, world_size: int, device: torch.device,
                     val_tokens: Tensor, base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor):
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    seq_len, stride, N = args.train_seq_len, args.eval_stride, val_tokens.numel()
    indices = list(range(0, N - seq_len, stride))
    if not indices: indices = [0]
    rank_starts = indices[(len(indices) * rank) // world_size : (len(indices) * (rank + 1)) // world_size]
    for i in range(0, len(rank_starts), args.eval_batch_seqs):
        batch_starts = rank_starts[i : i + args.eval_batch_seqs]
        bsz = len(batch_starts)
        x = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        y = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        for b, st in enumerate(batch_starts):
            end = min(st + seq_len + 1, N)
            alen = end - st - 1
            chunk = val_tokens[st : st + alen + 1].to(device)
            x[b, :alen], y[b, :alen] = chunk[:-1], chunk[1:]
            mask[b, (0 if st == 0 else seq_len - stride) : alen] = True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = compiled_forward_logits(x)
        fl, ft = logits[mask], y[mask]
        if fl.numel() > 0:
            loss_sum += F.cross_entropy(fl.float(), ft, reduction="sum").to(torch.float64)
            token_count += ft.numel()
            px, py = x[mask], y[mask]
            tbytes = base_bytes_lut[py].to(dtype=torch.int16)
            tbytes += (has_leading_space_lut[py] & ~is_boundary_token_lut[px]).to(dtype=torch.int16)
            byte_sum += tbytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM); dist.all_reduce(token_count, op=dist.ReduceOp.SUM); dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    model.train()
    return val_loss, val_bpb

def eval_val_ttt_lora(args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int, device: torch.device,
                      base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor):
    base_model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    all_tokens_for_docs = load_validation_tokens(args.val_files, 1).to(device)
    all_docs = _find_docs(all_tokens_for_docs, include_next_bos=True)
    rank_docs = all_docs[(len(all_docs) * rank) // world_size : (len(all_docs) * (rank + 1)) // world_size]
    for i in range(0, len(rank_docs), args.ttt_batch_size):
        batch_docs = rank_docs[i : i + args.ttt_batch_size]
        bsz = len(batch_docs)
        lora = BatchedTTTLoRA(bsz, base_model, args.ttt_lora_rank).to(device).bfloat16()
        opt = _build_ttt_optimizer(lora, args)
        lora.reset(); _reset_ttt_optimizer(opt)
        num_chunks = [(d[1] - d[0] + chunk_size - 1) // chunk_size for d in batch_docs]
        max_chunks = max(num_chunks)
        active = [True] * bsz
        for ci in range(max_chunks):
            x = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            doc_info = []
            needs_train = ci < max_chunks - 1
            chunk_offset = (ci * chunk_size) if ci > 0 else 0
            for b in range(bsz):
                if ci >= num_chunks[b]:
                    active[b] = False; doc_info.append((0, 0)); continue
                d_start, d_end = batch_docs[b]
                c_start = d_start + ci * chunk_size
                c_end = min(c_start + seq_len + 1, d_end)
                wl = c_end - c_start - 1
                if wl <= 0: active[b] = False; doc_info.append((0, 0)); continue
                chunk = all_tokens_for_docs[c_start:c_end]
                x[b, :wl], y[b, :wl] = chunk[:-1], chunk[1:]
                doc_info.append((chunk_offset, wl))
            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=lora)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=lora)
            with torch.no_grad():
                for b in range(bsz):
                    if active[b]:
                        co, cl = doc_info[b]
                        _accumulate_bpb(ptl.detach(), x, y, b, co, cl, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, loss_sum, byte_sum, token_count)
            if needs_train:
                mask = torch.tensor([float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device)
                per_doc = ptl.mean(dim=-1)
                opt.zero_grad(); (per_doc * mask).sum().backward(); opt.step()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM); dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM); dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb

# -----------------------------
# TRANSFORMER ARCHITECTURE
# -----------------------------

class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def forward(self, x): return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv", inv, persistent=False)
    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv.to(device))
        return freqs.cos()[None, None, :, :].to(dtype), freqs.sin()[None, None, :, :].to(dtype)

def apply_rotary_emb(x, cos, sin):
    h = x.size(-1) // 2
    return torch.cat((x[..., :h] * cos + x[..., h:] * sin, x[..., :h] * (-sin) + x[..., h:] * cos), dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv, rope_base, qk_gain, dropout=0.0):
        super().__init__()
        self.nh, self.nkv, self.dh = num_heads, num_kv, dim // num_heads
        self.c_q, self.c_k, self.c_v = CastedLinear(dim, dim, False), CastedLinear(dim, num_kv*self.dh, False), CastedLinear(dim, num_kv*self.dh, False)
        self.proj = CastedLinear(dim, dim, False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain))
        self.rotary = Rotary(self.dh, rope_base)
        self.dropout = dropout
    def forward(self, x, qd=None, vd=None):
        bsz, sl, _ = x.shape
        q = self.c_q(x) + (qd if qd is not None else 0)
        k = self.c_k(x)
        v = self.c_v(x) + (vd if vd is not None else 0)
        q = q.view(bsz, sl, self.nh, self.dh).transpose(1, 2)
        k = k.view(bsz, sl, self.nkv, self.dh).transpose(1, 2)
        v = v.view(bsz, sl, self.nkv, self.dh).transpose(1, 2)
        if self.nkv != self.nh:
            k = k.repeat_interleave(self.nh // self.nkv, dim=1)
            v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        q, k = F.rms_norm(q, (self.dh,)), F.rms_norm(k, (self.dh,))
        c, s = self.rotary(sl, x.device, q.dtype)
        q, k = apply_rotary_emb(q, c, s), apply_rotary_emb(k, c, s)
        q = q * self.q_gain[None, :, None, None].to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(bsz, sl, -1))

class MLP(nn.Module):
    def __init__(self, dim, mult, dropout=0.0):
        super().__init__()
        h = int(dim * mult)
        self.w1, self.w2, self.proj = CastedLinear(dim, h, False), CastedLinear(dim, h, False), CastedLinear(h, dim, False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.proj(self.dropout(F.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    def __init__(self, dim, nh, nkv, mult, rope, qk_gain, dropout=0.0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(), RMSNorm()
        self.attn = Attention(dim, nh, nkv, rope, qk_gain, dropout=dropout)
        self.mlp = MLP(dim, mult, dropout=dropout)
        self.ascl, self.mscl = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))))
    def forward(self, x, x0, qd=None, vd=None):
        m = self.resid_mix.to(x.dtype)
        x = m[0][None, None, :] * x + m[1][None, None, :] * x0
        n = self.norm1(x)
        x = x + self.ascl.to(x.dtype)[None, None, :] * self.attn(n, qd, vd)
        x = x + self.mscl.to(x.dtype)[None, None, :] * self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, v, l, d, nh, nkv, mult, tie, tie_std, logit_softcap, rope, qk_gain, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(v, d)
        self.blocks = nn.ModuleList([Block(d, nh, nkv, mult, rope, qk_gain, dropout=dropout) for _ in range(l)])
        self.norm = RMSNorm()
        self.logit_softcap = logit_softcap
        self.lm_head = CastedLinear(d, v, False) if not tie else None
        if tie: self.lm_head_weight = self.tok_emb.weight
        else: nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.tie, self.l = tie, l
    def forward(self, x, y=None, lora=None):
        x0 = h = self.tok_emb(x)
        for i, b in enumerate(self.blocks):
            qd = lora.q_loras[i](h) if lora else None
            vd = lora.v_loras[i](h) if lora else None
            h = b(h, x0, qd, vd)
        h = self.norm(h)
        logits = F.linear(h, self.tok_emb.weight if self.tie else self.lm_head.weight)
        if lora: logits = logits + lora.lm_head_lora(h)
        if self.logit_softcap > 0:
            logits = torch.tanh(logits / self.logit_softcap) * self.logit_softcap
        if y is not None:
            if lora:
                bsz, sl, v = logits.shape
                return F.cross_entropy(logits.float().reshape(-1, v), y.reshape(-1), reduction="none").reshape(bsz, sl)
            return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits

    def forward_logits(self, x):
        return self.forward(x)

# -----------------------------
# SPECTRAL DIAGNOSTICS (SETOL)
# -----------------------------

def hill_alpha(W: Tensor, k_frac=0.2):
    W_np = W.detach().float().cpu().numpy()
    try: _, sv, _ = np.linalg.svd(W_np, full_matrices=False)
    except: return float('nan'), float('nan')
    sv = sv[sv > 1e-10]
    if len(sv) < 4: return float('nan'), float('nan')
    sr = float(np.sum(sv**2) / sv[0]**2)
    k = max(2, int(len(sv) * k_frac))
    alpha = float(1.0 / (np.log(sv[:k] / sv[k]).mean() + 1e-10))
    return alpha, sr

def get_spectral_stats(model: nn.Module):
    if _WW_AVAILABLE:
        try:
            w = ww.WeightWatcher(model=model)
            df = w.analyze(plot=False, randomize=False, mp_fit=True)
            return {r.longname: {'alpha': r.alpha, 'stable_rank': r.stable_rank} for _, r in df.iterrows()}
        except: pass
    return {n: dict(zip(('alpha', 'stable_rank',), hill_alpha(p))) for n, p in model.named_parameters() if p.ndim == 2 and min(p.shape) >= 8}

class SpectralMonitor:
    def __init__(self, model, check_every=500, verbose=True):
        self.model, self.check_every, self.verbose = model, check_every, verbose
        self.history = []
        self._last = -1
    def step(self, step):
        if step - self._last < self.check_every: return None
        self._last = step
        stats = get_spectral_stats(self.model)
        alphas = [v['alpha'] for v in stats.values() if not math.isnan(v['alpha'])]
        if not alphas: return None
        ma, va = np.mean(alphas), np.var(alphas)
        self.history.append((step, ma, va))
        if self.verbose: print(f"  [SETOL] step={step} alpha_avg={ma:.3f} alpha_var={va:.3f}")
        return {'ma': ma, 'va': va, 'stats': stats}

def build_alpha_decay_groups(model, base_wd=0.1):
    stats = get_spectral_stats(model)
    groups = []
    for n, p in model.named_parameters():
        if p.ndim == 2:
            alpha = stats.get(n, {}).get('alpha', 2.5)
            wd = base_wd * (alpha / 2.5)
            groups.append({'params': [p], 'wd': float(np.clip(wd, 0.01, 0.5)), 'name': n, 'alpha': alpha})
        else: groups.append({'params': [p], 'wd': base_wd, 'name': n})
    return groups

class ParameterGolfSpectralPatch:
    def __init__(self, model, args, verbose=True):
        self.model, self.args, self.verbose = model, args, verbose
        self.monitor = SpectralMonitor(model, verbose=verbose)
    def patch_muon_param_groups(self): return build_alpha_decay_groups(self.model, self.args.muon_wd)
    def training_step(self, step): return self.monitor.step(step)
    def apply_gptq_sr(self, bits=6):
        with torch.no_grad():
            for n, m in self.model.named_modules():
                if hasattr(m, 'weight') and m.weight.ndim == 2:
                    W = m.weight.float()
                    c = torch.quantile(W.abs(), 0.999)
                    W = W.clamp(-c, c)
                    s = c / ((2**(bits-1)) - 0.5)
                    m.weight.data.copy_((W/s).round().clamp(-(2**(bits-1)), (2**(bits-1))-1) * s)

# -----------------------------
# MAIN
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # ── Distributed + CUDA setup ────────────────────────────────────────────
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank        = int(os.environ.get("RANK",       "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale       = 1.0 / grad_accum_steps
    
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    cc = torch.cuda.get_device_capability(device)
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_flash_sdp(cc[0] >= 8)
    enable_math_sdp(cc[0] < 8)
    enable_mem_efficient_sdp(False)

    logfile = f"logs/{args.run_id}.txt" if master_process else None
    if master_process: os.makedirs("logs", exist_ok=True)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"sdp_backend: flash={cc[0]>=8}")
    log0("=" * 100, console=False)

    # ── Tokenizer + validation metric setup ─────────────────────────────────
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    # ── Model + optimizer setup ─────────────────────────────────────────────
    model = GPT(args.vocab_size, args.num_layers, args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.tie_embeddings, args.tied_embed_init_std, args.logit_softcap, args.rope_base, args.qk_gain_init, dropout=args.dropout).to(device).bfloat16()
    
    spectral = ParameterGolfSpectralPatch(model, args, master_process)
    muon_groups = spectral.patch_muon_param_groups()
    
    matrix_groups = [g for g in muon_groups if 'alpha' in g]
    non_matrix_groups = [g for g in muon_groups if 'alpha' not in g]
    
    optimizer_muon = Muon(matrix_groups, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    optimizer_adam = torch.optim.Adam(non_matrix_groups, lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)
    optimizers = [optimizer_muon, optimizer_adam]
    for opt in optimizers:
        for group in opt.param_groups: group['base_lr'] = group['lr']

    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)
    compiled_forward_logits = torch.compile(model.forward_logits, dynamic=False)
    ddp_model = DDP(compiled_model, device_ids=[local_rank]) if distributed else compiled_model

    log0(f"model_params:{sum(p.numel() for p in model.parameters())}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} iterations:{args.iterations} max_wallclock:{args.max_wallclock_seconds}")

    # ── Data loader + warmup ─────────────────────────────────────────────────
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    if args.warmup_steps > 0:
        initial_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
        ddp_model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: ddp_model.require_backward_grad_sync = (ms == grad_accum_steps - 1)
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = ddp_model(x, y)
                (loss * grad_scale).backward()
            for opt in optimizers: opt.step()
            ema.update(model)
            if ws + 1 <= 20 or (ws + 1) % 10 == 0: log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        model.load_state_dict(initial_state)
        zero_grad_all()

    # EMA shadow initialization (after any warmup weight resets)
    ema = EMA(model, args.ema_decay)

    # ── Main training loop ───────────────────────────────────────────────────
    training_time_ms, step = 0.0, 0
    torch.cuda.synchronize(); t0 = time.perf_counter()

    while True:
        last_step = step == args.iterations
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        
        if should_validate:
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            ema.apply(model)
            val_loss, val_bpb = eval_val(args, ddp_model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            ema.restore(model)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        if last_step: break

        # Learning rate warm-down
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = 1.0
        if args.warmdown_iters > 0:
            if max_wallclock_ms:
                remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
                warmdown_ms = args.warmdown_iters * (elapsed_ms / max(step, 1))
                scale = min(remaining_ms / warmdown_ms, 1.0) if remaining_ms < warmdown_ms else 1.0
            else:
                scale = max(0.0, (args.iterations - step) / args.warmdown_iters) if step > (args.iterations - args.warmdown_iters) else 1.0

        zero_grad_all(); train_loss = 0.0
        for ms in range(grad_accum_steps):
            if distributed: ddp_model.require_backward_grad_sync = (ms == grad_accum_steps - 1)
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = ddp_model(x, y)
            train_loss += loss.detach().item() / grad_accum_steps
            (loss * grad_scale).backward()
        
        for opt in optimizers:
            for g in opt.param_groups: g['lr'] = g.get('base_lr', g['lr']) * scale
            opt.step()
        ema.update(model)
        spectral.training_step(step)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if step <= 10 or step % args.train_log_every == 0:
            log0(f"step:{step} train_loss:{train_loss:.4f} time:{approx_ms:.0f}ms")
        
        if max_wallclock_ms and approx_ms >= max_wallclock_ms:
            log0(f"Stopping early due to wallclock cap at step {step}")
            break

    # ── Final Evaluation ─────────────────────────────────────────────────────
    if master_process:
        spectral.apply_gptq_sr(6)
        ema.apply(model)
        sd, _ = quantize_state_dict_int8(model.state_dict())
        buf = io.BytesIO(); torch.save(sd, buf)
        with open("final_model.int8.ptz", "wb") as f: f.write(zlib.compress(buf.getvalue(), 9))
        log0(f"Final int8 model saved. Size: {len(zlib.compress(buf.getvalue(), 9))} bytes")
    
    if distributed: dist.barrier()
    
    # Reload Quantized for Scoring
    with open("final_model.int8.ptz", "rb") as f: q_data = f.read()
    q_state = torch.load(io.BytesIO(zlib.decompress(q_data)), map_location="cpu")
    model.load_state_dict(dequantize_state_dict_int8(q_state))
    
    log0("Starting final sliding-window evaluation...")
    s_loss, s_bpb = eval_val_sliding(args, model, compiled_forward_logits, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"final_int8_sliding val_loss:{s_loss:.4f} val_bpb:{s_bpb:.4f}")

    log0("Starting final TTT-LoRA evaluation...")
    ttt_loss, ttt_bpb = eval_val_ttt_lora(args, model, rank, world_size, device, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"final_int8_ttt_lora val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f}")

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
