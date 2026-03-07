"""A100/H100-optimized model setup for NST-VERI.

Creates models with:
  - DeBERTa-v3-large (or base for smaller GPUs) backbone
  - LoRA adapters for parameter-efficient fine-tuning
  - Gradient checkpointing for memory efficiency
  - SDPA (Flash Attention) where available
  - Fused AdamW optimizer for GPU speedup
  - Proper parameter group separation (backbone vs. heads vs. LoRA)

Tier-based auto-configuration:
  H100/A100-80GB: large + LoRA r=16, bs=128, seq=384
  A100-40GB:      large + LoRA r=16, bs=64,  seq=384
  L4-24GB:        large + LoRA r=8,  bs=32,  seq=320
  T4-16GB:        base  + LoRA r=8,  bs=16,  seq=256
  MPS/CPU:        base  + full FT,   bs=8,   seq=256
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    DebertaV2Tokenizer,
)

from data.fever_dataset import NUM_LABELS, ID2LABEL, LABEL2ID

logger = logging.getLogger("model_setup")


def setup_gpu_optimizations(device: str = "cuda") -> dict:
    """Configure PyTorch for maximum GPU throughput.

    Returns dict of what was enabled.
    """
    enabled = {}

    if device != "cuda" or not torch.cuda.is_available():
        return enabled

    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)

    # TF32 for Ampere+ (≥sm80): free ~3× matmul speedup
    if cc >= (8, 0):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        enabled["tf32"] = True
        logger.info("TF32 enabled for matmul and cuDNN")

    # cuDNN benchmark: auto-tune conv algorithms
    torch.backends.cudnn.benchmark = True
    enabled["cudnn_benchmark"] = True

    # SDPA (Scaled Dot-Product Attention) environment hint
    os.environ.setdefault("ATTN_IMPL", "sdpa")
    enabled["sdpa_hint"] = True

    return enabled


def detect_gpu_tier() -> dict:
    """Auto-detect GPU and return tier-based configuration.

    Returns a config dict suitable for merging into training config.
    """
    if not torch.cuda.is_available():
        # CPU / MPS fallback
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return {
            "tier": "MPS" if has_mps else "CPU",
            "device": "mps" if has_mps else "cpu",
            "model_name": "microsoft/deberta-v3-base",
            "use_lora": False,
            "lora_rank": 0,
            "batch_size": 8,
            "grad_accum_steps": 4,
            "max_length": 256,
            "num_workers": 0,
            "gradient_checkpointing": False,
            "bf16": False,
            "fp16": False,
            "torch_compile": False,
            "fused_optimizer": False,
            "epochs": 3,
        }

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    cc = (props.major, props.minor)
    supports_bf16 = cc >= (8, 0)

    if vram_gb >= 70:
        tier = "H100/A100-80GB"
        cfg = {
            "model_name": "microsoft/deberta-v3-large",
            "use_lora": True, "lora_rank": 16, "lora_alpha": 32,
            "batch_size": 128, "grad_accum_steps": 2,
            "max_length": 384, "num_workers": 4,
            "gradient_checkpointing": True,
            "compile_mode": "reduce-overhead",
            "epochs": 5,
        }
    elif vram_gb >= 35:
        tier = "A100-40GB"
        cfg = {
            "model_name": "microsoft/deberta-v3-large",
            "use_lora": True, "lora_rank": 16, "lora_alpha": 32,
            "batch_size": 64, "grad_accum_steps": 2,
            "max_length": 384, "num_workers": 4,
            "gradient_checkpointing": True,
            "compile_mode": "reduce-overhead",
            "epochs": 5,
        }
    elif vram_gb >= 20:
        tier = "L4-24GB"
        cfg = {
            "model_name": "microsoft/deberta-v3-large",
            "use_lora": True, "lora_rank": 8, "lora_alpha": 16,
            "batch_size": 32, "grad_accum_steps": 4,
            "max_length": 320, "num_workers": 2,
            "gradient_checkpointing": True,
            "compile_mode": "default",
            "epochs": 4,
        }
    elif vram_gb >= 14:
        tier = "T4-16GB"
        cfg = {
            "model_name": "microsoft/deberta-v3-base",
            "use_lora": True, "lora_rank": 8, "lora_alpha": 16,
            "batch_size": 16, "grad_accum_steps": 4,
            "max_length": 256, "num_workers": 2,
            "gradient_checkpointing": True,
            "compile_mode": None,
            "epochs": 3,
        }
    else:
        tier = f"Small-GPU-{vram_gb:.0f}GB"
        cfg = {
            "model_name": "microsoft/deberta-v3-base",
            "use_lora": False, "lora_rank": 0,
            "batch_size": 8, "grad_accum_steps": 4,
            "max_length": 256, "num_workers": 0,
            "gradient_checkpointing": True,
            "compile_mode": None,
            "epochs": 3,
        }

    cfg["tier"] = tier
    cfg["device"] = "cuda"
    cfg["bf16"] = supports_bf16
    cfg["fp16"] = not supports_bf16
    cfg["torch_compile"] = cfg.get("compile_mode") is not None
    cfg["fused_optimizer"] = True

    gpu_name = torch.cuda.get_device_name(0)
    eff_bs = cfg["batch_size"] * cfg["grad_accum_steps"]
    logger.info(
        f"GPU: {gpu_name} ({vram_gb:.1f} GB) → Tier: {tier}\n"
        f"  Model: {cfg['model_name']}, LoRA: r={cfg.get('lora_rank', 0)}\n"
        f"  Batch: {cfg['batch_size']}×{cfg['grad_accum_steps']}={eff_bs} eff\n"
        f"  Precision: {'BF16' if supports_bf16 else 'FP16'}\n"
        f"  Compile: {cfg.get('compile_mode', 'OFF')}"
    )

    return cfg


def build_tokenizer(model_name: str) -> AutoTokenizer:
    """Build tokenizer, handling DeBERTa-v3 slow tokenizer requirement."""
    if "deberta-v3" in model_name.lower():
        return DebertaV2Tokenizer.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)


def build_nst_veri_model(
    model_name: str = "microsoft/deberta-v3-large",
    num_labels: int = NUM_LABELS,
    label_smoothing: float = 0.05,
    dropout: float = 0.1,
    n_constraints: int = 6,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
    class_weights: Optional[torch.Tensor] = None,
    contrastive_temperature: float = 0.07,
    pretrained_path: Optional[str] = None,
) -> tuple[AutoTokenizer, "NSTVeriModel"]:
    """Build the complete NST-VERI model with all components.

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output classes.
        label_smoothing: Label smoothing factor.
        dropout: Dropout for all auxiliary heads.
        n_constraints: Number of constraint types (matches ConstraintEngineV2).
        use_lora: Whether to use LoRA adapters.
        lora_rank: LoRA rank (r parameter).
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout.
        gradient_checkpointing: Enable gradient checkpointing.
        class_weights: Per-class loss weights.
        contrastive_temperature: Temperature for contrastive head.
        pretrained_path: Optional path to a pre-fine-tuned model (e.g., MNLI).

    Returns:
        (tokenizer, nst_veri_model) tuple.
    """
    from models.nst_veri import NSTVeriModel

    logger.info(f"Building NST-VERI model: {model_name}")

    # Tokenizer
    tokenizer = build_tokenizer(model_name)

    # Config
    load_from = pretrained_path or model_name
    config = AutoConfig.from_pretrained(
        load_from,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=dropout,
    )

    # Backbone
    backbone_kwargs: dict[str, Any] = {
        "config": config,
    }

    # Try to use SDPA attention
    try:
        backbone = AutoModelForSequenceClassification.from_pretrained(
            load_from, attn_implementation="sdpa", **backbone_kwargs
        )
        logger.info("Using SDPA attention implementation")
    except Exception:
        backbone = AutoModelForSequenceClassification.from_pretrained(
            load_from, **backbone_kwargs
        )
        logger.info("Using default attention implementation")

    hidden_dim = config.hidden_size

    # Gradient checkpointing
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # LoRA
    if use_lora and lora_rank > 0:
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            # Determine target modules based on model type
            if "deberta" in model_name.lower():
                target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )

            backbone = get_peft_model(backbone, lora_config)
            trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in backbone.parameters())
            logger.info(
                f"LoRA applied: r={lora_rank}, α={lora_alpha}\n"
                f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
                f"({100*trainable/total:.2f}%)"
            )
        except ImportError:
            logger.warning("peft not installed — using full fine-tuning")

    # Build NST-VERI model
    model = NSTVeriModel(
        backbone=backbone,
        hidden_dim=hidden_dim,
        n_constraints=n_constraints,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        dropout=dropout,
        contrastive_temperature=contrastive_temperature,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"NST-VERI model built:\n"
        f"  Total params: {total_params/1e6:.2f}M\n"
        f"  Trainable: {trainable_params/1e6:.2f}M\n"
        f"  Verification heads: {n_constraints}\n"
        f"  Contrastive head: projection → {min(256, hidden_dim)}d"
    )

    return tokenizer, model


def create_optimizer(
    model: nn.Module,
    lr_backbone: float = 1e-5,
    lr_lora: float = 3e-4,
    lr_heads: float = 5e-4,
    lr_gate: float = 1e-3,
    weight_decay: float = 0.01,
    use_fused: bool = True,
    adaptive_lambda_module: Optional[nn.Module] = None,
) -> torch.optim.AdamW:
    """Create optimizer with parameter-group-specific learning rates.

    Separates:
      - LoRA parameters (3× base LR)
      - Classification / verification / contrastive heads (high LR)
      - Adaptive lambda / gate module (highest LR)
      - Other backbone parameters (base LR)

    Uses fused AdamW on CUDA for 5-10% speedup.
    """
    lora_params = []
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if "lora" in name_lower:
            lora_params.append(param)
        elif any(k in name_lower for k in [
            "classifier", "verification", "contrastive",
            "residual_net", "residual_scale", "prototypes", "projector",
        ]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params, "lr": lr_backbone, "name": "backbone"
        })
    if lora_params:
        param_groups.append({
            "params": lora_params, "lr": lr_lora, "name": "lora"
        })
    if head_params:
        param_groups.append({
            "params": head_params, "lr": lr_heads, "name": "heads"
        })

    # Adaptive lambda module
    if adaptive_lambda_module is not None:
        gate_params = list(adaptive_lambda_module.parameters())
        if gate_params:
            param_groups.append({
                "params": gate_params, "lr": lr_gate, "name": "adaptive_lambda"
            })

    # Log parameter groups
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        logger.info(f"  Optimizer group '{pg.get('name', '?')}': "
                     f"{n_params/1e6:.2f}M params, lr={pg['lr']}")

    # Fused optimizer: CUDA kernel fusion for 5-10% speedup
    fused = use_fused and torch.cuda.is_available()
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            fused=fused,
        )
        if fused:
            logger.info("Using fused AdamW optimizer")
    except TypeError:
        # Older PyTorch doesn't support fused
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
        )

    return optimizer


def create_dataloader(
    dataset,
    tokenizer,
    batch_size: int,
    max_length: int = 384,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> torch.utils.data.DataLoader:
    """A100-optimized DataLoader with proper worker configuration.

    Key optimizations:
      - pin_memory: faster CPU→GPU transfers
      - persistent_workers: avoid worker restart overhead
      - prefetch_factor: pre-load batches for overlap with compute
      - drop_last during training: even batch sizes for torch.compile
    """
    from functools import partial
    from data.fever_dataset import fever_collate_fn

    collate = partial(fever_collate_fn, tokenizer=tokenizer, max_length=max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)

    use_persistent = persistent_workers and num_workers > 0
    pf = prefetch_factor if num_workers > 0 else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=shuffle,  # Drop last incomplete batch during training
        generator=generator if shuffle else None,
        persistent_workers=use_persistent,
        prefetch_factor=pf,
    )
