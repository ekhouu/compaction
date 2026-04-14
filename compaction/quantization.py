# compaction/quantization.py
"""
Simulate KV cache quantization noise for ceteris paribus evaluation.

Provides quantize-then-dequantize functions that mirror the precision loss of
integer quantization formats (q8_0, q4_0, etc.) used by inference engines like
llama.cpp and TurboQuant+.  The tensors stay in their original dtype -- only
the *noise pattern* of low-bit storage is injected.

Supported bit-widths:
    8  -- analogous to llama.cpp q8_0 (group_size=32)
    4  -- analogous to llama.cpp q4_0 (group_size=32)
    2  -- aggressive low-bit (group_size=32)
"""
from __future__ import annotations

import torch
from torch import Tensor


def quantize_dequantize(
    tensor: Tensor,
    bits: int = 8,
    group_size: int = 32,
) -> Tensor:
    """Round-trip quantise a tensor and return the dequantised result.

    Uses symmetric per-group quantization: each contiguous group of
    ``group_size`` elements along the *last* dimension shares a single
    absmax scale factor.

    Parameters
    ----------
    tensor : Tensor
        Input tensor (any shape).  The last dimension must be divisible by
        ``group_size`` when ``group_size > 0``.
    bits : int
        Quantization bit-width (2, 4, or 8).
    group_size : int
        Number of elements per quantization group.  Set to 0 or negative
        to fall back to per-tensor quantization.

    Returns
    -------
    Tensor
        Dequantised tensor with the same shape and dtype as the input.
    """
    if bits >= 16:
        return tensor  # no-op for fp16 / bf16

    orig_dtype = tensor.dtype
    # Work in float32 for precision of scale computation
    t = tensor.float()

    qmax = (1 << (bits - 1)) - 1  # e.g. 127 for int8, 7 for int4

    if group_size > 0 and t.shape[-1] >= group_size:
        assert t.shape[-1] % group_size == 0, (
            f"Last dim ({t.shape[-1]}) must be divisible by group_size ({group_size})"
        )
        shape = t.shape
        n_groups = shape[-1] // group_size
        grouped = t.reshape(*shape[:-1], n_groups, group_size)

        # Per-group absmax scale
        amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = amax / qmax

        quantized = (grouped / scale).round().clamp(-qmax, qmax)
        dequantized = (quantized * scale).reshape(shape)
    else:
        # Per-tensor fallback
        amax = t.abs().amax().clamp(min=1e-12)
        scale = amax / qmax
        quantized = (t / scale).round().clamp(-qmax, qmax)
        dequantized = quantized * scale

    return dequantized.to(orig_dtype)


def quantize_kv_cache(
    past_key_values: tuple[tuple[Tensor, Tensor], ...],
    bits: int = 8,
    group_size: int = 32,
) -> tuple[tuple[Tensor, Tensor], ...]:
    """Apply simulated quantization noise to an entire KV cache.

    Parameters
    ----------
    past_key_values : tuple of (keys, values)
        Standard HF KV cache.  Each entry is a pair of tensors with shape
        ``(batch, heads, seq_len, head_dim)``.
    bits : int
        Quantization bit-width (2, 4, or 8).
    group_size : int
        Group size for per-group quantization.

    Returns
    -------
    tuple of (keys, values)
        KV cache with quantization noise injected.  Same shapes and dtype
        as the input.
    """
    quantized: list[tuple[Tensor, Tensor]] = []
    for keys, values in past_key_values:
        q_keys = quantize_dequantize(keys, bits=bits, group_size=group_size)
        q_values = quantize_dequantize(values, bits=bits, group_size=group_size)
        quantized.append((q_keys, q_values))
    return tuple(quantized)


# Mapping from human-readable names to (bits, group_size) pairs.
QUANT_PRESETS: dict[str, tuple[int, int]] = {
    "none": (16, 0),      # no quantization
    "int8": (8, 32),      # ~q8_0
    "int4": (4, 32),      # ~q4_0
    "int2": (2, 32),      # aggressive 2-bit
    "q8_0": (8, 32),      # llama.cpp alias
    "q4_0": (4, 32),      # llama.cpp alias
}


def resolve_quant_preset(name: str) -> tuple[int, int]:
    """Look up ``(bits, group_size)`` for a preset name.

    Raises ``ValueError`` for unknown presets.
    """
    key = name.lower().strip()
    if key not in QUANT_PRESETS:
        raise ValueError(
            f"Unknown quantization preset {name!r}. "
            f"Available: {sorted(QUANT_PRESETS)}"
        )
    return QUANT_PRESETS[key]
