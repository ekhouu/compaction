# compaction/compaction_methods/quantized_wrapper.py
"""
Compaction methods that inject KV cache quantization noise.

QuantizedCacheMethod
    Quantise-only (no compaction).  Simulates "TQ+" in the ceteris paribus
    eval: same model, same prompts, same metrics -- the only difference is
    that K/V tensors are round-tripped through low-bit quantization.

QuantizeThenCompact
    Quantise *then* compact.  Simulates "TQ+AM": first inject quantization
    noise, then hand the noisy cache to an inner compaction method (e.g. AM).
"""
from __future__ import annotations

import time
import torch
from typing import Any, Dict, Optional, Tuple, Type

from .base import FullCacheCompactionAlgorithm
from ..quantization import quantize_dequantize, resolve_quant_preset
from ..query_generation import QueryConfig


def _quantize_cache_layers(
    past_key_values,
    bits: int,
    group_size: int,
    sliding_layer_indices: set,
):
    """Quantise K/V tensors in a cache, handling multiple formats.

    Supports:
    - Standard HF 2-tuple per layer: (keys, values)
    - Compacted 3-tuple per layer: (C1, beta, C2) -- quantises C1 and C2, keeps beta
    - Cache objects (e.g. CompactedPrefixCache) -- passed through unchanged with a
      warning, since quantising an already-compacted cache mid-generation is not the
      intended use case.

    Returns the same type/structure as the input.
    """
    # If it's not a plain tuple/list of tuples (e.g. a Cache subclass), pass through.
    # Quantization is meant for the raw HF cache before compaction, not for
    # already-wrapped Cache objects in the reasoning eval's mid-trajectory path.
    if not isinstance(past_key_values, (tuple, list)):
        print(f"Warning: QuantizeThenCompact received {type(past_key_values).__name__} "
              f"instead of raw KV tuple -- passing through without quantization")
        return past_key_values

    quantized: list[tuple] = []
    for layer_idx, layer in enumerate(past_key_values):
        if layer_idx in sliding_layer_indices:
            quantized.append(layer)
            continue

        if len(layer) == 3:
            # Compacted format: (C1, beta, C2)
            c1, beta, c2 = layer
            quantized.append((
                quantize_dequantize(c1, bits=bits, group_size=group_size),
                beta,
                quantize_dequantize(c2, bits=bits, group_size=group_size),
            ))
        else:
            # Standard HF format: (keys, values)
            keys, values = layer
            quantized.append((
                quantize_dequantize(keys, bits=bits, group_size=group_size),
                quantize_dequantize(values, bits=bits, group_size=group_size),
            ))

    return tuple(quantized)


class QuantizedCacheMethod(FullCacheCompactionAlgorithm):
    """Quantise the full KV cache without compacting (sequence length unchanged).

    This is the "TQ+" arm of the ceteris paribus comparison.
    Generation proceeds through the compacted-cache path (C1/beta/C2),
    with beta = 0 everywhere and C1/C2 = quantised K/V.
    """

    def __init__(
        self,
        quant_preset: str = "int8",
        config_name: Optional[str] = None,
    ):
        self.quant_preset = quant_preset
        self.bits, self.group_size = resolve_quant_preset(quant_preset)
        self._config_name = config_name or f"quantized_{quant_preset}"

    def name(self) -> str:
        return self._config_name

    def compact_kv_cache(
        self,
        past_key_values,
        target_size,
        indices,
        query_config,
        model=None,
        tokenizer=None,
        formatted_context=None,
        compute_stats=False,
        vllm_model=None,
        verbose_logging=False,
        sliding_layer_indices=None,
        **kwargs,
    ):
        """Quantise K/V and return as (C1, beta=0, C2) with full seq length."""
        sliding_layer_indices = sliding_layer_indices or set()
        num_layers = len(past_key_values)

        # Find a non-sliding layer for reference shapes
        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        batch_size, num_heads, seq_len, head_dim = past_key_values[ref_layer_idx][0].shape

        start = time.time()
        compacted_layers: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(num_layers):
            keys, values = past_key_values[layer_idx]

            if layer_idx in sliding_layer_indices:
                # Keep sliding-window layers unchanged (same as other methods)
                compacted_layers.append((
                    keys.new_zeros(batch_size, num_heads, 0, head_dim),
                    keys.new_zeros(batch_size, num_heads, 0),
                    values.new_zeros(batch_size, num_heads, 0, head_dim),
                ))
                continue

            q_keys = quantize_dequantize(keys, bits=self.bits, group_size=self.group_size)
            q_values = quantize_dequantize(values, bits=self.bits, group_size=self.group_size)
            beta = keys.new_zeros(batch_size, num_heads, keys.shape[2])

            compacted_layers.append((q_keys, beta, q_values))

        elapsed = time.time() - start

        stats = {
            'method': self._config_name,
            'quantization': self.quant_preset,
            'quantization_bits': self.bits,
            'quantization_group_size': self.group_size,
            'quantization_time_sec': elapsed,
            'compaction_ratio': 1.0,
            'tensor_compacted_seq_len': seq_len,
            'effective_compacted_seq_len': seq_len,
        }
        return tuple(compacted_layers), stats


class QuantizeThenCompact(FullCacheCompactionAlgorithm):
    """Quantise the KV cache, then apply an inner compaction method.

    This is the "TQ+AM" arm of the ceteris paribus comparison:
    inject quantization noise first, then compact with AM (or any other method).
    """

    def __init__(
        self,
        inner_method: FullCacheCompactionAlgorithm,
        quant_preset: str = "int8",
        config_name: Optional[str] = None,
    ):
        self.inner_method = inner_method
        self.quant_preset = quant_preset
        self.bits, self.group_size = resolve_quant_preset(quant_preset)
        self._config_name = config_name or f"quantized_{quant_preset}+{inner_method.name()}"

    def name(self) -> str:
        return self._config_name

    def requires_preextracted_cache(self) -> bool:
        return self.inner_method.requires_preextracted_cache()

    def compact_kv_cache(
        self,
        past_key_values,
        target_size,
        indices,
        query_config,
        model=None,
        tokenizer=None,
        formatted_context=None,
        compute_stats=False,
        vllm_model=None,
        verbose_logging=False,
        sliding_layer_indices=None,
        **kwargs,
    ):
        """Quantise the KV cache, then delegate to the inner method."""
        sliding_layer_indices = sliding_layer_indices or set()

        # Step 1: quantise (skip if cache is None -- inner method handles its own extraction)
        quant_start = time.time()
        if past_key_values is not None:
            kv_for_inner = _quantize_cache_layers(
                past_key_values, self.bits, self.group_size, sliding_layer_indices,
            )
        else:
            kv_for_inner = None
        quant_time = time.time() - quant_start

        # Step 2: delegate to inner compaction method
        compacted_cache, inner_stats = self.inner_method.compact_kv_cache(
            past_key_values=kv_for_inner,
            target_size=target_size,
            indices=indices,
            query_config=query_config,
            model=model,
            tokenizer=tokenizer,
            formatted_context=formatted_context,
            compute_stats=compute_stats,
            vllm_model=vllm_model,
            verbose_logging=verbose_logging,
            sliding_layer_indices=sliding_layer_indices,
            **kwargs,
        )

        # Merge stats
        inner_stats['quantization'] = self.quant_preset
        inner_stats['quantization_bits'] = self.bits
        inner_stats['quantization_group_size'] = self.group_size
        inner_stats['quantization_time_sec'] = quant_time
        inner_stats['method'] = self._config_name

        return compacted_cache, inner_stats
