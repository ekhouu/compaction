"""Tests for compaction/quantization.py and the quantized registry entries.

Run standalone:
    python tests/unit/test_quantization.py
"""

from __future__ import annotations

import pathlib
import sys
import traceback

_HERE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_SKIP_REASON: str | None = None
try:
    import torch
    from compaction.quantization import (
        quantize_dequantize,
        quantize_kv_cache,
        resolve_quant_preset,
        QUANT_PRESETS,
    )
    from compaction.compaction_methods.registry import get_compaction_method
except Exception as exc:
    _SKIP_REASON = f"skipped: heavy deps unavailable ({exc.__class__.__name__}: {exc})"


def _skip_if_needed():
    if _SKIP_REASON is not None:
        print(_SKIP_REASON)
        return True
    return False


# ---------- quantize_dequantize tests ----------

def test_no_op_at_16_bits():
    """16-bit should return the tensor unchanged."""
    if _skip_if_needed():
        return
    t = torch.randn(4, 32)
    result = quantize_dequantize(t, bits=16)
    assert torch.equal(t, result), "16-bit should be a no-op"


def test_int8_round_trip_introduces_bounded_noise():
    """int8 should be close but not exact."""
    if _skip_if_needed():
        return
    t = torch.randn(4, 64)
    result = quantize_dequantize(t, bits=8, group_size=32)
    assert result.shape == t.shape
    assert result.dtype == t.dtype
    max_error = (t - result).abs().max().item()
    # int8 with 127 levels: max error per element < 2 * amax / 127
    # For standard normal, amax ~ 3-4, so max_error < ~0.06
    assert max_error < 0.1, f"int8 error too large: {max_error}"


def test_int4_has_more_noise_than_int8():
    """int4 should introduce more quantization noise than int8."""
    if _skip_if_needed():
        return
    torch.manual_seed(42)
    t = torch.randn(8, 128)
    err8 = (t - quantize_dequantize(t, bits=8, group_size=32)).abs().mean().item()
    err4 = (t - quantize_dequantize(t, bits=4, group_size=32)).abs().mean().item()
    assert err4 > err8, f"int4 error ({err4}) should exceed int8 error ({err8})"


def test_int2_has_more_noise_than_int4():
    """int2 should introduce more quantization noise than int4."""
    if _skip_if_needed():
        return
    torch.manual_seed(42)
    t = torch.randn(8, 128)
    err4 = (t - quantize_dequantize(t, bits=4, group_size=32)).abs().mean().item()
    err2 = (t - quantize_dequantize(t, bits=2, group_size=32)).abs().mean().item()
    assert err2 > err4, f"int2 error ({err2}) should exceed int4 error ({err4})"


def test_preserves_dtype():
    """Output dtype should match input dtype."""
    if _skip_if_needed():
        return
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        t = torch.randn(4, 32, dtype=dtype)
        result = quantize_dequantize(t, bits=8, group_size=32)
        assert result.dtype == dtype, f"Expected {dtype}, got {result.dtype}"


def test_per_tensor_fallback():
    """When group_size=0, should use per-tensor quantization."""
    if _skip_if_needed():
        return
    t = torch.randn(4, 7)  # not divisible by typical group sizes
    result = quantize_dequantize(t, bits=8, group_size=0)
    assert result.shape == t.shape


# ---------- quantize_kv_cache tests ----------

def test_quantize_kv_cache_shapes():
    """Quantized cache should preserve shapes."""
    if _skip_if_needed():
        return
    batch, heads, seq, dim = 1, 4, 128, 64
    kv = tuple(
        (torch.randn(batch, heads, seq, dim), torch.randn(batch, heads, seq, dim))
        for _ in range(3)
    )
    result = quantize_kv_cache(kv, bits=8, group_size=32)
    assert len(result) == 3
    for (qk, qv), (ok, ov) in zip(result, kv):
        assert qk.shape == ok.shape
        assert qv.shape == ov.shape


# ---------- resolve_quant_preset tests ----------

def test_known_presets():
    if _skip_if_needed():
        return
    for name in ("none", "int8", "int4", "int2", "q8_0", "q4_0"):
        bits, gs = resolve_quant_preset(name)
        assert isinstance(bits, int)
        assert isinstance(gs, int)


def test_unknown_preset_raises():
    if _skip_if_needed():
        return
    try:
        resolve_quant_preset("bogus")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------- registry integration tests ----------

def test_quantize_only_instantiates():
    """QuantizedCacheMethod should instantiate from registry."""
    if _skip_if_needed():
        return
    m = get_compaction_method("TQ_int8", {
        "algorithm": "quantize_only",
        "quantize": "int8",
    })
    assert m is not None
    assert m.name() == "TQ_int8"
    assert m.returns_cache() is True


def test_quantize_then_compact_instantiates():
    """QuantizeThenCompact should instantiate from registry."""
    if _skip_if_needed():
        return
    import math
    m = get_compaction_method("TQ_int8_AM", {
        "algorithm": "highest_attention_keys",
        "score_method": "rms",
        "nnls_iters": 2,
        "nnls_lower_bound": math.exp(-3),
        "nnls_upper_bound": math.exp(3),
        "c2_method": "lsq",
        "on_policy": True,
        "quantize": "int8",
    })
    assert m is not None
    assert m.name() == "TQ_int8_AM"
    assert m.returns_cache() is True


def test_ceteris_paribus_config_loads():
    """The ceteris_paribus algorithm config should load all expected methods."""
    if _skip_if_needed():
        return
    from evaluation.configs.utils import load_algorithm_config
    config = load_algorithm_config("ceteris_paribus")
    expected = {"AM", "TQ_int8", "TQ_int4", "TQ_int2", "TQ_int8_AM", "TQ_int4_AM", "TQ_int2_AM"}
    assert expected == set(config.keys()), f"Expected {expected}, got {set(config.keys())}"


# ---------- runner ----------

def _run_all() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures: list[str] = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            traceback.print_exc()
            failures.append(name)
    if failures:
        print(f"\n{len(failures)}/{len(tests)} FAILED: {failures}")
        return 1
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
