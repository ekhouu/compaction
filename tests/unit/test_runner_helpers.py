"""Unit tests for pure helpers in tests/batched/run_batched_benchmarks.py.

These exercise alias resolution, RULER length parsing, and the smoke/
shared-niah override mutators. No torch, no subprocess, no GPU.

Run standalone:
    python tests/unit/test_runner_helpers.py

Run under pytest (pytest auto-collects the module-level test_* functions):
    pytest tests/unit/test_runner_helpers.py
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import traceback


_HERE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_RUNNER_PATH = _REPO_ROOT / "tests" / "batched" / "run_batched_benchmarks.py"


def _load_runner_module():
    module_name = "rbb_under_test"
    spec = importlib.util.spec_from_file_location(module_name, _RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register in sys.modules before execution so @dataclass can look up
    # the defining module via sys.modules[cls.__module__] (required on
    # Python 3.12+ and strictly required on 3.14).
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


rbb = _load_runner_module()


def _make_args(**overrides) -> argparse.Namespace:
    """Build a Namespace with every field the override helpers read.

    Keep this in lockstep with apply_smoke_overrides and
    apply_shared_niah_override in run_batched_benchmarks.py.
    """
    base = argparse.Namespace(
        smoke=False,
        shared_niah_ruler_config=None,
        compaction_dataset="quality",
        compaction_n_articles=10,
        compaction_n_questions_per_article=10,
        compaction_target_size=0.1,
        compaction_target_sizes=None,
        compaction_bootstrap_samples=1000,
        compaction_compute_perplexity=1,
        compaction_num_runs=1,
        tq_depths="4096,8192,16384",
        tq_depths_sweep="0,10,20,30,40,50,60,70,80,90,100",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_resolve_methods_expands_aliases_in_order():
    got = rbb.resolve_compaction_methods("AM,random_subset,truncate")
    assert got == [
        "highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy",
        "random_subset_keys_nnls2_-3_3_lsq_on-policy",
        "truncate_nnls2_-3_3_lsq_on-policy",
    ], got


def test_resolve_methods_passes_unknown_keys_through():
    got = rbb.resolve_compaction_methods("AM,custom_config_name")
    assert got == [
        "highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy",
        "custom_config_name",
    ], got


def test_resolve_methods_deduplicates():
    got = rbb.resolve_compaction_methods("AM, AM ,AM")
    assert got == ["highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy"], got


def test_resolve_methods_rejects_empty_list():
    try:
        rbb.resolve_compaction_methods(" , , ")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty method list")


def test_parse_ruler_context_length_basic():
    assert rbb.parse_ruler_context_length("ruler_4k") == 4096
    assert rbb.parse_ruler_context_length("ruler_128k") == 131072
    assert rbb.parse_ruler_context_length("ruler_16k_niah_single_1") == 16384


def test_parse_ruler_context_length_rejects_non_ruler():
    for bad in ("quality", "ruler_", "ruler_abc", "ruler4k"):
        try:
            rbb.parse_ruler_context_length(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_apply_shared_niah_override_rewrites_fields():
    args = _make_args(shared_niah_ruler_config="ruler_4k_niah_single_1")
    rbb.apply_shared_niah_override(args)
    assert args.compaction_dataset == "ruler_4k_niah_single_1"
    assert args.tq_depths == "4096"


def test_apply_shared_niah_override_noop_when_unset():
    args = _make_args()
    rbb.apply_shared_niah_override(args)
    assert args.compaction_dataset == "quality"
    assert args.tq_depths == "4096,8192,16384"


def test_apply_smoke_overrides_clamps_expected_fields():
    args = _make_args(smoke=True)
    rbb.apply_smoke_overrides(args)
    assert args.compaction_n_articles == 2
    assert args.compaction_n_questions_per_article == 3
    assert args.compaction_target_sizes == "0.1"
    assert args.compaction_bootstrap_samples == 0
    assert args.compaction_compute_perplexity == 0
    assert args.tq_depths == "4096"
    assert args.tq_depths_sweep == "0,50,100"


def test_apply_smoke_overrides_preserves_shared_niah_tq_depths():
    # When --shared-niah-ruler-config has already rewritten tq_depths,
    # smoke must not clobber it back to a generic default.
    args = _make_args(
        smoke=True,
        shared_niah_ruler_config="ruler_16k",
        tq_depths="16384",
    )
    rbb.apply_smoke_overrides(args)
    assert args.tq_depths == "16384"


def test_apply_smoke_overrides_noop_when_disabled():
    args = _make_args(smoke=False, compaction_n_articles=42)
    rbb.apply_smoke_overrides(args)
    assert args.compaction_n_articles == 42


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
