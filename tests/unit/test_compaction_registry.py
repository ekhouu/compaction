"""Smoke test for compaction/compaction_methods/registry.py.

Verifies that get_compaction_method(...) returns usable instances for
'original' and each of the baseline algorithms the batched runner
schedules by default. Does not load a model; just constructs the
method objects.

Gated behind a torch import so the test skips cleanly in environments
without the heavy ML stack installed (e.g. a fast pre-flight check on
a dev laptop).

Run standalone:
    python tests/unit/test_compaction_registry.py
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
    import torch  # noqa: F401
    from compaction.compaction_methods.registry import get_compaction_method
    from compaction.algorithms import ALGORITHM_REGISTRY
except Exception as exc:
    _SKIP_REASON = f"skipped: heavy deps unavailable ({exc.__class__.__name__}: {exc})"
    get_compaction_method = None  # type: ignore[assignment]
    ALGORITHM_REGISTRY = {}  # type: ignore[assignment]


def _skip_if_needed():
    if _SKIP_REASON is not None:
        print(_SKIP_REASON)
        return True
    return False


def test_original_method_instantiates():
    if _skip_if_needed():
        return
    m = get_compaction_method("original")
    assert m.name() == "original"
    assert m.returns_cache() is False


def test_baseline_algorithms_are_in_registry():
    if _skip_if_needed():
        return
    for algo in ("random_subset_keys", "random_vector_keys", "truncate"):
        assert algo in ALGORITHM_REGISTRY, f"{algo} missing from ALGORITHM_REGISTRY"


def test_highest_attention_keys_instantiates():
    if _skip_if_needed():
        return
    m = get_compaction_method(
        "highest_attention_keys",
        {"algorithm": "highest_attention_keys", "score_method": "rms"},
    )
    assert m is not None


def test_random_subset_instantiates():
    if _skip_if_needed():
        return
    m = get_compaction_method(
        "random_subset_keys",
        {"algorithm": "random_subset_keys"},
    )
    assert m is not None


def test_truncate_instantiates():
    if _skip_if_needed():
        return
    m = get_compaction_method(
        "truncate",
        {"algorithm": "truncate"},
    )
    assert m is not None


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
