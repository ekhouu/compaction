"""End-to-end dry-run tests for run_batched_benchmarks.py.

Launches the runner as a subprocess with --dry-run for a handful of
configurations that exercise the compaction-side code path (TQ-side
paths are covered separately because they require the llama.cpp
submodule to be checked out, which is not guaranteed in CI).

These tests assert on:
  * non-zero exit => test failure
  * the planned shell command actually contains the expected pieces
    (method list, dataset name, sample sizes)

Run standalone:
    python tests/unit/test_runner_dry_run.py
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import traceback


_HERE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_RUNNER = _REPO_ROOT / "tests" / "batched" / "run_batched_benchmarks.py"


def _run(extra_args: list[str]) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(_RUNNER),
        "--mode",
        "compaction",
        "--dry-run",
        "--hf-model",
        "Qwen/Qwen3-4B",
    ] + extra_args
    env = os.environ.copy()
    # Isolate from the user's env so tests are reproducible.
    for var in ("HF_MODEL_PATH", "HF_HOME", "UV_CACHE_DIR"):
        env.pop(var, None)
    return subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _assert_success(result: subprocess.CompletedProcess, label: str) -> None:
    if result.returncode != 0:
        raise AssertionError(
            f"{label}: runner exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_default_compaction_dry_run_lists_baseline_methods():
    result = _run([])
    _assert_success(result, "default compaction dry-run")
    # The resolved command must contain 'original' plus the three defaults.
    out = result.stdout
    assert "--methods original" in out, out
    assert "highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy" in out, out
    assert "random_subset_keys_nnls2_-3_3_lsq_on-policy" in out, out
    assert "truncate_nnls2_-3_3_lsq_on-policy" in out, out
    # Default algorithm-config should now be key-selection, not default.
    assert "--algorithm-config key-selection" in out, out


def test_smoke_mode_prints_clamp_and_reduces_samples():
    result = _run(["--smoke"])
    _assert_success(result, "smoke dry-run")
    out = result.stdout
    assert "[smoke] compaction_n_articles: 10 -> 2" in out, out
    assert "--n-articles 2" in out, out
    assert "--n-questions-per-article 3" in out, out


def test_shared_niah_override_rewrites_dataset_and_methods_still_present():
    result = _run(
        [
            "--shared-niah-ruler-config",
            "ruler_4k_niah_single_1",
            "--compaction-methods",
            "AM,random_subset",
        ]
    )
    _assert_success(result, "shared-niah dry-run")
    out = result.stdout
    assert "[shared-niah] overriding --compaction-dataset" in out, out
    assert "--dataset-name ruler_4k_niah_single_1" in out, out
    assert "highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy" in out, out
    assert "random_subset_keys_nnls2_-3_3_lsq_on-policy" in out, out


def test_missing_hf_model_errors_loudly():
    # Empty --hf-model should trigger require_compaction_args and exit non-zero.
    cmd = [
        sys.executable,
        str(_RUNNER),
        "--mode",
        "compaction",
        "--dry-run",
        "--hf-model",
        "",
    ]
    env = os.environ.copy()
    env.pop("HF_MODEL_PATH", None)
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        raise AssertionError(
            "expected non-zero exit when --hf-model is empty; "
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert "--hf-model is required" in result.stderr, result.stderr


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
