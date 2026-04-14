#!/usr/bin/env python3

"""
Simple batched benchmark runner.

Runs:
1) TurboQuant+ NIAH benchmark
2) Compaction QA benchmark
3) Both (parallel or sequential)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path


# Humane aliases for verbose hyperparameter-tagged config names in
# evaluation/configs/algorithms/key-selection.py. Each alias maps to the
# exact config key used by load_algorithm_config(); all five entries share
# matched NNLS hyperparameters so they form a ceteris-paribus comparison.
COMPACTION_METHOD_ALIASES: dict[str, str] = {
    "AM": "highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy",
    "OMP": "omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy",
    "random_subset": "random_subset_keys_nnls2_-3_3_lsq_on-policy",
    "random_vector": "random_vector_keys_nnls2_-3_3_lsq_on-policy",
    "truncate": "truncate_nnls2_-3_3_lsq_on-policy",
}

DEFAULT_COMPACTION_METHODS = "AM,random_subset,truncate"

# RULER NIAH dataset names are of the form 'ruler_<len>[_task]' where <len>
# is e.g. '4k', '16k', '128k'. This regex extracts just the length token.
_RULER_LENGTH_RE = re.compile(r"^ruler_(\d+k)(?:_.*)?$", re.IGNORECASE)


def resolve_compaction_methods(raw: str) -> list[str]:
    """Expand a CSV list of user-facing method names into config-key names."""
    out: list[str] = []
    for token in raw.split(","):
        name = token.strip()
        if not name:
            continue
        resolved = COMPACTION_METHOD_ALIASES.get(name, name)
        if resolved not in out:
            out.append(resolved)
    if not out:
        raise ValueError("--compaction-methods resolved to an empty list")
    return out


def parse_ruler_context_length(dataset_name: str) -> int:
    m = _RULER_LENGTH_RE.match(dataset_name)
    if not m:
        raise ValueError(
            f"--shared-niah-ruler-config must be a RULER dataset name like "
            f"'ruler_4k' or 'ruler_4k_niah_single_1', got: {dataset_name!r}"
        )
    token = m.group(1).lower()
    return int(token[:-1]) * 1024


@dataclass
class JobSpec:
    name: str
    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    timeout_sec: int


@dataclass
class JobResult:
    name: str
    success: bool
    return_code: int
    duration_sec: float
    log_path: str
    timed_out: bool
    message: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def format_cmd(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_job(job: JobSpec, run_dir: Path) -> JobResult:
    log_path = run_dir / f"{job.name}.log"
    start = time.time()
    timed_out = False

    with open(log_path, "a") as logf:
        logf.write(f"\n\n===== {utc_now_iso()} =====\n")
        logf.write(f"cwd: {job.cwd}\n")
        logf.write(f"cmd: {format_cmd(job.cmd)}\n")
        logf.flush()

        try:
            proc = subprocess.run(
                job.cmd,
                cwd=str(job.cwd),
                env=job.env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=job.timeout_sec,
                check=False,
            )
            rc = proc.returncode
            msg = "completed" if rc == 0 else f"exit code {rc}"
        except subprocess.TimeoutExpired:
            rc = 124
            timed_out = True
            msg = f"timed out after {job.timeout_sec}s"

    return JobResult(
        name=job.name,
        success=(rc == 0),
        return_code=rc,
        duration_sec=time.time() - start,
        log_path=str(log_path),
        timed_out=timed_out,
        message=msg,
    )


def run_jobs_parallel(jobs: list[JobSpec], run_dir: Path) -> list[JobResult]:
    if len(jobs) != 2:
        raise ValueError("parallel mode expects exactly 2 jobs")

    states: dict[str, dict] = {}
    for job in jobs:
        log_path = run_dir / f"{job.name}.log"
        logf = open(log_path, "a")
        logf.write(
            f"\n\n===== {utc_now_iso()} | parallel =====\n"
            f"cwd: {job.cwd}\n"
            f"cmd: {format_cmd(job.cmd)}\n"
        )
        logf.flush()
        proc = subprocess.Popen(
            job.cmd,
            cwd=str(job.cwd),
            env=job.env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        states[job.name] = {
            "job": job,
            "proc": proc,
            "logf": logf,
            "log_path": log_path,
            "start": time.time(),
            "finished": False,
            "rc": None,
            "timed_out": False,
            "message": "",
        }

    while not all(st["finished"] for st in states.values()):
        now = time.time()
        for st in states.values():
            if st["finished"]:
                continue
            rc = st["proc"].poll()
            if rc is not None:
                st["finished"] = True
                st["rc"] = rc
                st["message"] = "completed" if rc == 0 else f"exit code {rc}"
                continue

            if (now - st["start"]) > st["job"].timeout_sec:
                st["timed_out"] = True
                st["proc"].terminate()
                try:
                    st["proc"].wait(timeout=10)
                except Exception:
                    st["proc"].kill()
                    st["proc"].wait(timeout=5)
                st["finished"] = True
                st["rc"] = 124
                st["message"] = f"timed out after {st['job'].timeout_sec}s"
        time.sleep(1)

    results: list[JobResult] = []
    for name, st in states.items():
        st["logf"].close()
        results.append(
            JobResult(
                name=name,
                success=(st["rc"] == 0),
                return_code=int(st["rc"]) if st["rc"] is not None else -1,
                duration_sec=time.time() - st["start"],
                log_path=str(st["log_path"]),
                timed_out=bool(st["timed_out"]),
                message=str(st["message"]),
            )
        )
    return results


def build_tq_job(args: argparse.Namespace, run_dir: Path, env: dict[str, str]) -> JobSpec:
    tq_script = Path(args.tq_script).resolve()
    llama_dir = Path(args.llama_dir).resolve()
    gguf_model = Path(args.gguf_model).resolve()

    if not tq_script.exists():
        raise FileNotFoundError(f"TQ script not found: {tq_script}")
    if not llama_dir.exists():
        raise FileNotFoundError(f"llama.cpp directory not found: {llama_dir}")
    if not gguf_model.exists():
        raise FileNotFoundError(f"GGUF model not found: {gguf_model}")

    cmd = [
        args.python_bin,
        str(tq_script),
        str(llama_dir),
        str(gguf_model),
        "--mode",
        args.tq_mode,
        "--cache-types",
        args.tq_cache_types,
        "--depths",
        args.tq_depths,
        "--depths-sweep",
        args.tq_depths_sweep,
        "--output-dir",
        str((run_dir / "tq_results").resolve()),
        "--port",
        str(args.tq_port),
        "--server-timeout",
        str(args.tq_server_timeout),
        "--query-timeout",
        str(args.tq_query_timeout),
    ]
    if args.tq_mode == "multi-key":
        cmd += ["--num-distractors", str(args.tq_num_distractors)]
    if args.tq_mode == "multi-value":
        cmd += ["--value-counts", args.tq_value_counts]

    return JobSpec(
        name="tq_benchmark",
        cmd=cmd,
        cwd=Path.cwd(),
        env=env.copy(),
        timeout_sec=args.tq_timeout_sec,
    )


def parse_csv_values(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_compaction_variants(args: argparse.Namespace) -> list[dict[str, object]]:
    target_sizes_raw = parse_csv_values(args.compaction_target_sizes)
    if target_sizes_raw:
        target_sizes = [float(v) for v in target_sizes_raw]
    else:
        target_sizes = [float(args.compaction_target_size)]

    query_configs = parse_csv_values(args.compaction_query_configs)
    if not query_configs:
        query_configs = [args.compaction_query_config]

    budget_modes = ["uniform"]
    if args.compaction_compare_budgets:
        if not args.compaction_precomputed_budget_path:
            raise ValueError(
                "--compaction-compare-budgets requires --compaction-precomputed-budget-path"
            )
        budget_modes.append("precomputed")

    if args.compaction_num_runs < 1:
        raise ValueError("--compaction-num-runs must be >= 1")
    if args.compaction_bootstrap_samples < 0:
        raise ValueError("--compaction-bootstrap-samples must be >= 0")
    if not (0.0 < args.compaction_ci_confidence < 1.0):
        raise ValueError("--compaction-ci-confidence must be in (0, 1)")
    if args.seed is None:
        seeds: list[int | None] = [None] * args.compaction_num_runs
    else:
        seeds = [args.seed + i for i in range(args.compaction_num_runs)]

    variants: list[dict[str, object]] = []
    for target_size, query_config, budget_mode, seed in product(
        target_sizes, query_configs, budget_modes, seeds
    ):
        variants.append(
            {
                "target_size": target_size,
                "query_config": query_config,
                "budget_mode": budget_mode,
                "seed": seed,
            }
        )
    return variants


def build_compaction_job(
    args: argparse.Namespace,
    run_dir: Path,
    env: dict[str, str],
    variant_idx: int,
    variant: dict[str, object],
) -> JobSpec:
    target_size = float(variant["target_size"])
    query_config = str(variant["query_config"])
    budget_mode = str(variant["budget_mode"])
    seed = variant["seed"]

    safe_query = query_config.replace("/", "-").replace(" ", "_")
    run_name = (
        f"batched_{args.compaction_dataset}"
        f"_t{target_size:g}_q{safe_query}_{budget_mode}"
    )
    if seed is not None:
        run_name += f"_s{seed}"

    resolved_methods = resolve_compaction_methods(args.compaction_methods)
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "evaluation.run_qa_evaluation",
        "--algorithm-config",
        args.compaction_algorithm_config,
        "--methods",
        "original",
        *resolved_methods,
        "--dataset-name",
        args.compaction_dataset,
        "--n-articles",
        str(args.compaction_n_articles),
        "--target-size",
        str(target_size),
        "--compute-stats",
        str(args.compaction_compute_stats),
        "--compute-perplexity",
        str(args.compaction_compute_perplexity),
        "--device",
        "cuda",
        "--model-name",
        args.hf_model,
        "--query-config",
        query_config,
        "--log-dir",
        str((run_dir / "compaction_eval_logs").resolve()),
        "--name",
        run_name,
    ]
    if args.seed is not None:
        cmd += ["--seed", str(seed)]
    if args.compaction_bootstrap_samples > 0:
        cmd += ["--bootstrap-samples", str(args.compaction_bootstrap_samples)]
        cmd += ["--ci-confidence", str(args.compaction_ci_confidence)]
    if args.compaction_n_questions_per_article is not None:
        cmd += ["--n-questions-per-article", str(args.compaction_n_questions_per_article)]
    if budget_mode == "precomputed" and args.compaction_precomputed_budget_path:
        cmd += ["--precomputed-budget-path", args.compaction_precomputed_budget_path]
        cmd += ["--max-ratio-per-head", str(args.max_ratio_per_head)]

    run_env = env.copy()
    if seed is not None:
        run_env["PYTHONHASHSEED"] = str(seed)
    if args.uv_cache_dir:
        run_env["UV_CACHE_DIR"] = args.uv_cache_dir
    if args.hf_home:
        run_env["HF_HOME"] = args.hf_home
        run_env["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.hf_home, "hub")

    return JobSpec(
        name=f"compaction_benchmark_{variant_idx:03d}",
        cmd=cmd,
        cwd=Path.cwd(),
        env=run_env,
        timeout_sec=args.compaction_timeout_sec,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple batched benchmark runner")
    p.add_argument("--mode", choices=["tq", "compaction", "both", "all"], required=True)
    p.add_argument("--both-strategy", choices=["parallel", "sequential"], default="parallel")
    p.add_argument(
        "--all-order",
        default="tq,compaction,both",
        help="Order for --mode all, comma-separated subset of tq,compaction,both",
    )
    p.add_argument("--run-root", default="logs/batched")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--deterministic-cuda", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke-test config: clamps sample sizes, target-size sweep, and TQ "
            "depth sweep to a minimal run suitable for validating plumbing on a "
            "local GPU before spending H100 time. Overrides conflicting flags."
        ),
    )
    p.add_argument(
        "--shared-niah-ruler-config",
        default=None,
        help=(
            "Point both the compaction and TQ+ jobs at a shared RULER NIAH "
            "dataset (e.g. 'ruler_4k_niah_single_1'). Forces --compaction-dataset "
            "and derives --tq-depths from the context-length token."
        ),
    )

    p.add_argument("--python-bin", default=sys.executable)

    # TQ+ args
    p.add_argument("--tq-script", default="_external/turboquant_plus/scripts/niah_test.py")
    p.add_argument("--llama-dir", default=None)
    p.add_argument("--gguf-model", default=None)
    p.add_argument(
        "--gguf-sha256",
        default=None,
        help="Optional SHA256 checksum for GGUF file (recorded in metadata).",
    )
    p.add_argument("--tq-mode", choices=["single", "multi-key", "multi-value"], default="single")
    p.add_argument("--tq-cache-types", default="q8_0,turbo3")
    p.add_argument("--tq-depths", default="4096,8192,16384")
    p.add_argument("--tq-depths-sweep", default="0,10,20,30,40,50,60,70,80,90,100")
    p.add_argument("--tq-num-distractors", type=int, default=3)
    p.add_argument("--tq-value-counts", default="2,4,8")
    p.add_argument("--tq-port", type=int, default=8090)
    p.add_argument("--tq-server-timeout", type=int, default=180)
    p.add_argument("--tq-query-timeout", type=int, default=600)
    p.add_argument("--tq-timeout-sec", type=int, default=8 * 3600)

    # Compaction args
    p.add_argument(
        "--hf-model",
        default=os.environ.get("HF_MODEL_PATH", "Qwen/Qwen3-4B"),
        help="HF hub id or local path for the compaction-side model. "
             "Defaults to $HF_MODEL_PATH if set, otherwise 'Qwen/Qwen3-4B'.",
    )
    p.add_argument(
        "--uv-cache-dir",
        default=os.environ.get("UV_CACHE_DIR"),
        help="Forwarded as UV_CACHE_DIR to compaction subprocesses. "
             "Defaults to $UV_CACHE_DIR (unset => inherit parent env).",
    )
    p.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME"),
        help="Forwarded as HF_HOME to compaction subprocesses. "
             "Defaults to $HF_HOME (unset => inherit parent env).",
    )
    p.add_argument("--compaction-dataset", default="quality")
    p.add_argument("--compaction-n-articles", type=int, default=10)
    p.add_argument("--compaction-n-questions-per-article", type=int, default=10)
    p.add_argument("--compaction-target-size", type=float, default=0.1)
    p.add_argument(
        "--compaction-target-sizes",
        default=None,
        help="CSV target-size sweep for compaction (e.g. 0.05,0.1,0.2).",
    )
    p.add_argument(
        "--compaction-methods",
        default=DEFAULT_COMPACTION_METHODS,
        help=(
            "CSV list of compaction methods to run in a single evaluator "
            "invocation (alongside 'original'). Accepts aliases "
            f"({', '.join(sorted(COMPACTION_METHOD_ALIASES))}) or raw "
            f"algorithm-config keys. Default: {DEFAULT_COMPACTION_METHODS}."
        ),
    )
    p.add_argument(
        "--compaction-algorithm-config",
        default="key-selection",
        help=(
            "Name of configs/algorithms/*.py file to load. 'key-selection' "
            "exposes matched-hyperparameter AM, random_subset, random_vector, "
            "truncate, and OMP configs for ceteris-paribus comparison."
        ),
    )
    p.add_argument("--compaction-query-config", default="repeat")
    p.add_argument(
        "--compaction-query-configs",
        default=None,
        help="CSV query-config sweep for compaction (e.g. repeat,self-study).",
    )
    p.add_argument(
        "--compaction-num-runs",
        type=int,
        default=1,
        help="Number of repeated runs per compaction configuration (seed increments by +1 per run).",
    )
    p.add_argument(
        "--compaction-compute-stats",
        type=int,
        default=0,
        choices=[0, 1],
        help="Forwarded to --compute-stats in run_qa_evaluation.",
    )
    p.add_argument(
        "--compaction-compute-perplexity",
        type=int,
        default=1,
        choices=[0, 1],
        help="Forwarded to --compute-perplexity in run_qa_evaluation.",
    )
    p.add_argument("--compaction-precomputed-budget-path", default=None)
    p.add_argument(
        "--compaction-compare-budgets",
        action="store_true",
        help="Run each compaction config twice: uniform and precomputed budgets.",
    )
    p.add_argument("--max-ratio-per-head", type=float, default=1.0)
    p.add_argument(
        "--compaction-bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples for QA accuracy confidence intervals (0 disables).",
    )
    p.add_argument(
        "--compaction-ci-confidence",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals in QA evaluation.",
    )
    p.add_argument("--compaction-timeout-sec", type=int, default=8 * 3600)
    return p.parse_args()


def require_tq_args(args: argparse.Namespace) -> None:
    if not args.llama_dir:
        raise ValueError("--llama-dir is required for TQ modes")
    if not args.gguf_model:
        raise ValueError("--gguf-model is required for TQ modes")


def require_compaction_args(args: argparse.Namespace) -> None:
    if not args.hf_model:
        raise ValueError(
            "--hf-model is required for compaction modes. "
            "Pass a HF hub id or local path, or set $HF_MODEL_PATH."
        )


def apply_shared_niah_override(args: argparse.Namespace) -> None:
    """Point both benchmark sides at the same RULER NIAH task.

    When --shared-niah-ruler-config is set, the compaction side runs on that
    dataset and the TQ+ side runs its depth sweep at the matching context
    length. Existing --compaction-dataset and --tq-depths values are replaced
    (with a note printed so the user can see what changed).
    """
    name = args.shared_niah_ruler_config
    if not name:
        return
    context_length = parse_ruler_context_length(name)
    if args.compaction_dataset != name:
        print(
            f"[shared-niah] overriding --compaction-dataset "
            f"{args.compaction_dataset!r} -> {name!r}"
        )
    args.compaction_dataset = name
    depths_str = str(context_length)
    if args.tq_depths != depths_str:
        print(
            f"[shared-niah] overriding --tq-depths {args.tq_depths!r} -> "
            f"{depths_str!r} (parsed from {name!r})"
        )
    args.tq_depths = depths_str


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    """Clamp sample sizes and sweeps to a minimal validation run.

    Smoke mode is for sanity-checking the runner plumbing on a local GPU
    before spending H100 time. It unconditionally overrides the fields
    listed below (with a note so nothing silently disappears).
    """
    if not args.smoke:
        return
    overrides = {
        "compaction_n_articles": 2,
        "compaction_n_questions_per_article": 3,
        "compaction_target_size": 0.1,
        "compaction_target_sizes": "0.1",
        "compaction_bootstrap_samples": 0,
        "compaction_compute_perplexity": 0,
        "compaction_num_runs": 1,
        "tq_depths": args.tq_depths if args.shared_niah_ruler_config else "4096",
        "tq_depths_sweep": "0,50,100",
    }
    for field, new_value in overrides.items():
        current = getattr(args, field)
        if current != new_value:
            print(f"[smoke] {field}: {current!r} -> {new_value!r}")
        setattr(args, field, new_value)


def main() -> int:
    args = parse_args()
    apply_shared_niah_override(args)
    apply_smoke_overrides(args)
    run_dir = (Path(args.run_root) / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "all":
        requested = [x.strip() for x in args.all_order.split(",") if x.strip()]
        valid = {"tq", "compaction", "both"}
        if not requested:
            raise ValueError("--all-order cannot be empty")
        unknown = [m for m in requested if m not in valid]
        if unknown:
            raise ValueError(f"Invalid mode(s) in --all-order: {unknown}")
        mode_sequence = requested
    else:
        mode_sequence = [args.mode]

    base_env = os.environ.copy()
    if args.seed is not None:
        base_env["PYTHONHASHSEED"] = str(args.seed)
    if args.deterministic_cuda:
        # Required by CUDA libraries for deterministic GEMM kernels.
        base_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    plan = []
    for mode_name in mode_sequence:
        scenario_dir = run_dir if len(mode_sequence) == 1 else (run_dir / f"scenario_{mode_name}")
        scenario_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[JobSpec] = []
        if mode_name in {"tq", "both"}:
            require_tq_args(args)
            jobs.append(build_tq_job(args, scenario_dir, base_env))
        if mode_name in {"compaction", "both"}:
            require_compaction_args(args)
            compaction_variants = build_compaction_variants(args)
            if mode_name == "both" and len(compaction_variants) != 1:
                raise ValueError(
                    "mode=both supports exactly one compaction configuration. "
                    "Use mode=compaction for sweeps over target sizes/query configs/budget modes/runs."
                )
            for variant_idx, variant in enumerate(compaction_variants):
                jobs.append(
                    build_compaction_job(
                        args,
                        scenario_dir,
                        base_env,
                        variant_idx=variant_idx,
                        variant=variant,
                    )
                )
        plan.append((mode_name, scenario_dir, jobs))

    metadata = {
        "started_utc": utc_now_iso(),
        "args": vars(args),
        "run_dir": str(run_dir),
        "mode_sequence": mode_sequence,
        "plan": [
            {
                "mode": mode_name,
                "scenario_dir": str(scenario_dir),
                "jobs": [
                    {
                        "name": j.name,
                        "cmd": j.cmd,
                        "cwd": str(j.cwd),
                        "timeout_sec": j.timeout_sec,
                    }
                    for j in jobs
                ],
            }
            for mode_name, scenario_dir, jobs in plan
        ],
    }
    write_json(run_dir / "metadata.json", metadata)

    print(f"Run dir: {run_dir}")
    for mode_name, scenario_dir, jobs in plan:
        print(f"\n=== Scenario: {mode_name} ===")
        print(f"scenario_dir: {scenario_dir}")
        for j in jobs:
            print(f"\n[{j.name}]")
            print(format_cmd(j.cmd))

    if args.dry_run:
        print("\nDry-run only; exiting.")
        return 0

    scenario_summaries = []
    for mode_name, scenario_dir, jobs in plan:
        print(f"\n>>> Running scenario: {mode_name}")
        if mode_name == "both" and args.both_strategy == "parallel":
            results = run_jobs_parallel(jobs, scenario_dir)
        else:
            results = [run_job(j, scenario_dir) for j in jobs]

        scenario_summary = {
            "mode": mode_name,
            "scenario_dir": str(scenario_dir),
            "results": [asdict(r) for r in results],
            "all_success": all(r.success for r in results),
        }
        scenario_summaries.append(scenario_summary)
        write_json(scenario_dir / "summary.json", scenario_summary)

    summary = {
        "finished_utc": utc_now_iso(),
        "mode_sequence": mode_sequence,
        "scenarios": scenario_summaries,
        "all_success": all(s["all_success"] for s in scenario_summaries),
    }
    write_json(run_dir / "summary.json", summary)

    print("\n=== Summary ===")
    for s in scenario_summaries:
        status = "OK" if s["all_success"] else "FAIL"
        print(f"{status:4} scenario={s['mode']} summary={Path(s['scenario_dir']) / 'summary.json'}")
        for r in s["results"]:
            jstatus = "OK" if r["success"] else "FAIL"
            print(
                f"  {jstatus:4} {r['name']:24} rc={r['return_code']:4} "
                f"time={r['duration_sec']:.1f}s timeout={r['timed_out']}"
            )
    print(f"\nSummary JSON: {run_dir / 'summary.json'}")
    return 0 if summary["all_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
