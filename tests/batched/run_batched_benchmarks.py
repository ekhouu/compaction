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
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


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
    return " ".join(cmd)


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


def build_compaction_job(args: argparse.Namespace, run_dir: Path, env: dict[str, str]) -> JobSpec:
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
        args.compaction_method,
        "--dataset-name",
        args.compaction_dataset,
        "--n-articles",
        str(args.compaction_n_articles),
        "--target-size",
        str(args.compaction_target_size),
        "--compute-stats",
        "0",
        "--compute-perplexity",
        "1",
        "--device",
        "cuda",
        "--model-name",
        args.hf_model,
        "--query-config",
        args.compaction_query_config,
        "--log-dir",
        str((run_dir / "compaction_eval_logs").resolve()),
        "--name",
        f"batched_{args.compaction_dataset}_t{args.compaction_target_size}",
    ]
    if args.compaction_n_questions_per_article is not None:
        cmd += ["--n-questions-per-article", str(args.compaction_n_questions_per_article)]
    if args.compaction_precomputed_budget_path:
        cmd += ["--precomputed-budget-path", args.compaction_precomputed_budget_path]
        cmd += ["--max-ratio-per-head", str(args.max_ratio_per_head)]

    run_env = env.copy()
    if args.uv_cache_dir:
        run_env["UV_CACHE_DIR"] = args.uv_cache_dir
    if args.hf_home:
        run_env["HF_HOME"] = args.hf_home
        run_env["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.hf_home, "hub")

    return JobSpec(
        name="compaction_benchmark",
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

    p.add_argument("--python-bin", default=sys.executable)

    # TQ+ args
    p.add_argument("--tq-script", default="_external/turboquant_plus/scripts/niah_test.py")
    p.add_argument("--llama-dir", default=None)
    p.add_argument("--gguf-model", default=None)
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
    p.add_argument("--hf-model", default="/home/mikhoiuo/repos/compaction/.hf-models/Qwen3-4B")
    p.add_argument("--uv-cache-dir", default="/tmp/uv-cache")
    p.add_argument("--hf-home", default="/home/mikhoiuo/repos/compaction/.hf-cache")
    p.add_argument("--compaction-dataset", default="quality")
    p.add_argument("--compaction-n-articles", type=int, default=10)
    p.add_argument("--compaction-n-questions-per-article", type=int, default=10)
    p.add_argument("--compaction-target-size", type=float, default=0.1)
    p.add_argument("--compaction-method", default="AM-HighestAttnKeys")
    p.add_argument("--compaction-algorithm-config", default="default")
    p.add_argument("--compaction-query-config", default="repeat")
    p.add_argument("--compaction-precomputed-budget-path", default=None)
    p.add_argument("--max-ratio-per-head", type=float, default=1.0)
    p.add_argument("--compaction-timeout-sec", type=int, default=8 * 3600)
    return p.parse_args()


def require_tq_args(args: argparse.Namespace) -> None:
    if not args.llama_dir:
        raise ValueError("--llama-dir is required for TQ modes")
    if not args.gguf_model:
        raise ValueError("--gguf-model is required for TQ modes")


def main() -> int:
    args = parse_args()
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
    plan = []
    for mode_name in mode_sequence:
        scenario_dir = run_dir if len(mode_sequence) == 1 else (run_dir / f"scenario_{mode_name}")
        scenario_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[JobSpec] = []
        if mode_name in {"tq", "both"}:
            require_tq_args(args)
            jobs.append(build_tq_job(args, scenario_dir, base_env))
        if mode_name in {"compaction", "both"}:
            jobs.append(build_compaction_job(args, scenario_dir, base_env))
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
