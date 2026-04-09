#!/usr/bin/env python3
"""
Parallel obfuscation launcher for all new L1/L2/L3/L4 prompts.

Assigns prompts round-robin to available GPUs, runs obfuscate.py
followed by evaluate_obfuscation.py on each GPU sequentially.
Already-completed runs (results_dir exists and has best_candidate.pt) are skipped.

Usage:
    python launch_all_experiments.py
    python launch_all_experiments.py --gpus 0 1 2 3 4 5
    python launch_all_experiments.py --dry-run       # print plan, no execution
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from prompt_configs import PROMPTS

PYTHON  = "/scratch/Hanzhi/paper reproduce/venv/bin/python3"
WORKDIR = Path(__file__).parent

# Obfuscation hyperparams (matches our complex_L* setup)
OBF_PARAMS = dict(
    obfuscation_method = "soft",
    dataset_size       = 200,
    optimizer_iter     = 5,
    batch_size         = 2,
    obf_sys_prompt_len = 10,
    output_token_count = 15,
    window_size        = 5,
    lr                 = 0.01,
    seed               = 42,
)

EVAL_PARAMS = dict(
    eval_batch_size = 32,
)


def is_done(name: str) -> bool:
    return (WORKDIR / "precomputed_results" / name / "best_candidate.pt").exists()


def run_one(prompt: dict, gpu: int) -> tuple[str, bool, str]:
    """Run obfuscate + evaluate for one prompt on a given GPU. Returns (name, ok, msg)."""
    name   = prompt["name"]
    outdir = WORKDIR / "precomputed_results" / name
    logdir = WORKDIR / "logs"
    logdir.mkdir(exist_ok=True)
    log    = logdir / f"{name}.log"
    env    = {"CUDA_VISIBLE_DEVICES": str(gpu)}

    # Build obfuscate command
    obf_cmd = [
        PYTHON, "obfuscate.py",
        "--system_prompt", prompt["system_prompt"],
        "--dataset_name",  prompt["dataset"],
        "--output_dir",    str(outdir),
    ]
    for k, v in OBF_PARAMS.items():
        obf_cmd += [f"--{k}", str(v)]

    # Build evaluate command
    eval_cmd = [
        PYTHON, "evaluate_obfuscation.py",
        "--results_dir", str(outdir),
    ]
    for k, v in EVAL_PARAMS.items():
        eval_cmd += [f"--{k}", str(v)]

    import os
    full_env = {**os.environ, **env}

    with open(log, "w") as lf:
        lf.write(f"=== {name} on GPU {gpu} ===\n")
        lf.flush()

        for phase, cmd in [("OBFUSCATE", obf_cmd), ("EVALUATE", eval_cmd)]:
            lf.write(f"\n--- {phase} ---\n{' '.join(shlex.quote(c) for c in cmd)}\n\n")
            lf.flush()
            t0 = time.time()
            ret = subprocess.run(
                cmd, cwd=str(WORKDIR), env=full_env,
                stdout=lf, stderr=subprocess.STDOUT
            )
            elapsed = time.time() - t0
            lf.write(f"\n[{phase} exit={ret.returncode} in {elapsed/60:.1f} min]\n")
            lf.flush()

            if ret.returncode != 0:
                return name, False, f"{phase} failed (exit {ret.returncode})"

    return name, True, f"GPU {gpu}"


def run_gpu_queue(args_tuple):
    """Top-level function (picklable) that runs a queue of prompts on one GPU."""
    gpu, prompts = args_tuple
    gpu_results = []
    for p in prompts:
        print(f"  [GPU {gpu}] START {p['name']}", flush=True)
        name, ok, msg = run_one(p, gpu)
        status = "OK" if ok else "FAIL"
        print(f"  [GPU {gpu}] {status}  {name}  ({msg})", flush=True)
        gpu_results.append((name, ok, msg))
    return gpu_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--levels", nargs="+", default=None,
                        help="Filter to specific levels, e.g. --levels L2 L3")
    args = parser.parse_args()

    todo = [p for p in PROMPTS if not is_done(p["name"])]
    if args.levels:
        todo = [p for p in todo if p["level"] in args.levels]

    done_already = [p for p in PROMPTS if is_done(p["name"])]
    print(f"Already done : {len(done_already)}")
    print(f"To run       : {len(todo)}")
    print(f"GPUs         : {args.gpus}")
    print(f"Est. time    : ~{len(todo) / len(args.gpus) * 1.5:.1f}h wall-clock\n")

    if not todo:
        print("Nothing to do.")
        return

    assignments = [(p, args.gpus[i % len(args.gpus)]) for i, p in enumerate(todo)]

    if args.dry_run:
        print("DRY RUN — plan:")
        for p, gpu in assignments:
            print(f"  GPU {gpu}  {p['name']:<25} ({p['level']}, {p['dataset']})")
        return

    print("Starting jobs...\n")
    results = {"ok": [], "fail": []}

    from collections import defaultdict
    gpu_jobs: dict[int, list[dict]] = defaultdict(list)
    for p, gpu in assignments:
        gpu_jobs[gpu].append(p)

    with ProcessPoolExecutor(max_workers=len(args.gpus)) as ex:
        futures = {ex.submit(run_gpu_queue, (gpu, prompts)): gpu
                   for gpu, prompts in gpu_jobs.items()}
        for fut in as_completed(futures):
            gpu = futures[fut]
            try:
                for name, ok, msg in fut.result():
                    (results["ok"] if ok else results["fail"]).append(name)
            except Exception as e:
                print(f"  [GPU {gpu}] EXCEPTION: {e}")

    print(f"\n{'='*50}")
    print(f"Done: {len(results['ok'])} ok, {len(results['fail'])} failed")
    if results["fail"]:
        print(f"Failed: {results['fail']}")

    # Save summary
    summary_path = WORKDIR / "results" / "launch_summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
