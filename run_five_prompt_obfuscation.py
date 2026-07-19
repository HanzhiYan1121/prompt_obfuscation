#!/usr/bin/env python3
"""Run soft prompt obfuscation for a scored system-prompt JSON file.

The script is intentionally a thin runner around the existing obfuscate.py and
evaluate_obfuscation.py entrypoints. It keeps one result directory per prompt
and skips completed stages so interrupted runs can be resumed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


BASE = Path(__file__).resolve().parent
DEFAULT_PYTHON = "/scratch/Hanzhi/paper reproduce/venv/bin/python3"


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug or "prompt"


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        completed = subprocess.run(
            cmd,
            cwd=BASE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with code {completed.returncode}: {log_path}")


def save_last_candidate(result_dir: Path, reason: str) -> None:
    candidates_path = result_dir / "obfuscated_system_prompt_list.pt"
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)
    candidates = torch.load(candidates_path, map_location="cpu", weights_only=True)
    if not candidates:
        raise ValueError(f"no candidates in {candidates_path}")
    torch.save(candidates[-1].cpu(), result_dir / "best_candidate.pt")
    selection = {
        "selection_method": "last_training_candidate",
        "candidate_index": len(candidates) - 1,
        "num_candidates": len(candidates),
        "reason": reason,
    }
    (result_dir / "best_candidate_selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )


def process_prompt(prompt: dict, gpu: str, config: dict[str, str | int]) -> dict:
    prompt_id = prompt["id"]
    slug = slugify(prompt_id)
    out_prefix = str(config["out_prefix"])
    out_dir = BASE / "precomputed_results" / f"{out_prefix}_{slug}_tok{config['tokens']}"
    log_dir = BASE / "logs"
    train_log = log_dir / f"{out_prefix}_{slug}_tok{config['tokens']}.log"
    eval_log = log_dir / f"{out_prefix}_{slug}_tok{config['tokens']}_eval.log"

    candidates_path = out_dir / "obfuscated_system_prompt_list.pt"
    best_candidate_path = out_dir / "best_candidate.pt"
    if out_dir.exists() and not candidates_path.exists() and not best_candidate_path.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "prompt_metadata.json"
    if not metadata_path.exists():
        metadata = {
            "id": prompt.get("id"),
            "prompt_label": prompt.get("prompt_label"),
            "paper_title": prompt.get("paper_title"),
            "scores": prompt.get("scores"),
            "source_file": str(config["prompts_file"]),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    prompt_text_path = out_dir / "original_prompt.txt"
    if not prompt_text_path.exists():
        prompt_text_path.write_text(prompt["prompt_text"], encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    python_bin = str(config["python"])
    if not candidates_path.exists():
        cmd = [
            python_bin,
            "-u",
            "obfuscate.py",
            "--system_prompt",
            prompt["prompt_text"],
            "--dataset_name",
            str(config["dataset"]),
            "--obfuscation_method",
            "soft",
            "--dataset_size",
            str(config["dataset_size"]),
            "--batch_size",
            str(config["batch_size"]),
            "--obf_sys_prompt_len",
            str(config["tokens"]),
            "--output_token_count",
            str(config["output_token_count"]),
            "--window_size",
            str(config["window_size"]),
            "--optimizer_iter",
            str(config["optimizer_iter"]),
            "--lr",
            str(config["lr"]),
            "--seed",
            str(config["seed"]),
            "--output_dir",
            str(out_dir.relative_to(BASE)),
        ]
        run_command(cmd, train_log, env)

    if not candidates_path.exists():
        raise FileNotFoundError(f"obfuscation did not produce candidates: {candidates_path}")

    should_eval = bool(config["run_eval"]) and (
        bool(config["force_eval"]) or not best_candidate_path.exists()
    )
    if should_eval:
        cmd = [
            python_bin,
            "-u",
            "evaluate_obfuscation.py",
            "--results_dir",
            str(out_dir.relative_to(BASE)),
            "--eval_batch_size",
            str(config["eval_batch_size"]),
            "--max_new_tokens",
            str(config["eval_max_new_tokens"]),
            "--num_return_sequences",
            str(config["eval_num_return_sequences"]),
            "--metrics",
            "sacrebleu",
            "rougeL",
            "cosine_similarity",
        ]
        try:
            run_command(cmd, eval_log, env)
            selection = {
                "selection_method": "evaluate_obfuscation",
                "metrics": ["sacrebleu", "rougeL", "cosine_similarity"],
            }
            (out_dir / "best_candidate_selection.json").write_text(
                json.dumps(selection, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            if bool(config["fallback_last_on_eval_fail"]):
                save_last_candidate(out_dir, f"evaluate_obfuscation failed: {exc}")
            else:
                raise
    elif not bool(config["run_eval"]) and not best_candidate_path.exists():
        save_last_candidate(out_dir, "RUN_EVAL=0")

    return {
        "id": prompt_id,
        "gpu": gpu,
        "result_dir": str(out_dir.relative_to(BASE)),
        "has_candidates": candidates_path.exists(),
        "has_best_candidate": best_candidate_path.exists(),
    }


def main() -> int:
    prompts_file = Path(env_str("PROMPTS_FILE", "system prompt/five_prompts_scored.json"))
    if not prompts_file.is_absolute():
        prompts_file = BASE / prompts_file
    prompts = json.loads(prompts_file.read_text(encoding="utf-8"))
    gpu_list = env_str("GPUS", "0 1 2 3 4").split()
    if not gpu_list:
        raise ValueError("GPUS must contain at least one GPU id")

    config: dict[str, str | int | bool] = {
        "python": env_str("PYTHON", DEFAULT_PYTHON),
        "out_prefix": env_str("OUT_PREFIX", "fiveprompt_recovery"),
        "prompts_file": str(prompts_file.relative_to(BASE)),
        "dataset": env_str("DATASET", "truthfulqa"),
        "dataset_size": env_int("DATASET_SIZE", 80),
        "batch_size": env_int("BATCH_SIZE", 4),
        "tokens": env_int("TOKENS", 10),
        "output_token_count": env_int("OUTPUT_TOKEN_COUNT", 15),
        "window_size": env_int("WINDOW_SIZE", 5),
        "optimizer_iter": env_int("OPT_ITERS", 3),
        "lr": env_str("LR", "0.01"),
        "seed": env_int("SEED", 42),
        "run_eval": env_str("RUN_EVAL", "1") == "1",
        "force_eval": env_str("FORCE_EVAL", "0") == "1",
        "eval_batch_size": env_int("EVAL_BATCH_SIZE", 4),
        "eval_max_new_tokens": env_int("EVAL_MAX_NEW_TOKENS", 80),
        "eval_num_return_sequences": env_int("EVAL_NUM_RETURN_SEQUENCES", 1),
        "fallback_last_on_eval_fail": env_str("FALLBACK_LAST_ON_EVAL_FAIL", "1") == "1",
    }

    selected = prompts[: env_int("N_PROMPTS", len(prompts))]
    print("Prompt obfuscation config:")
    print(json.dumps({**config, "gpus": gpu_list, "n_prompts": len(selected)}, indent=2))

    results: list[dict] = []
    max_workers = min(len(gpu_list), len(selected))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for idx, prompt in enumerate(selected):
            gpu = gpu_list[idx % len(gpu_list)]
            futures.append(pool.submit(process_prompt, prompt, gpu, config))
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, indent=2))

    summary_path = BASE / "results" / f"{config['out_prefix']}_summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps(sorted(results, key=lambda r: r["id"]), indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
