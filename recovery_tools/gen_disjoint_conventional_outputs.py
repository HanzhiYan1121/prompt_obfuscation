#!/usr/bin/env python3
"""Generate original-prompt (conventional) outputs on the DISJOINT eval probes.

For every held-out prompt, the recovery evidence uses test queries [0:6].
Many prompts only have cached conventional outputs for those first queries, so
the query-disjoint utility eval ([6:16]) silently dropped them. This script
fills the gap: it runs the true original prompt (target_original_prompt) as the
system prompt on test_data queries [offset:offset+n] for every held-out prompt
and writes result_dir/<output_name>. The disjoint utility eval then reads this
file (input + output) so ALL held-out prompts get an evidence-disjoint score.

Victim/original model = Llama-3.1-8B-Instruct 4-bit, same generation config as
the original conventional_output cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.utils import TextDataset, create_collate_fn
from src.model import Model
from src.output_generation import generate_model_responses
from src.utils import set_seed
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parents[1]


def generate_with_prompt(model, system_prompt, queries, batch_size, gen_args):
    dataset = TextDataset(queries)
    collate_fn = create_collate_fn(tokenizer=model.tokenizer, system_prompt=system_prompt or "")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    raw = generate_model_responses(model, dataloader, gen_args)
    out = []
    for group in raw:
        out.append("\n".join(str(x) for x in group) if isinstance(group, list) else str(group))
    return out


def resolve_under_base(p: Path) -> Path:
    return p if p.is_absolute() else (BASE / p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_prefix", default="sixcat200")
    ap.add_argument("--model_tag", default="qwen3_32b")
    ap.add_argument("--categories", default="role_persona task_instruction format_spec constraints context_knowledge tool_protocol")
    ap.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--offset", type=int, default=6)
    ap.add_argument("--n_queries", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--output_name", default="conventional_output_disjoint.json")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    gen_args = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": 1,
    }

    set_seed(args.seed)
    model = Model(args.model_name, "4bit")

    for cat in args.categories.split():
        inputs_path = BASE / f"results/{args.result_prefix}_{cat}_structured_recovery/recovery_inputs.json"
        split_path = BASE / f"results/{args.result_prefix}_{cat}_structured_{args.model_tag}_lora/split.json"
        if not inputs_path.exists() or not split_path.exists():
            print(f"[{cat}] SKIP missing inputs/split")
            continue
        records = {r["id"]: r for r in json.loads(inputs_path.read_text(encoding="utf-8"))}
        test_ids = json.loads(split_path.read_text(encoding="utf-8"))["test_ids"]

        done = 0
        for k, rid in enumerate(test_ids, 1):
            rec = records.get(rid)
            if rec is None:
                print(f"[{cat} {k}/{len(test_ids)}] missing record {rid}")
                continue
            result_dir = resolve_under_base(Path(rec["result_dir"]))
            out_path = result_dir / args.output_name
            if out_path.exists() and not args.overwrite:
                print(f"[{cat} {k}/{len(test_ids)}] skip existing {rid}")
                done += 1
                continue
            td_path = result_dir / "prepared_data" / "test_data.json"
            if not td_path.exists():
                print(f"[{cat} {k}/{len(test_ids)}] missing test_data {rid}")
                continue
            test_data = json.loads(td_path.read_text(encoding="utf-8"))
            queries = [str(x) for x in test_data[args.offset : args.offset + args.n_queries]]
            if not queries:
                print(f"[{cat} {k}/{len(test_ids)}] no disjoint queries {rid}")
                continue
            system_prompt = rec.get("target_original_prompt") or ""
            outputs = generate_with_prompt(model, system_prompt, queries, args.batch_size, gen_args)
            payload = {
                "input": queries,
                "output": outputs,
                "generation_config": gen_args,
                "seed": args.seed,
                "system_prompt": system_prompt,
                "source": "target_original_prompt",
                "query_offset": args.offset,
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            done += 1
            print(f"[{cat} {k}/{len(test_ids)}] wrote {len(outputs)} outputs for {rid}")
        print(f"[{cat}] DONE {done}/{len(test_ids)}")


if __name__ == "__main__":
    main()
