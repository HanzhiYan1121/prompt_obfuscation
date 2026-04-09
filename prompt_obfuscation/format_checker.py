#!/usr/bin/env python3
"""
Format Checker for complex prompt levels (L2 / L3).

For each level, applies a set of regex-based rules to every generated response
and computes:
  - Rule Adherence Rate  : mean fraction of rules satisfied (per response → averaged)
  - Strict Pass Rate     : fraction of responses satisfying ALL rules

Usage:
    python format_checker.py --level L2 --results_dir precomputed_results/complex_L1
    python format_checker.py --level L3 --results_dir precomputed_results/complex_L3
    python format_checker.py --level L2 L3 --compare    # compare both in one table
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ── Rule definition ────────────────────────────────────────────────────────────

@dataclass
class Rule:
    name: str
    description: str
    check: Callable[[str], bool]   # True = passes


def _sentences(text: str) -> list[str]:
    """Split text into sentences (simple regex, good enough for format checks)."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def _paragraphs(text: str) -> list[str]:
    """Split into non-empty paragraphs."""
    return [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]


# ── L2 Rules (complex_L1 prompt) ──────────────────────────────────────────────
#
# Prompt: "Always structure your response in exactly three parts:
#   1) A one-sentence direct answer.
#   2) A brief explanation in no more than two sentences.
#   3) One concrete example.
#   Never use bullet points. Never start a sentence with 'I'."

L2_RULES: list[Rule] = [
    Rule(
        name="three_part_structure",
        description="Response has ≥ 3 non-empty paragraphs (three-part structure)",
        check=lambda t: len(_paragraphs(t)) >= 3,
    ),
    Rule(
        name="part1_one_sentence",
        description="First paragraph is a single sentence (≤ 2 as tolerance)",
        check=lambda t: len(_sentences(_paragraphs(t)[0])) <= 2
                        if _paragraphs(t) else False,
    ),
    Rule(
        name="part2_max_two_sentences",
        description="Second paragraph has ≤ 2 sentences",
        check=lambda t: len(_sentences(_paragraphs(t)[1])) <= 2
                        if len(_paragraphs(t)) >= 2 else False,
    ),
    Rule(
        name="has_example",
        description="Response contains a concrete example signal",
        check=lambda t: bool(re.search(
            r'\bfor\s+(?:example|instance)\b'
            r'|\be\.g\.\b'
            r'|\bsuch\s+as\b'
            r'|\bto\s+illustrate\b'
            r'|\bspecifically[,:]'
            r'|\bconcrete(?:ly)?\b',
            t, re.IGNORECASE
        )) or len(_paragraphs(t)) >= 3,
    ),
    Rule(
        name="no_bullet_points",
        description="No bullet-point lines (-, *, •, numbered lists)",
        check=lambda t: not bool(re.search(
            r'(?m)^\s*[-*•]\s|^\s*\d+[.)]\s', t
        )),
    ),
    Rule(
        name="no_sentence_starts_with_I",
        description="No sentence begins with 'I ' (first-person prohibition)",
        check=lambda t: not bool(re.search(
            r'(?:^|(?<=[.!?])\s+)I\s', t
        )),
    ),
]


# ── L3 Rules (complex_L3 prompt) ──────────────────────────────────────────────
#
# Rule 1: Begin with 'Analysis:' + one-sentence summary.
# Rule 2: Exactly three key points starting with -1. / -2. / -3.
# Rule 3: 'Confidence:' section rated Low / Medium / High.
# Rule 4: Never use passive voice.
# Rule 5: Never mention own limitations.
# Rule 6: Total response ≤ 120 words.

_PASSIVE_PATTERN = re.compile(
    r'\b(?:is|are|was|were|be|been|being)\s+'
    r'(?:\w+ly\s+)?'          # optional adverb
    r'\w+ed\b',
    re.IGNORECASE
)

_LIMITATION_PATTERN = re.compile(
    r"\bI\s+(?:cannot|can't|am\s+unable|don't\s+have|lack|am\s+not\s+able)\b"
    r"|\bas\s+an\s+AI\b"
    r"|\bI\s+(?:do\s+not|don't)\s+(?:have\s+(?:access|information|data)|know)\b"
    r"|\bI\s+(?:must\s+)?(?:refuse|decline)\b"
    r"|\bbeyond\s+my\s+(?:knowledge|training|capability)\b",
    re.IGNORECASE
)

L3_RULES: list[Rule] = [
    Rule(
        name="starts_with_analysis",
        description="Response begins with 'Analysis:'",
        check=lambda t: t.lstrip().startswith("Analysis:"),
    ),
    Rule(
        name="exactly_three_dash_points",
        description="Contains exactly -1. / -2. / -3. formatted key points",
        check=lambda t: (
            bool(re.search(r'-1\.', t)) and
            bool(re.search(r'-2\.', t)) and
            bool(re.search(r'-3\.', t)) and
            len(re.findall(r'-\d+\.', t)) == 3
        ),
    ),
    Rule(
        name="has_confidence_section",
        description="Contains 'Confidence:' followed by Low / Medium / High",
        check=lambda t: bool(re.search(
            r'Confidence\s*:\s*(?:Low|Medium|High)', t, re.IGNORECASE
        )),
    ),
    Rule(
        name="no_passive_voice",
        description="No common passive constructions (is/was/were + -ed; approximate)",
        check=lambda t: not bool(_PASSIVE_PATTERN.search(t)),
    ),
    Rule(
        name="no_limitation_phrases",
        description="No self-limitation phrases ('I cannot', 'as an AI', etc.)",
        check=lambda t: not bool(_LIMITATION_PATTERN.search(t)),
    ),
    Rule(
        name="under_120_words",
        description="Total response ≤ 120 words",
        check=lambda t: len(t.split()) <= 120,
    ),
]

LEVELS = {"L2": L2_RULES, "L3": L3_RULES}


# ── Evaluation logic ───────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    name: str
    description: str
    pass_rate: float        # fraction of all responses that pass this rule


@dataclass
class LevelResult:
    level: str
    prompt_type: str        # "conventional" or "obfuscated"
    n_samples: int
    n_sequences: int        # responses per sample (num_return_sequences)
    rule_results: list[RuleResult] = field(default_factory=list)
    rule_adherence_rate: float = 0.0   # mean pass_rate across rules
    strict_pass_rate: float = 0.0      # fraction where ALL rules pass


def evaluate(outputs: list[list[str]], rules: list[Rule]) -> LevelResult:
    """
    outputs: list[n_samples] of list[n_sequences] of str
    Returns aggregated rule scores.
    """
    per_rule_passes = {r.name: [] for r in rules}
    strict_passes = []

    for seqs in outputs:
        for seq in seqs:
            all_pass = True
            for rule in rules:
                passed = rule.check(seq)
                per_rule_passes[rule.name].append(passed)
                all_pass = all_pass and passed
            strict_passes.append(all_pass)

    n_total = sum(len(seqs) for seqs in outputs)

    rule_results = [
        RuleResult(
            name=r.name,
            description=r.description,
            pass_rate=sum(per_rule_passes[r.name]) / n_total,
        )
        for r in rules
    ]

    rar = sum(rr.pass_rate for rr in rule_results) / len(rule_results)
    spr = sum(strict_passes) / n_total

    result = LevelResult(
        level="",
        prompt_type="",
        n_samples=len(outputs),
        n_sequences=len(outputs[0]) if outputs else 0,
        rule_results=rule_results,
        rule_adherence_rate=rar,
        strict_pass_rate=spr,
    )
    return result


def load_outputs(path: Path) -> list[list[str]]:
    with open(path) as f:
        d = json.load(f)
    return d["output"]


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_comparison(level: str, conv: LevelResult, obf: LevelResult):
    rules = LEVELS[level]
    print(f"\n{'='*72}")
    print(f"  {level} Format Compliance  |  "
          f"n_samples={conv.n_samples}, n_seq={conv.n_sequences}")
    print(f"{'='*72}")
    print(f"{'Rule':<35} {'Conv':>8}  {'Obf':>8}  {'Δ':>8}")
    print(f"{'-'*35} {'-'*8}  {'-'*8}  {'-'*8}")

    for rr_c, rr_o in zip(conv.rule_results, obf.rule_results):
        delta = rr_o.pass_rate - rr_c.pass_rate
        flag = " ⚠" if delta < -0.10 else ("  ✓" if delta >= 0 else "  ↓")
        print(f"{rr_c.name:<35} {rr_c.pass_rate:>7.1%}  {rr_o.pass_rate:>7.1%}  "
              f"{delta:>+7.1%}{flag}")

    print(f"{'-'*35} {'-'*8}  {'-'*8}  {'-'*8}")
    delta_rar = obf.rule_adherence_rate - conv.rule_adherence_rate
    delta_spr = obf.strict_pass_rate - conv.strict_pass_rate
    print(f"{'Rule Adherence Rate':<35} {conv.rule_adherence_rate:>7.1%}  "
          f"{obf.rule_adherence_rate:>7.1%}  {delta_rar:>+7.1%}")
    print(f"{'Strict Pass Rate':<35} {conv.strict_pass_rate:>7.1%}  "
          f"{obf.strict_pass_rate:>7.1%}  {delta_spr:>+7.1%}")
    print()

    print("Rule descriptions:")
    for rr in conv.rule_results:
        print(f"  {rr.name}: {rr.description}")


def save_results(level: str, conv: LevelResult, obf: LevelResult,
                 out_path: Path):
    data = {
        "level": level,
        "n_samples": conv.n_samples,
        "n_sequences": conv.n_sequences,
        "rule_adherence_rate": {
            "conventional": conv.rule_adherence_rate,
            "obfuscated": obf.rule_adherence_rate,
            "delta": obf.rule_adherence_rate - conv.rule_adherence_rate,
        },
        "strict_pass_rate": {
            "conventional": conv.strict_pass_rate,
            "obfuscated": obf.strict_pass_rate,
            "delta": obf.strict_pass_rate - conv.strict_pass_rate,
        },
        "rules": [
            {
                "name": rc.name,
                "description": rc.description,
                "conventional": rc.pass_rate,
                "obfuscated": ro.pass_rate,
                "delta": ro.pass_rate - rc.pass_rate,
            }
            for rc, ro in zip(conv.rule_results, obf.rule_results)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", nargs="+", choices=["L2", "L3"], required=True)
    parser.add_argument(
        "--results_dir", nargs="+",
        help="One results_dir per level (same order). Defaults: complex_L1 for L2, complex_L3 for L3."
    )
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    default_dirs = {"L2": "precomputed_results/complex_L1",
                    "L3": "precomputed_results/complex_L3"}

    levels = args.level
    dirs   = args.results_dir if args.results_dir else [default_dirs[l] for l in levels]

    if len(dirs) != len(levels):
        print("Error: --results_dir must match the number of levels.")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}

    for level, results_dir in zip(levels, dirs):
        rdir = Path(results_dir)
        rules = LEVELS[level]

        conv_outputs = load_outputs(rdir / "conventional_output.json")
        obf_outputs  = load_outputs(rdir / "best_candidate_output.json")

        conv = evaluate(conv_outputs, rules)
        obf  = evaluate(obf_outputs,  rules)
        conv.level, conv.prompt_type = level, "conventional"
        obf.level,  obf.prompt_type  = level, "obfuscated"

        print_comparison(level, conv, obf)

        out_path = Path(args.output_dir) / f"format_check_{level}.json"
        save_results(level, conv, obf, out_path)
        all_results[level] = {"conv": conv, "obf": obf}

    # summary if both levels
    if len(levels) == 2:
        print(f"\n{'='*72}")
        print("  Cross-Level Summary")
        print(f"{'='*72}")
        print(f"{'Level':<6} {'Conv RAR':>10} {'Obf RAR':>10} {'ΔRAR':>8} "
              f"{'Conv SPR':>10} {'Obf SPR':>10} {'ΔSPR':>8}")
        print("-"*72)
        for lv, d in all_results.items():
            c, o = d["conv"], d["obf"]
            print(f"{lv:<6} {c.rule_adherence_rate:>10.1%} {o.rule_adherence_rate:>10.1%} "
                  f"{o.rule_adherence_rate - c.rule_adherence_rate:>+8.1%} "
                  f"{c.strict_pass_rate:>10.1%} {o.strict_pass_rate:>10.1%} "
                  f"{o.strict_pass_rate - c.strict_pass_rate:>+8.1%}")


if __name__ == "__main__":
    main()
