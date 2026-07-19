#!/usr/bin/env python3
"""Error analysis for six-category 150-prompt recovery experiments."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recovery_tools.eval_recovery_utility import token_f1


ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "results/sixcat150_recovery_error_analysis.md"
OUT_JSON = ROOT / "results/sixcat150_recovery_error_analysis.json"
MANUAL_AUDIT = ROOT / "results/sixcat150_high_utility_manual_audit.json"

CATEGORIES = [
    ("role_persona", "Role/Persona", "role"),
    ("task_instruction", "Task Instruction", "task"),
    ("format_spec", "Format Spec", "format"),
    ("constraints", "Constraints", "constraints"),
    ("context_knowledge", "Context Knowledge", "context"),
    ("tool_protocol", "Tool/Protocol", "tool_protocol"),
]
FIELDS = ["role", "task", "format", "constraints", "context", "tool_protocol"]


def parse_json_object(text: str) -> dict[str, str] | None:
    if not text:
        return None
    start = text.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return {k: str(obj.get(k, "") or "").strip() for k in FIELDS}
        start = text.find("{", start + 1)
    return None


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def short(text: str, limit: int = 420) -> str:
    clean = " ".join((text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def load_index(path: Path, key: str = "id") -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    return {str(r[key]): r for r in records}


def utility_bucket(value: float) -> str:
    if value < 0.30:
        return "<0.30"
    if value < 0.40:
        return "0.30-0.39"
    if value < 0.50:
        return "0.40-0.49"
    return ">=0.50"


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return numerator / (dx * dy) if dx and dy else 0.0


def analyze_category(slug: str, label: str, dominant_field: str) -> dict:
    base = ROOT / f"results/sixcat150_{slug}_structured_qwen14b_lora"
    recovered = load_index(base / "lora_recovered_prompts.json")
    text_eval = load_index(base / "lora_recovery_eval.json")
    utility_eval = load_index(base / "recovery_utility_eval_matched_n6.json")

    task_dir = ROOT / f"results/sixcat150_{slug}_structured_qwen14b_lora_taskmatched_v3"
    task_utility = load_index(task_dir / "recovery_utility_eval_taskmatched.json")

    rows = []
    for rec_id, rec in recovered.items():
        if rec_id not in text_eval or rec_id not in utility_eval:
            continue
        parsed_recovered = parse_json_object(rec.get("recovered_prompt", ""))
        parsed_target = parse_json_object(rec.get("target", ""))
        target_original = rec.get("target_original_prompt", "")
        text_score = float(text_eval[rec_id]["token_f1"])
        utility_score = float(utility_eval[rec_id]["mean_pair_token_f1"])
        task_utility_score = (
            float(task_utility[rec_id]["mean_pair_token_f1"])
            if rec_id in task_utility else None
        )

        errors = []
        field_scores = {f: 0.0 for f in FIELDS}
        missing_fields = []
        spurious_fields = []
        duplicate_fields = []

        if parsed_recovered is None:
            errors.append("json_parse_failure")
            parsed_recovered = {f: "" for f in FIELDS}
        if parsed_target is None:
            parsed_target = {f: "" for f in FIELDS}

        for field in FIELDS:
            rv = parsed_recovered[field]
            tv = parsed_target[field]
            field_scores[field] = token_f1(rv, tv) if rv and tv else 0.0
            if tv and not rv:
                missing_fields.append(field)
            if rv and not tv:
                spurious_fields.append(field)

        value_to_fields = defaultdict(list)
        for field, value in parsed_recovered.items():
            norm = normalize(value)
            if norm:
                value_to_fields[norm].append(field)
        for fields in value_to_fields.values():
            if len(fields) > 1:
                duplicate_fields.extend(fields)

        if missing_fields:
            errors.append("missing_target_fields")
        if spurious_fields:
            errors.append("spurious_fields")
        if duplicate_fields:
            errors.append("cross_field_duplication")

        target_dom = parsed_target.get(dominant_field, "")
        recovered_dom = parsed_recovered.get(dominant_field, "")
        dominant_f1 = field_scores[dominant_field]
        if target_dom and not recovered_dom:
            errors.append("dominant_field_missing")
        elif target_dom and dominant_f1 < 0.20:
            errors.append("dominant_field_low_overlap")

        combined_recovered = " ".join(parsed_recovered.values()).lower()
        if any(x in combined_recovered for x in ["helpful assistant", "assist the user", "general assistant"]):
            errors.append("generic_assistant_collapse")
        if text_score >= 0.40 and utility_score < 0.30:
            errors.append("high_text_low_utility")
        if text_score < 0.30 and utility_score >= 0.50:
            errors.append("low_text_high_utility")
        if utility_score < 0.30:
            errors.append("functional_failure")
        elif utility_score >= 0.50:
            errors.append("functional_success")

        rows.append(
            {
                "id": rec_id,
                "category": label,
                "dominant_field": dominant_field,
                "text_f1": text_score,
                "utility_f1": utility_score,
                "taskmatched_utility_f1": task_utility_score,
                "utility_bucket": utility_bucket(utility_score),
                "field_f1": field_scores,
                "dominant_field_f1": dominant_f1,
                "missing_fields": missing_fields,
                "spurious_fields": spurious_fields,
                "duplicate_fields": sorted(set(duplicate_fields)),
                "errors": sorted(set(errors)),
                "original_prompt": target_original,
                "target_structured": parsed_target,
                "recovered_structured": parsed_recovered,
                "recovered_prompt_text": utility_eval[rec_id].get("recovered_prompt_text", ""),
            }
        )

    bucket_counts = Counter(r["utility_bucket"] for r in rows)
    error_counts = Counter(e for r in rows for e in r["errors"])
    field_means = {
        f: sum(r["field_f1"][f] for r in rows) / len(rows) if rows else 0.0
        for f in FIELDS
    }
    mean_text = sum(r["text_f1"] for r in rows) / len(rows)
    mean_utility = sum(r["utility_f1"] for r in rows) / len(rows)
    text_utility_correlation = pearson(
        [r["text_f1"] for r in rows],
        [r["utility_f1"] for r in rows],
    )
    task_values = [r["taskmatched_utility_f1"] for r in rows if r["taskmatched_utility_f1"] is not None]
    mean_task = sum(task_values) / len(task_values) if task_values else None

    sorted_utility = sorted(rows, key=lambda r: r["utility_f1"])
    misleading = sorted(rows, key=lambda r: r["text_f1"] - r["utility_f1"], reverse=True)
    examples = {
        "lowest": sorted_utility[:2],
        "highest": sorted_utility[-2:][::-1],
        "text_behavior_disagreement": misleading[:2],
    }

    return {
        "slug": slug,
        "category": label,
        "dominant_field": dominant_field,
        "n": len(rows),
        "mean_text_f1": mean_text,
        "mean_utility_f1": mean_utility,
        "text_utility_correlation": text_utility_correlation,
        "mean_taskmatched_utility_f1": mean_task,
        "bucket_counts": dict(bucket_counts),
        "error_counts": dict(error_counts),
        "field_mean_f1": field_means,
        "examples": examples,
        "records": rows,
    }


def render_report(results: list[dict]) -> str:
    manual = json.loads(MANUAL_AUDIT.read_text(encoding="utf-8"))
    manual_by_category = defaultdict(list)
    for row in manual:
        manual_by_category[row["category"]].append(row)
    manual_total = len(manual)
    manual_success = sum(bool(row["plausible_partial_recovery"]) for row in manual)

    lines = [
        "# Six-Category 150-Prompt Recovery Error Analysis",
        "",
        "Primary evaluation: matched original-test probes. Task-matched category probes are included as a stress-test comparison.",
        "",
        "## 1. Score Distribution",
        "",
        "| Category | Text F1 | Matched Utility F1 | Task-matched Utility F1 | Text/Utility r | <0.30 | 0.30-0.39 | 0.40-0.49 | >=0.50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        b = r["bucket_counts"]
        lines.append(
            f"| {r['category']} | {r['mean_text_f1']:.3f} | {r['mean_utility_f1']:.3f} | "
            f"{r['mean_taskmatched_utility_f1']:.3f} | {r['text_utility_correlation']:.3f} | "
            f"{b.get('<0.30', 0)} | {b.get('0.30-0.39', 0)} | "
            f"{b.get('0.40-0.49', 0)} | {b.get('>=0.50', 0)} |"
        )

    lines += [
        "",
        "## 2. Structured Field Recovery",
        "",
        "Mean field-level Token F1 between recovered and target JSON fields:",
        "",
        "| Category | Role | Task | Format | Constraints | Context | Tool/Protocol | Dominant field |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        f = r["field_mean_f1"]
        lines.append(
            f"| {r['category']} | {f['role']:.3f} | {f['task']:.3f} | {f['format']:.3f} | "
            f"{f['constraints']:.3f} | {f['context']:.3f} | {f['tool_protocol']:.3f} | {r['dominant_field']} |"
        )

    error_order = [
        "functional_failure",
        "high_text_low_utility",
        "dominant_field_missing",
        "dominant_field_low_overlap",
        "missing_target_fields",
        "spurious_fields",
        "cross_field_duplication",
        "generic_assistant_collapse",
        "json_parse_failure",
    ]
    lines += [
        "",
        "## 3. Error Taxonomy",
        "",
        "Counts are out of 30 held-out prompts per category.",
        "",
        "| Category | Functional failure | High-text/low-utility | Dominant missing | Dominant low-overlap | Missing fields | Spurious fields | Duplicated fields | Generic collapse | JSON failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        e = r["error_counts"]
        vals = [e.get(name, 0) for name in error_order]
        lines.append(f"| {r['category']} | " + " | ".join(str(x) for x in vals) + " |")

    lines += [
        "",
        "## 4. Representative Cases",
        "",
    ]
    for r in results:
        lines.append(f"### {r['category']}")
        lines.append("")
        for group, title in [
            ("highest", "High Utility"),
            ("lowest", "Low Utility"),
            ("text_behavior_disagreement", "Text/Behavior Disagreement"),
        ]:
            lines.append(f"#### {title}")
            lines.append("")
            for ex in r["examples"][group]:
                lines.append(
                    f"- **{ex['id']}**: Text F1={ex['text_f1']:.3f}, Utility F1={ex['utility_f1']:.3f}, "
                    f"Dominant-field F1={ex['dominant_field_f1']:.3f}; errors={', '.join(ex['errors']) or 'none'}"
                )
                lines.append(f"  - Original: `{short(ex['original_prompt'])}`")
                lines.append(f"  - Recovered: `{short(ex['recovered_prompt_text'])}`")
            lines.append("")

    lines += [
        "## 5. Manual Audit of Utility F1 >= 0.50",
        "",
        f"The strict threshold identifies {manual_total} high-Utility cases. Manual comparison of the original and recovered prompts marks {manual_success}/{manual_total} ({manual_success/manual_total:.1%}) as plausible partial task recovery.",
        "",
        "| Category | Utility>=0.50 cases | Plausible partial recovery | Apparent false positives |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        audited = manual_by_category[r["category"]]
        good = sum(bool(x["plausible_partial_recovery"]) for x in audited)
        lines.append(f"| {r['category']} | {len(audited)} | {good} | {len(audited)-good} |")

    lines += [
        "",
        "Important: Utility Token F1 is still a lexical output-overlap metric. A score above 0.50 can be a false positive when two prompts produce generic or vocabulary-similar answers despite different tasks. The manually audited precision of the threshold is therefore about 65%, not 100%.",
        "",
        "## 6. Main Findings",
        "",
        "1. **The main failure is underspecification, not JSON parsing.** Most recovered outputs are valid structured JSON, but they omit, duplicate, or generalize important fields.",
        "2. **Text overlap does not guarantee behavior recovery.** High-text/low-utility cases show that copying task words can still miss operational rules, label spaces, context, or output constraints.",
        "3. **Role and Task recover broad intent more often than exact rules.** They can reproduce general behavior, but exact personas, label definitions, and edge conditions are frequently lost.",
        "4. **Format and Constraints fail through missing operational details.** A generic statement such as 'use JSON' or 'follow safety rules' does not recover exact fields, ordering, forbidden actions, or exception logic.",
        "5. **Context and Tool prompts often recover the task shell but not the hidden payload.** The model infers that context/tool use is required, while missing the specific inventory, API arguments, call order, or background facts.",
        "6. **Task-matched category probes remain weaker than original-test matched probes.** Category-level probes are still too broad; per-prompt input alignment is the stronger evaluation setting.",
        "7. **Utility F1 >= 0.50 is not sufficient by itself.** Manual audit finds plausible partial recovery in 17/26 high-score cases; nine cases are apparent false positives caused by broad output overlap or generic behavior.",
        "",
        "## 7. Limitations",
        "",
        "- The six-field target is generated by regex heuristics, not human annotation. One original sentence can be copied into several target fields, so field omission and duplication counts partly reflect target-construction noise.",
        "- Utility F1 compares output tokens and does not directly score task correctness, schema validity, rule compliance, grounding, or tool-call accuracy.",
        "- Manual high-score labels are qualitative and should be validated by a second annotator before publication.",
        "- Category assignment is based on the strongest six-dimensional score; some prompts contain multiple task types and may be ambiguous.",
        "",
        "## 8. Recommended Next Changes",
        "",
        "- Add field-specific losses or examples for the dominant category field.",
        "- Penalize cross-field duplication during structured recovery.",
        "- Evaluate exact format keys, constraint rules, and tool arguments with category-specific functional metrics.",
        "- Use per-prompt adaptive probes generated from observed behavior, without exposing the original prompt text.",
        "- Keep matched original-test probes as the primary reference and task-matched probes as a stress test.",
    ]
    return "\n".join(lines)


def main() -> int:
    results = [analyze_category(*cat) for cat in CATEGORIES]
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_report(results), encoding="utf-8")
    print(f"Saved {OUT_MD.relative_to(ROOT)}")
    print(f"Saved {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
