#!/bin/bash
BASE="/scratch/Hanzhi/paper reproduce/prompt_obfuscation/prompt_obfuscation"
PYTHON="/scratch/Hanzhi/paper reproduce/venv/bin/python3"

echo "[$(date '+%H:%M:%S')] Auto-save: running final analysis..."

"$PYTHON" - << 'EOF'
import json, csv
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

all_styles = [
    ("pirate",            "soft_pirate_truthfulqa_full",            800, 10, "baseline"),
    ("robot",             "soft_robot_truthfulqa_full",             800, 10, "baseline"),
    ("poem",              "soft_poem_truthfulqa_full",              800, 10, "baseline"),
    ("stutter",           "soft_stutter_truthfulqa_full",           800, 10, "baseline"),
    ("friendly",          "soft_friendly_truthfulqa_full",          800, 10, "baseline"),
    ("southern_usa",      "soft_southern_usa_truthfulqa_full",      800, 10, "baseline"),
    ("david_attenborough","soft_david_attenborough_truthfulqa_full",800, 10, "baseline"),
    ("manga_miko",        "soft_manga_miko_case_study",             800, 10, "baseline"),
    ("complex_L1",        "complex_L1",                            200,  5, "complex"),
    ("complex_L2",        "complex_L2",                            200,  5, "complex"),
    ("complex_L3",        "complex_L3",                            200,  5, "complex"),
]

rows = []
for name, dirname, ds, iters, etype in all_styles:
    base = Path(f"precomputed_results/{dirname}")
    sf = base / "best_candidate_scores.json"
    pf = base / "params.json"
    if not sf.exists() or not pf.exists():
        print(f"SKIP {name}: missing files")
        continue
    s = json.loads(sf.read_text())
    p = json.loads(pf.read_text())
    prompt = p['system_prompt'].replace('<|pad|>','').strip()
    n_tok = len(tok.encode(prompt))
    ratio = round(10 / n_tok, 4)
    rows.append({
        "name": name, "type": etype,
        "prompt_tokens": n_tok, "obf_tokens": 10,
        "compression_ratio": ratio,
        "dataset_size": ds, "optimizer_iter": iters,
        "cosine_similarity": round(s.get('cosine_similarity',0), 4),
        "bertscore":         round(s.get('bertscore',0), 4),
        "rouge1":            round(s.get('rouge1',0), 4),
        "sacrebleu":         round(s.get('sacrebleu',0), 4),
        "prompt_preview": prompt[:120],
    })

Path("results").mkdir(exist_ok=True)
with open("results/complexity_vs_utility.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
with open("results/complexity_vs_utility.json","w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print("\n=== FINAL RESULTS ===")
print(f"{'Name':<22} {'type':<10} {'tokens':<8} {'ratio':<8} {'cosine':<8} {'bertscore':<11} {'rouge1':<8} sacrebleu")
print("="*95)
for r in rows:
    print(f"{r['name']:<22} {r['type']:<10} {r['prompt_tokens']:<8} {r['compression_ratio']:<8} {r['cosine_similarity']:<8} {r['bertscore']:<11} {r['rouge1']:<8} {r['sacrebleu']}")
print("\nSaved to results/complexity_vs_utility.csv and .json")
EOF
