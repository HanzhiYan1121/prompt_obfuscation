"""
Decode recovered prompts for all 8 styles and analyze keyword recovery.
"""
import torch
from transformers import AutoTokenizer
import json, os

TOKENIZER_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_CACHE = "/scratch/Hanzhi/huggingface"
BASE = "precomputed_results"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, cache_dir=HF_CACHE)

# Style definitions: keywords the attacker would need to recover core meaning
STYLES = {
    "pirate":            {"dir": f"{BASE}/soft_pirate_truthfulqa_full",    "keywords": ["pirate","accent","arr","matey"]},
    "robot":             {"dir": f"{BASE}/soft_robot_truthfulqa_full",     "keywords": ["robot","mechanical","beep","processing"]},
    "manga_miko":        {"dir": f"{BASE}/soft_manga_miko_case_study",     "keywords": ["manga","miko","anime","girlfriend","playful","affectionate"]},
    "southern_usa":      {"dir": f"{BASE}/soft_southern_usa_new",          "keywords": ["southern","usa","accent","y'all","dixie"]},
    "poem":              {"dir": f"{BASE}/soft_poem_new",                  "keywords": ["poem","rhyme","verse","stanza","poetry"]},
    "stutter":           {"dir": f"{BASE}/soft_stutter_new",               "keywords": ["stutter","stammer","repetition"]},
    "david_attenborough":{"dir": f"{BASE}/soft_david_attenborough_new",    "keywords": ["attenborough","david","documentary","animal","nature","breathtaking"]},
    "friendly":          {"dir": f"{BASE}/soft_friendly_new",              "keywords": ["friendly","casual","relaxed","friend","informal"]},
}

PAD_ID = 128256  # Llama3 pad token

def decode_pt(path):
    if not os.path.exists(path):
        return None
    ids = torch.load(path, map_location="cpu", weights_only=False).tolist()
    valid = [i for i in ids if i != PAD_ID and i < tokenizer.vocab_size]
    text = tokenizer.decode(ids, skip_special_tokens=False).strip()
    words = tokenizer.decode(valid, skip_special_tokens=True).strip()
    return {"full": text, "words": words, "ids": ids}

def keyword_match(text, keywords):
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]

print("=" * 72)
print(f"{'Style':<20} {'Method':<6} {'Recovered Text':<35} {'Keywords Found'}")
print("=" * 72)

results = {}
for style, info in STYLES.items():
    d = info["dir"]
    kws = info["keywords"]
    row = {"style": style, "keywords": kws, "soft": None, "hard": None}

    for method, fname in [("soft", "best_deobf_soft.pt"), ("hard", "best_deobf_hard.pt")]:
        # Try new filename first, fall back to original filenames
        pt_path = os.path.join(d, fname)
        if not os.path.exists(pt_path):
            alt = "best_deobf_sys_prompt_soft.pt" if method == "soft" else "best_deobf_sys_prompt_hard.pt"
            pt_path = os.path.join(d, alt)

        decoded = decode_pt(pt_path)
        if decoded is None:
            print(f"  {style:<20} {method:<6} {'(not found)':<35} —")
            continue

        found = keyword_match(decoded["words"], kws)
        row[method] = {"text": decoded["words"], "found": found}

        short = decoded["words"][:33] + ".." if len(decoded["words"]) > 35 else decoded["words"]
        found_str = ", ".join(found) if found else "❌ none"
        print(f"  {style:<20} {method:<6} {short:<35} {found_str}")

    results[style] = row

print("=" * 72)

# Summary
print("\n=== KEYWORD RECOVERY SUMMARY ===")
print(f"{'Style':<20} {'Core keyword':<20} {'Soft':>6} {'Hard':>6} {'Verdict'}")
print("-" * 65)
for style, row in results.items():
    core_kw = row["keywords"][0] if row["keywords"] else "?"
    soft_ok = "✅" if row["soft"] and row["soft"]["found"] else "❌"
    hard_ok = "✅" if row["hard"] and row["hard"]["found"] else "❌"
    if (row["soft"] and row["soft"]["found"]) or (row["hard"] and row["hard"]["found"]):
        verdict = "⚠️  CORE MEANING RECOVERABLE"
    else:
        verdict = "✅ Protected"
    print(f"  {style:<20} {core_kw:<20} {soft_ok:>6} {hard_ok:>6}  {verdict}")

with open("all_styles_keyword_analysis.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nSaved: all_styles_keyword_analysis.json")
