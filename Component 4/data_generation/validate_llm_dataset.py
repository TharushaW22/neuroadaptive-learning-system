"""Reject rows that violate per-strategy rules."""
from __future__ import annotations
import json

STRATEGY_RULES = {
    "simplify":       {"max_words": 90,  "must_not": [],              "must": []},
    "analogy":        {"max_words": 120, "must_not": [],              "must": ["like"]},
    "worked_example": {"max_words": 140, "must_not": [],              "must": ["step 1"]},
    "hint":           {"max_words": 60,  "must_not": ["answer", "is "], "must": ["hint"]},
    "quiz":           {"max_words": 120, "must_not": [],              "must": ["question"]},
    "visual":         {"max_words": 120, "must_not": [],              "must": ["picture"]},
    "break":          {"max_words": 60,  "must_not": [],              "must": ["break"]},
}


def ok(row):
    s = row["strategy"]; out = row["output"].lower()
    r = STRATEGY_RULES.get(s)
    if r is None:
        return False, f"unknown strategy {s}"
    if len(out.split()) > r["max_words"]:
        return False, f"too long for {s}"
    for tok in r["must"]:
        if tok not in out:
            return False, f"missing '{tok}' for {s}"
    for tok in r["must_not"]:
        if tok in out:
            return False, f"contains forbidden '{tok}' for {s}"
    return True, ""


def main(path="data/llm_dataset.jsonl"):
    good, bad = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ok_, why = ok(row)
            (good if ok_ else bad).append((row, why))
    print(f"[SIMULATED dataset] kept={len(good)} rejected={len(bad)}")
    with open(path, "w", encoding="utf-8") as f:
        for row, _ in good:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for _, why in bad[:10]:
        print("  rejected:", why)


if __name__ == "__main__":
    main()