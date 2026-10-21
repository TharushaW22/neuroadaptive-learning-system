"""Generate the 1000+ instruction-output JSONL for QLoRA fine-tuning."""
from __future__ import annotations
import os, json, itertools
import numpy as np
from schemas import Strategy, PromptTemplate

SEED = 42
N = 1200
OUT = "data/llm_dataset.jsonl"

TOPICS = ["mitosis", "meiosis", "DNA replication", "photosynthesis",
          "cellular respiration", "protein synthesis"]


def _context_phrase(rng) -> str:
    return (f"attention={rng.uniform(0,1):.2f}, fatigue={rng.uniform(0,1):.2f}, "
            f"confusion={rng.uniform(0,1):.2f}, readiness={rng.uniform(0,1):.2f}, "
            f"workload={rng.uniform(0,1):.2f}")


def _output(strategy: Strategy, template: PromptTemplate, topic: str, rng) -> str:
    if strategy is Strategy.SIMPLIFY:
        return ("Mitosis is cell division producing two identical cells in four "
                "stages: prophase, metaphase, anaphase, telophase (PMAT).")
    if strategy is Strategy.ANALOGY:
        return (f"Think of {topic} like a sports team making substitutions: the "
                f"captain (cell) copies the playbook (DNA), the coach organises "
                f"the players, and two identical teams are ready to play.")
    if strategy is Strategy.WORKED_EXAMPLE:
        return ("Step 1: chromosomes condense. Step 2: chromosomes align. "
                "Step 3: chromosomes separate. Step 4: two cells form.")
    if strategy is Strategy.HINT:
        return "Hint: focus on the first letter of each stage — P, M, A, T."
    if strategy is Strategy.QUIZ:
        return ("Question 1: which stage comes first? "
                "Question 2: what separates during anaphase? "
                "Question 3: name the four stages in order.")
    if strategy is Strategy.VISUAL:
        return ("Picture four panels: (1) threads condensing, (2) threads lined up, "
                "(3) threads pulling apart, (4) two nuclei forming.")
    if strategy is Strategy.BREAK:
        return ("Take a two-minute break and recall: PMAT — prophase, metaphase, "
                "anaphase, telophase.")
    raise ValueError(strategy)


def build(n: int = N, seed: int = SEED) -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rng = np.random.default_rng(seed)
    combos = list(itertools.product(list(Strategy), list(PromptTemplate)))
    rows = []
    for i in range(n):
        strategy = Strategy(combos[i % len(combos)][0].value)
        template = PromptTemplate(combos[i % len(combos)][1].value)
        topic = TOPICS[int(rng.integers(0, len(TOPICS)))]
        ctx = _context_phrase(rng)
        instruction = (f"You are a biology tutor. Student context ({ctx}). "
                       f"Use strategy '{strategy.value}' and prompt style "
                       f"'{template.value}'. Topic: {topic}.")
        out = _output(strategy, template, topic, rng)
        rows.append({"instruction": instruction, "output": out,
                     "strategy": strategy.value, "prompt_template": template.value})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {OUT}  n={len(rows)}")


if __name__ == "__main__":
    build()