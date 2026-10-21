# Component 4: Multi-Dimensional Adaptive Pedagogical Policy Learning
> **SLIIT Research Project (J26-DS-333)** — Bridging Real-Time Cognitive Sensing with Generative AI Tutoring

---

## 📌 Overview

**MD-AP2L (Multi-Dimensional Adaptive Pedagogical Policy Learning)** serves as the central intelligent decision-making engine (Component 4) of an adaptive tutoring research system. Traditional tutoring systems rely on rigid rule trees or unconstrained generative AI models that lack awareness of the student's real-time mental and affective state.

MD-AP2L bridges real-time neuro-affective sensing (EEG and Computer Vision from Components 1 & 2) with Large Language Models (LLMs) by formulating pedagogical strategy and prompt adaptation as a **hierarchical contextual bandit problem (Disjoint LinUCB)** with closed-loop online learning and deterministic safety guardrails.

---

## 🎯 Specific Objectives (SO)

* **SO1 — Multimodal Context Vector Formulation:** Encodes 5 continuous cognitive states (`attention`, `fatigue`, `confusion`, `readiness`, `workload`) and 6 learner profile traits (`style`, `interest`, `goal`, `prior`, `motivation`, `emotion`) into a standardized **23-dimensional context vector** $x_t \in \mathbb{R}^{23}$.
* **SO2 — Empirical Policy Algorithm Benchmarking:** Rigorously compares 4 candidate algorithms (**LinUCB**, **TOPSIS**, **BKT**, **Q-Learning**) across 10 random seeds on a 2,200-sample dataset using a balanced multi-attribute utility score ($0.6 \cdot \text{Accuracy} + 0.3 \cdot \text{Gain} + 0.1 \cdot \text{Speed}$).
* **SO3 — Two-Stage Hierarchical Bandit Architecture:**
  * **Bandit 1 (Strategy):** Selects 1 of 7 pedagogical strategies (`simplify`, `analogy`, `worked_example`, `hint`, `quiz`, `visual`, `break`).
  * **Bandit 2 (Prompt Template):** Selects 1 of 6 linguistic prompt templates (`Direct`, `Step-by-Step`, `Analogy`, `Question`, `Visual`, `Simplified`) conditioning on $[x_t \,\|\, \text{one\_hot}(s^*)] \in \mathbb{R}^{30}$.
  * **Safety Guardrails:** Two-way deterministic rules protecting learning momentum (Fatigue $< 0.40$) and preventing cognitive burnout (Fatigue $\ge 0.70$).
* **SO4 — Closed-Loop Online Adaptation & Persistence:** Adapts policy parameters online via Ridge regression updates ($A \leftarrow A + x x^T, b \leftarrow b + g x$) using session-based post-test feedback, serialized to `results/policy_state.pkl`.
* **SO5 — Ablation Benchmarking & Statistical Validation:** Evaluates full multimodal context against `cognitive_only`, `personal_only`, `fixed_rules`, and `random` baselines with Paired $t$-tests, Wilcoxon signed-rank tests, Cohen’s $d$ effect sizes, and System Usability Scale (SUS) analysis.

---

## 🏛️ System Architecture

The end-to-end pipeline operates in 5 modular stages:
1. **Perception Layer:** Ingests live EEG cognitive indices (Component 1), Computer Vision affect metrics (Component 2), and learner questionnaire traits.
2. **Context Engine (`student_profile.py`):** Converts inputs into the normalized 23-dim feature representation.
3. **Policy & Safety Engine (`policy_service.py` & `expert_rules.py`):** Disjoint LinUCB selects strategy and prompt framing, filtered through two-way safety guardrails.
4. **Tutoring & Multi-Backend Dispatcher (`llm/`):** Constructs constrained system prompts dispatched to local fine-tuned **Llama 3.2 3B** via Ollama (or cloud fallbacks via Groq / Gemini).
5. **Dual Evaluation Signal (`schemas.py` & `app.py`):** Captures **Hake’s Normalized Learning Gain** ($g$) and **EEG Workload Delta** ($\Delta W$) to drive online policy updates.

---

## 📂 Repository Structure

```
Component 4/
├── architecture_diagram.png         # High-resolution (300 DPI) system architecture diagram
├── generate_architecture_diagram.py # Matplotlib generation script for the architecture diagram
├── app.py                           # Full interactive Streamlit adaptive tutor interface
├── api.py                           # FastAPI REST microservice (/decide, /feedback, /trace)
├── policy_service.py                # Hierarchical two-stage LinUCB engine with persistence
├── student_profile.py               # SO1: 23-dimensional context vector encoder
├── expert_rules.py                  # Ground-truth expert rules & two-way safety guardrails
├── reward_simulator.py              # Action-dependent simulated reward environment
├── comparison.py                    # SO2: 4-algorithm benchmark across 10 seeds
├── plot_comparison.py               # Visual analytics generator (bars, boxplots, radar, heatmap)
├── schemas.py                       # Pydantic models, data contracts, and evaluation schemas
├── config.py                        # Centralized .env configuration loader
├── requirements.txt                 # Python project dependencies
│
├── algorithms/                      # Policy learning algorithms
│   ├── base.py                      # Abstract PolicyAlgorithm base class
│   ├── linucb.py                    # Disjoint LinUCB with Ridge regression
│   ├── topsis.py                    # TOPSIS multi-criteria decision ranker
│   ├── bkt.py                       # Bayesian Knowledge Tracing with strategy mapping
│   ├── qlearning.py                 # Discretized tabular Q-Learning
│   └── baselines.py                 # Ablation baselines (fixed rules, random, cog-only, trait-only)
│
├── data_generation/                 # Synthetic data generation & validation
│   ├── policy_dataset.py            # Generates 2,200 simulated policy training samples
│   ├── llm_dataset.py               # Generates 1,200 QLoRA instruction-tuning JSONL pairs
│   └── validate_llm_dataset.py      # Rule-based validation for LLM instruction pairs
│
├── evaluation/                      # Empirical benchmarks & statistical testing
│   ├── benchmark.py                 # Multi-episode simulation benchmark against baselines
│   ├── stats.py                     # Paired t-test, Wilcoxon, Cohen's d, and 95% CI
│   ├── sus.py                       # 10-item System Usability Scale (SUS) scorer
│   └── user_study_analysis.py       # Live user-study log analyzer
│
├── llm/                             # LLM integration & fine-tuning
│   ├── client.py                    # Multi-provider unified client (Ollama / Groq / Gemini)
│   ├── prompts.py                   # Context-injected pedagogical prompt builder
│   ├── ollama_client.py             # Local Ollama client (Llama 3.2 3B)
│   ├── groq_client.py               # Cloud fallback client (Groq llama-3.1-8b)
│   ├── gemini_client.py             # Cloud fallback client (Google Gemini 2.0 Flash)
│   ├── eval_llm.py                  # Strategy compliance verification script
│   └── LLM_finetuning.ipynb         # Google Colab Unsloth QLoRA fine-tuning notebook
│
├── scripts/                         # Objective-specific demonstration scripts
│   ├── run_all.py                   # Master test pipeline
│   ├── so1_profile_demo.py          # SO1 context encoding demo
│   ├── so2_comparison_demo.py       # SO2 4-algorithm comparison demo
│   ├── so3_bandits_demo.py          # SO3 two-stage bandit decision demo
│   ├── so4_online_learning_demo.py  # SO4 online policy adaptation demo
│   └── so5_benchmark_demo.py        # SO5 ablation benchmark & t-test demo
│
└── tests/                           # Unit tests
    ├── test_profile.py              # Validates 23-dim shape and one-hot constraints
    ├── test_algorithms.py           # Validates algorithm initialization & select/update
    ├── test_reward_simulator.py     # Confirms action-dependent reward sensitivity
    └── test_stats.py                # Verifies statistical testing routines
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository and enter the directory
cd "Component 4"

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configure Backend Credentials
Copy `.env.example` to `.env` and configure your preferred LLM provider:
```bash
cp .env.example .env
```
Inside `.env`:
* `LLM_BACKEND=ollama` *(Default / Recommended for offline, private deployment)*
* `LLM_BACKEND=groq` *(Fast free-tier cloud fallback)*
* `LLM_BACKEND=gemini` *(Secondary free-tier cloud fallback)*

If using Ollama, ensure your model is running:
```bash
ollama run llama3.2:3b
```

### 3. Generate Datasets (Simulated)
```bash
python -m data_generation.policy_dataset
python -m data_generation.llm_dataset
python -m data_generation.validate_llm_dataset
```

### 4. Run the 4-Algorithm Benchmark
Train and evaluate LinUCB, TOPSIS, BKT, and Q-Learning over 10 random seeds:
```bash
python -m comparison
```
*Outputs: `results/comparison_raw.csv`, `results/comparison_summary.csv`, and `results/comparison_winner.json`.*

To render the publication figures:
```bash
python plot_comparison.py
# -> Generates 4 figures in results/plots/
```

### 5. Run Verification & Ablation Demos
```bash
python scripts/run_all.py
```
Or run unit tests:
```bash
pytest tests/
```

### 6. Serve the REST API
Launch the FastAPI microservice for integration with external components:
```bash
uvicorn api:app --reload --port 8000
```
Interactive API docs available at: `http://localhost:8000/docs`

### 7. Launch the Interactive Tutor UI
```bash
streamlit run app.py
```

---

## 📊 Scientific Equations & Peer-Reviewed Standards

Every scoring formula and metric implemented in this codebase is backed by established academic standards:

| Metric / Formulation | Mathematical Definition | Seminal Academic Source | Field |
|---|---|---|---|
| **Normalized Learning Gain ($g$)** | $g = \frac{\text{Post} - \text{Pre}}{100 - \text{Pre}}$ | Hake, R. R. (1998), *Am. J. Phys.* | Educational Data Mining (EDM) |
| **Objective Cognitive Load Delta ($\Delta W$)** | $\Delta W = W_{\text{end}} - W_{\text{baseline}}$ | Sweller (1988); Antonenko et al. (2010), *Educ. Psychol. Rev.* | Cognitive Load Theory / EEG |
| **Disjoint LinUCB Formulation** | $\text{UCB}(a) = \hat{\theta}_a^T x + \alpha \sqrt{x^T A_a^{-1} x}$ | Li, Chu, Langford, & Schapire (2010), *ACM WWW* | Contextual Multi-Armed Bandits |
| **Bayesian Knowledge Tracing (BKT)** | $P(L_{t+1}) = P(L_t \mid \text{Obs}) + (1 - P(L_t \mid \text{Obs})) P(T)$ | Corbett, A. T., & Anderson, J. R. (1994), *UMUAI* | Intelligent Tutoring Systems |
| **TOPSIS Closeness Coefficient** | $CC_i = \frac{d_i^-}{d_i^+ + d_i^-}$ | Hwang, C. L., & Yoon, K. (1981), *Springer* | Multi-Criteria Decision Analysis |
| **System Usability Scale (SUS)** | $2.5 \times [\sum(r_{\text{odd}}-1) + \sum(5-r_{\text{even}})]$ | Brooke, J. (1996); Bangor et al. (2008), *IJHCI* | Human-Computer Interaction |
| **Effect Size (Cohen's $d$)** | $d = \frac{\bar{d}}{s_d}$ (Paired Differences) | Cohen, J. (1988), *Lawrence Erlbaum* | Inferential Statistics |
| **Composite Evaluation Metric** | $0.6 \cdot \text{Acc} + 0.3 \cdot \text{Gain} + 0.1 \cdot \left(1 - \frac{\text{Lat}}{50\text{ms}}\right)$ | Keeney & Raiffa (1993), *MAUT*; ISO/IEC 25010 | Systems Engineering |

---

## 🛡️ Research Integrity Rules

This project strictly adheres to academic research integrity:
1. **No Hardcoded Winner Strings:** The winning policy algorithm (`linucb`) is strictly selected in `comparison.py` based on measured performance metrics across 10 random seeds.
2. **Transparent Labeling:** All synthetic outputs, mock streams, and simulated environments are explicitly labeled `[SIMULATED]`.
3. **Action-Dependent Rewards:** The reward simulator calculates feedback directly based on whether the chosen action matches the student's latent state and applies realistic mismatch penalties.
4. **Resilient LLM Dispatch:** The system uses true API and local model dispatches via `.env`; no fake or hardcoded response fallbacks are permitted.