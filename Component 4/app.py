
import streamlit as st

from schemas import (CognitiveState, StudentTraits, LearningStyle, Interest,
                     Goal, Prior, Motivation, Emotion, EvaluationSignal)
from policy_service import PolicyService
from llm.prompts import build_messages
from llm.client import chat, LLMUnavailable
from comparison import run as run_comparison
from config import Config

try:
    from student_profile import PRIOR_SCALAR
except ImportError:
    PRIOR_SCALAR = {"Beginner": 0.0, "Intermediate": 0.5, "Advanced": 1.0}

"""Beginner -> 30, Intermediate -> 55, Advanced -> 80."""
def pre_score_from_prior(prior_value: str) -> float:
    return 30.0 + PRIOR_SCALAR[prior_value] * 50.0


# ============================================================
# TOPIC QUESTION BANK (used to build the post-test)
# ============================================================
TOPIC_KEYWORDS = {
    "dna":            ["dna", "deoxyribonucleic", "nucleotide", "base pair", "double helix"],
    "rna":            ["rna", "ribonucleic", "mrna", "trna", "transcription", "translation"],
    "mitosis":        ["mitosis", "cell division", "prophase", "metaphase", "anaphase", "telophase"],
    "meiosis":        ["meiosis", "gamete", "crossing over", "haploid"],
    "photosynthesis": ["photosynthesis", "chloroplast", "calvin", "light reaction"],
    "respiration":    ["respiration", "glycolysis", "krebs", "atp", "mitochondria"],
    "blood":          ["blood", "red blood", "rbc", "hemoglobin", "circulation",
                       "white blood", "platelet"],
    "protein":        ["protein", "amino acid", "ribosome", "polypeptide"],
    "enzyme":         ["enzyme", "catalyst", "substrate", "active site"],
}

TOPIC_QUESTION_BANK = {
    "dna": {
        "name": "DNA",
        "questions": [
            {"q": "What is the main function of DNA?",
             "options": ["Store genetic information", "Produce energy",
                         "Build cell walls", "Transport oxygen"],
             "correct": "Store genetic information"},
            {"q": "During which phase of the cell cycle does DNA replication occur?",
             "options": ["Prophase", "S phase", "Anaphase", "Telophase"],
             "correct": "S phase"},
            {"q": "Which base pairs with Adenine in DNA?",
             "options": ["Guanine", "Cytosine", "Thymine", "Uracil"],
             "correct": "Thymine"},
        ],
    },
    "rna": {
        "name": "RNA",
        "questions": [
            {"q": "What sugar is found in RNA?",
             "options": ["Deoxyribose", "Ribose", "Glucose", "Fructose"],
             "correct": "Ribose"},
            {"q": "Which base is found in RNA but NOT in DNA?",
             "options": ["Adenine", "Thymine", "Uracil", "Guanine"],
             "correct": "Uracil"},
            {"q": "What is the main role of mRNA?",
             "options": ["Carry amino acids", "Carry genetic message from DNA to ribosome",
                         "Form the ribosome", "Store energy"],
             "correct": "Carry genetic message from DNA to ribosome"},
        ],
    },
    "mitosis": {
        "name": "Mitosis",
        "questions": [
            {"q": "What is the main result of mitosis?",
             "options": ["Two identical cells", "Four different cells",
                         "One cell", "No cells"],
             "correct": "Two identical cells"},
            {"q": "During which phase do chromosomes align at the cell's center?",
             "options": ["Prophase", "Metaphase", "Anaphase", "Telophase"],
             "correct": "Metaphase"},
            {"q": "What is the correct order of mitosis stages?",
             "options": ["Prophase, Metaphase, Anaphase, Telophase",
                         "Metaphase, Prophase, Telophase, Anaphase",
                         "Anaphase, Metaphase, Prophase, Telophase",
                         "Telophase, Anaphase, Metaphase, Prophase"],
             "correct": "Prophase, Metaphase, Anaphase, Telophase"},
        ],
    },
    "meiosis": {
        "name": "Meiosis",
        "questions": [
            {"q": "How many cells does meiosis produce?",
             "options": ["Two", "Three", "Four", "Eight"],
             "correct": "Four"},
            {"q": "What is the main purpose of meiosis?",
             "options": ["Growth", "Repair", "Produce gametes", "Produce energy"],
             "correct": "Produce gametes"},
            {"q": "What is 'crossing over'?",
             "options": ["Chromosomes condensing", "DNA replicating",
                         "Homologous chromosomes exchanging segments", "Cells dividing"],
             "correct": "Homologous chromosomes exchanging segments"},
        ],
    },
    "photosynthesis": {
        "name": "Photosynthesis",
        "questions": [
            {"q": "Where does photosynthesis occur?",
             "options": ["Mitochondria", "Chloroplast", "Nucleus", "Ribosome"],
             "correct": "Chloroplast"},
            {"q": "What gas is released during photosynthesis?",
             "options": ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"],
             "correct": "Oxygen"},
            {"q": "What are the main inputs of photosynthesis?",
             "options": ["Glucose and oxygen", "Carbon dioxide and water",
                         "ATP and glucose", "Nitrogen and water"],
             "correct": "Carbon dioxide and water"},
        ],
    },
    "respiration": {
        "name": "Cellular respiration",
        "questions": [
            {"q": "What is the main product of cellular respiration?",
             "options": ["Glucose", "ATP", "Oxygen", "DNA"],
             "correct": "ATP"},
            {"q": "Where does the Krebs cycle occur?",
             "options": ["Cytoplasm", "Nucleus", "Mitochondrial matrix", "Ribosome"],
             "correct": "Mitochondrial matrix"},
            {"q": "What gas is consumed during cellular respiration?",
             "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"],
             "correct": "Oxygen"},
        ],
    },
    "blood": {
        "name": "Red blood cells and circulation",
        "questions": [
            {"q": "What is the main function of red blood cells?",
             "options": ["Transport oxygen", "Produce hormones",
                         "Fight infection", "Digest food"],
             "correct": "Transport oxygen"},
            {"q": "What molecule inside red blood cells carries oxygen?",
             "options": ["Hemoglobin", "Insulin", "Collagen", "Keratin"],
             "correct": "Hemoglobin"},
            {"q": "What is the approximate lifespan of a red blood cell?",
             "options": ["120 days", "10 days", "1 year", "24 hours"],
             "correct": "120 days"},
        ],
    },
    "protein": {
        "name": "Protein synthesis",
        "questions": [
            {"q": "What monomers make up proteins?",
             "options": ["Fatty acids", "Amino acids", "Nucleotides", "Monosaccharides"],
             "correct": "Amino acids"},
            {"q": "Where does translation occur?",
             "options": ["Nucleus", "Ribosome", "Mitochondria", "Golgi apparatus"],
             "correct": "Ribosome"},
            {"q": "Which molecule carries amino acids to the ribosome?",
             "options": ["mRNA", "tRNA", "rRNA", "DNA"],
             "correct": "tRNA"},
        ],
    },
    "enzyme": {
        "name": "Enzymes",
        "questions": [
            {"q": "What does an enzyme do to a chemical reaction?",
             "options": ["Slows it down", "Speeds it up",
                         "Stops it", "Reverses it"],
             "correct": "Speeds it up"},
            {"q": "What is the region where the substrate binds called?",
             "options": ["Allosteric site", "Active site",
                         "Binding pocket", "Receptor"],
             "correct": "Active site"},
            {"q": "What happens to most enzymes at very high temperatures?",
             "options": ["They speed up", "They denature",
                         "They double", "They stay the same"],
             "correct": "They denature"},
        ],
    },
    "general": {
        "name": "General biology",
        "questions": [
            {"q": "What is the basic unit of life?",
             "options": ["Atom", "Molecule", "Cell", "Organ"],
             "correct": "Cell"},
            {"q": "What organelle produces most of the cell's ATP?",
             "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi"],
             "correct": "Mitochondria"},
            {"q": "What is the process by which cells make proteins called?",
             "options": ["Photosynthesis", "Respiration", "Protein synthesis", "Replication"],
             "correct": "Protein synthesis"},
        ],
    },
}


def detect_topic(messages) -> str:
    """Scan user messages and return the best-matching topic key."""
    user_text = " ".join(
        m["content"].lower() for m in messages if m.get("role") == "user"
    )
    if not user_text.strip():
        return "general"
    scores = {
        topic: sum(1 for kw in kws if kw in user_text)
        for topic, kws in TOPIC_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title=Config.APP_TITLE, page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #080815 0%, #0f0f23 50%, #130f22 100%);
        color: #e0e0ff;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #080815 100%);
        border-right: 1px solid rgba(124, 58, 237, 0.25);
    }
    h1, h2, h3 { color: #c4b5fd !important; }
    h1 {
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    [data-testid="stMetric"] {
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 10px;
        padding: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px;
        width: 100%;
        font-weight: 600;
    }
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        margin-bottom: 10px;
    }
    .strategy-chip {
        display: inline-block; background: rgba(96, 165, 250, 0.18);
        color: #93c5fd; padding: 3px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600; margin: 2px 4px 2px 0;
    }
    .prompt-chip {
        display: inline-block; background: rgba(52, 211, 153, 0.18);
        color: #6ee7b7; padding: 3px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600; margin: 2px 4px 2px 0;
    }
    .algo-chip {
        display: inline-block; background: rgba(168, 85, 247, 0.18);
        color: #d8b4fe; padding: 3px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600; margin: 2px 4px 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "svc" not in st.session_state:
    st.session_state.svc = PolicyService()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "traits" not in st.session_state:
    st.session_state.traits = None
if "pre_score" not in st.session_state:
    st.session_state.pre_score = None
if "posttest_open" not in st.session_state:
    st.session_state.posttest_open = False
if "posttest_questions" not in st.session_state:
    st.session_state.posttest_questions = None
if "posttest_topic" not in st.session_state:
    st.session_state.posttest_topic = None
if "last_decision" not in st.session_state:
    st.session_state.last_decision = None
if "last_state" not in st.session_state:
    st.session_state.last_state = None
if "comparison_done" not in st.session_state:
    st.session_state.comparison_done = False
if "winner_name" not in st.session_state:
    st.session_state.winner_name = None
if "interactions" not in st.session_state:
    st.session_state.interactions = 0

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 MD-AP2L")
    st.caption("Multi-Dimensional Adaptive Pedagogical Policy Learning · J26-DS-333")
    st.caption(f"`{Config.summary()}`")
    st.divider()

    # 1. Profile
    st.subheader("1️⃣ Learner Profile")
    with st.form("questionnaire"):
        style = st.selectbox("How do you learn best?", [e.value for e in LearningStyle])
        interest = st.selectbox("What interests you most?", [e.value for e in Interest])
        goal = st.selectbox("Main goal for learning?", [e.value for e in Goal])
        prior = st.selectbox("How much do you already know?", [e.value for e in Prior])
        motivation = st.selectbox("Motivation right now?", [e.value for e in Motivation])
        emotion = st.selectbox("How are you feeling now?", [e.value for e in Emotion])
        saved = st.form_submit_button("💾 Save profile")

    if saved:
        st.session_state.traits = StudentTraits(
            style=LearningStyle(style), interest=Interest(interest),
            goal=Goal(goal), prior=Prior(prior),
            motivation=Motivation(motivation), emotion=Emotion(emotion))
        st.session_state.pre_score = pre_score_from_prior(prior)
        st.success("Profile saved.")
        st.info(
            f"📊 Prior knowledge: **{prior}** "
            f"(scalar = {PRIOR_SCALAR[prior]:.2f}) → "
            f"estimated pre-test score: **{st.session_state.pre_score:.0f}%**"
        )

    if st.session_state.traits is None:
        st.warning("⚠️ Save your profile before chatting.")
    else:
        st.caption(f"Pre-test baseline: **{st.session_state.pre_score:.0f}%** (from PRIOR_SCALAR)")

    st.divider()

    # 2. Cognitive state
    st.subheader("2️⃣ Cognitive State")
    st.caption("**SIMULATED** — from Components 1 & 2 (EEG + CV) in the full system.")
    attention = st.slider("🎯 Attention", 0.0, 1.0, 0.65, 0.05)
    fatigue   = st.slider("😴 Fatigue",   0.0, 1.0, 0.70, 0.05)
    confusion = st.slider("🤔 Confusion", 0.0, 1.0, 0.80, 0.05)
    readiness = st.slider("📚 Readiness", 0.0, 1.0, 0.40, 0.05)
    workload  = st.slider("📊 Workload",  0.0, 1.0, 0.55, 0.05)

    st.divider()

    # 3. Session status
    st.subheader("3️⃣ Session Status")
    st.metric("💬 Interactions", st.session_state.interactions)

    current_topic = detect_topic(st.session_state.messages)
    if current_topic != "general":
        st.caption(f"Detected topic: **{TOPIC_QUESTION_BANK[current_topic]['name']}**")

    if st.session_state.interactions > 0 and not st.session_state.posttest_open:
        st.markdown("**Finished studying?**")
        if st.button("✅ Take the post-test"):
            topic = detect_topic(st.session_state.messages)
            st.session_state.posttest_topic = topic
            st.session_state.posttest_questions = TOPIC_QUESTION_BANK[topic]["questions"]
            st.session_state.posttest_open = True
            st.rerun()
    elif st.session_state.posttest_open:
        st.info("Post-test is open — see the main panel.")

    st.divider()

    # 4. Algorithm comparison
    st.subheader("4️⃣ Algorithm Comparison")
    if not st.session_state.comparison_done:
        if st.button("▶ Run 4-Algorithm Comparison"):
            with st.spinner("Running comparison (10 seeds × 400 contexts)..."):
                result = run_comparison(save_dir="results")
                st.session_state.comparison_done = True
                st.session_state.winner_name = result["winner"]
            st.success(f"Winner (computed): {result['winner']}")
    else:
        st.success(f"Winner (computed): {st.session_state.winner_name}")

    st.divider()
    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_decision = None
        st.session_state.posttest_open = False
        st.session_state.posttest_questions = None
        st.session_state.interactions = 0
        st.rerun()

# ============================================================
# MAIN
# ============================================================
st.title("🧬 MD-AP2L — Adaptive Tutor")
st.caption("Chat with a tutor that adapts its teaching strategy to *how you feel* and *who you are*.")

if st.session_state.traits is None:
    st.info("👈 Complete the questionnaire in the sidebar to begin.")
    st.stop()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            m = msg["meta"]
            st.markdown(
                f"<span class='algo-chip'>🤖 {m['algorithm']}</span>"
                f"<span class='strategy-chip'>🎯 #{m['strategy']}</span>"
                f"<span class='prompt-chip'>📝 {m['prompt']}</span>",
                unsafe_allow_html=True)
            with st.expander("Why this decision?"):
                st.json(m["rationale"])

# Chat input (unless post-test is open)
if not st.session_state.posttest_open:
    user_input = st.chat_input("Ask a question, e.g. 'Explain red blood cells'")

    if user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build state + policy decision
        state = CognitiveState(
            attention=attention, fatigue=fatigue,
            confusion=confusion, readiness=readiness, workload=workload)
        st.session_state.last_state = state

        with st.chat_message("assistant"):
            with st.spinner("Selecting strategy + prompt..."):
                decision = st.session_state.svc.decide(state, st.session_state.traits)
            st.session_state.last_decision = decision

            # Build messages WITH HISTORY
            # history = everything before the current user message
            history = st.session_state.messages[:-1]
            messages = build_messages(
                state, st.session_state.traits,
                decision["strategy"], decision["prompt"],
                topic=user_input,
                history=history,
            )

            try:
                with st.spinner("LLM is generating…"):
                    content, backend_used = chat(messages)
                st.caption(f"Backend used: **{backend_used}**")
            except LLMUnavailable as e:
                content = None
                st.error(
                    "**LLM UNAVAILABLE**\n\n"
                    "Fix:\n"
                    "1. Ensure Ollama is running (`ollama serve`) OR\n"
                    "2. Set `GROQ_API_KEY` in `.env`\n\n"
                    f"Details: {e}")

            if content is not None:
                st.markdown(
                    f"<span class='algo-chip'>🤖 {st.session_state.winner_name or 'linucb'}</span>"
                    f"<span class='strategy-chip'>🎯 #{decision['strategy']}</span>"
                    f"<span class='prompt-chip'>📝 {decision['prompt']}</span>",
                    unsafe_allow_html=True)
                st.markdown(content)
                with st.expander("Why this decision?"):
                    st.json(decision["rationale"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": content,
                    "meta": {
                        "algorithm": st.session_state.winner_name or "linucb",
                        "strategy": decision["strategy"],
                        "prompt": decision["prompt"],
                        "rationale": decision["rationale"],
                    }})
                st.session_state.interactions += 1
                st.rerun()

    if st.session_state.interactions > 0:
        st.caption("💡 Tip: Keep asking questions. When you're done, click **✅ Take the post-test** in the sidebar.")
    else:
        st.caption("💡 Tip: Try 'Explain red blood cells' or 'What is DNA?'")

# Post-test
if st.session_state.posttest_open:
    st.divider()
    topic_key = st.session_state.posttest_topic or "general"
    topic_name = TOPIC_QUESTION_BANK[topic_key]["name"]
    st.subheader(f"📊 Post-Test — {topic_name}")
    st.caption(f"Pre-test baseline (from PRIOR_SCALAR): **{st.session_state.pre_score:.0f}%**")

    questions = st.session_state.posttest_questions or TOPIC_QUESTION_BANK["general"]["questions"]

    with st.form("posttest"):
        answers = []
        for i, item in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {item['q']}")
            ans = st.radio(f"Q{i+1}", item["options"], key=f"pt_q{i}",
                           label_visibility="collapsed")
            answers.append(ans)
            st.markdown("")

        c1, c2 = st.columns(2)
        with c1:
            submitted = st.form_submit_button("✅ Submit post-test")
        with c2:
            skip = st.form_submit_button("⏭ Skip (no update)")

    if submitted:
        correct = sum(1 for i, item in enumerate(questions)
                      if answers[i] == item["correct"])
        post_score = correct / len(questions) * 100

        sig = EvaluationSignal(
            pre_score=st.session_state.pre_score,
            post_score=post_score).compute()

        st.success(
            f"Post-test: **{post_score:.0f}%** ({correct}/{len(questions)}) | "
            f"Learning gain: **{sig.learning_gain:.2f}** | "
            f"Normalized gain: **{sig.normalized_gain:.2f}**"
        )

        d = st.session_state.last_decision
        s = st.session_state.last_state
        if d and s:
            st.session_state.svc.feedback(
                s, st.session_state.traits,
                d["strategy"], d["prompt"], sig.learning_gain)
            st.info("Policy updated — the system will adapt better next time.")

        # Reset post-test state
        st.session_state.posttest_open = False
        st.session_state.posttest_questions = None
        st.session_state.posttest_topic = None
        st.balloons()

    if skip:
        st.session_state.posttest_open = False
        st.session_state.posttest_questions = None
        st.session_state.posttest_topic = None
        st.rerun()

# Trace
with st.expander("🔍 Decision trace (last 10)"):
    if st.session_state.svc and st.session_state.svc.trace:
        st.json(st.session_state.svc.trace[-10:])
    else:
        st.caption("No decisions logged yet.")