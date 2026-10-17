
import json as _json
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

def pre_score_from_prior(prior_value: str) -> float:
    """Beginner -> 30, Intermediate -> 55, Advanced -> 80."""
    return 30.0 + PRIOR_SCALAR[prior_value] * 50.0


# ============================================================
# TOPIC BANK (used for the post-test only)
# ============================================================
TOPIC_KEYWORDS = {
    "dna":            ["dna", "deoxyribonucleic", "nucleotide", "base pair", "double helix"],
    "rna":            ["rna", "ribonucleic", "mrna", "trna", "transcription", "translation"],
    "mitosis":        ["mitosis", "prophase", "metaphase", "anaphase", "telophase"],
    "meiosis":        ["meiosis", "gamete", "crossing over", "haploid"],
    "photosynthesis": ["photosynthesis", "chloroplast", "calvin", "light reaction"],
    "respiration":    ["respiration", "glycolysis", "krebs", "mitochondria", "atp"],
    "blood":          ["blood", "red blood", "rbc", "hemoglobin", "circulation",
                       "white blood", "platelet", "plasma"],
    "protein":        ["protein synthesis", "amino acid", "ribosome", "polypeptide"],
    "enzyme":         ["enzyme", "catalyst", "substrate", "active site"],
    "cell":           ["organelle", "cell membrane", "cytoplasm", "nucleus", "vacuole", "lysosome"],
    "genetics":       ["genetic", "allele", "phenotype", "genotype", "punnett", "inheritance"],
    "evolution":      ["evolution", "darwin", "natural selection", "fossil", "mutation"],
    "nervous":        ["nervous", "neuron", "synapse", "brain", "nerve"],
    "immune":         ["immune", "antibody", "pathogen", "b cell", "t cell", "vaccine"],
    "hormone":        ["hormone", "insulin", "pituitary", "thyroid", "adrenaline"],
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
             "options": ["Carry amino acids",
                         "Carry genetic message from DNA to ribosome",
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
                         "Homologous chromosomes exchanging segments",
                         "Cells dividing"],
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
    "cell": {
        "name": "Cell structure",
        "questions": [
            {"q": "Which organelle is the site of ATP production?",
             "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"],
             "correct": "Mitochondria"},
            {"q": "What controls what enters and exits the cell?",
             "options": ["Cytoplasm", "Cell membrane", "Nucleus", "Vacuole"],
             "correct": "Cell membrane"},
            {"q": "Which organelle contains digestive enzymes?",
             "options": ["Lysosome", "Ribosome", "Nucleolus", "Centriole"],
             "correct": "Lysosome"},
        ],
    },
    "genetics": {
        "name": "Genetics",
        "questions": [
            {"q": "What is the term for different forms of the same gene?",
             "options": ["Chromosomes", "Alleles", "Codons", "Ribosomes"],
             "correct": "Alleles"},
            {"q": "What term describes the physical expression of a trait?",
             "options": ["Genotype", "Phenotype", "Karyotype", "Allele"],
             "correct": "Phenotype"},
            {"q": "In a cross Aa x Aa, what fraction of offspring are typically recessive?",
             "options": ["0", "1/2", "1/4", "All"],
             "correct": "1/4"},
        ],
    },
    "evolution": {
        "name": "Evolution",
        "questions": [
            {"q": "What did Darwin call the process by which favourable traits spread?",
             "options": ["Mutation", "Natural selection", "Genetic drift", "Migration"],
             "correct": "Natural selection"},
            {"q": "What is a change in DNA sequence called?",
             "options": ["Replication", "Mutation", "Transcription", "Translation"],
             "correct": "Mutation"},
            {"q": "What are the preserved remains of ancient organisms called?",
             "options": ["Artifacts", "Fossils", "Minerals", "Sediments"],
             "correct": "Fossils"},
        ],
    },
    "nervous": {
        "name": "Nervous system",
        "questions": [
            {"q": "What is the basic functional unit of the nervous system?",
             "options": ["Neuron", "Synapse", "Axon", "Ganglion"],
             "correct": "Neuron"},
            {"q": "What part of the neuron receives incoming signals?",
             "options": ["Axon", "Dendrites", "Myelin sheath", "Terminal"],
             "correct": "Dendrites"},
            {"q": "What is the gap between two neurons called?",
             "options": ["Node", "Synapse", "Gap junction", "Vesicle"],
             "correct": "Synapse"},
        ],
    },
    "immune": {
        "name": "Immune system",
        "questions": [
            {"q": "Which cells produce antibodies?",
             "options": ["B cells", "T cells", "Macrophages", "Platelets"],
             "correct": "B cells"},
            {"q": "What is the first line of defence against pathogens?",
             "options": ["Antibodies", "Skin and mucous membranes",
                         "T cells", "Bone marrow"],
             "correct": "Skin and mucous membranes"},
            {"q": "What is the process of white blood cells engulfing pathogens called?",
             "options": ["Osmosis", "Phagocytosis", "Diffusion", "Apoptosis"],
             "correct": "Phagocytosis"},
        ],
    },
    "hormone": {
        "name": "Hormones",
        "questions": [
            {"q": "Which gland is known as the master gland?",
             "options": ["Thyroid", "Adrenal", "Pituitary", "Pancreas"],
             "correct": "Pituitary"},
            {"q": "Which hormone regulates blood glucose?",
             "options": ["Insulin", "Adrenaline", "Thyroxine", "Cortisol"],
             "correct": "Insulin"},
            {"q": "Hormones are transported mainly via which system?",
             "options": ["Nervous system", "Circulatory system",
                         "Lymphatic system", "Digestive system"],
             "correct": "Circulatory system"},
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
             "options": ["Photosynthesis", "Respiration",
                         "Protein synthesis", "Replication"],
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


def session_topic_hint(messages) -> str:
    """Return the first substantive user message as a topic anchor."""
    for m in messages:
        if m.get("role") == "user":
            text = m.get("content", "").strip()
            if len(text.split()) >= 3:
                return text
    return ""


def generate_posttest_with_llm(messages) -> dict:
    """
    Ask the LLM to write 3 MCQs based on the student's session.
    Returns the same schema as TOPIC_QUESTION_BANK entries.
    Falls back to the 'general' bank if the LLM fails or returns bad JSON.
    """
    transcript = "\n".join(
        f"{m['role']}: {m['content'][:300]}"
        for m in messages[-6:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    )

    prompt = f"""You are an exam writer for a biology tutor.

Below is a short transcript of a tutoring session between a student and an AI tutor.

--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---

Task:
Generate exactly 3 multiple-choice questions (4 options each) that test the
student's understanding of the SPECIFIC TOPIC discussed in the transcript.

Requirements:
- Detect the main topic from the transcript.
- Every question must be about that topic.
- Exactly 4 options per question.
- Mark one option as the correct answer.
- Do NOT ask about the tutor's teaching strategy or the student's state.
- Keep each question short and clear.

Return ONLY valid JSON in this exact shape, no commentary:
{{
  "name": "<short topic name>",
  "questions": [
    {{"q": "<q1>", "options": ["a","b","c","d"], "correct": "<exact correct option text>"}},
    {{"q": "<q2>", "options": ["a","b","c","d"], "correct": "<exact correct option text>"}},
    {{"q": "<q3>", "options": ["a","b","c","d"], "correct": "<exact correct option text>"}}
  ]
}}"""

    try:
        text, _ = chat([{"role": "user", "content": prompt}])
    except LLMUnavailable:
        return TOPIC_QUESTION_BANK["general"]

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = _json.loads(text[start:end])
        if (
            isinstance(data.get("name"), str)
            and isinstance(data.get("questions"), list)
            and len(data["questions"]) == 3
            and all(
                isinstance(q.get("q"), str)
                and isinstance(q.get("options"), list)
                and len(q["options"]) == 4
                and isinstance(q.get("correct"), str)
                and q["correct"] in q["options"]
                for q in data["questions"]
            )
        ):
            return data
    except (ValueError, _json.JSONDecodeError, KeyError):
        pass

    return TOPIC_QUESTION_BANK["general"]


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title=Config.APP_TITLE, page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- MAIN BACKGROUND ---- */
    .stApp {
        background: #FFFFFF;
        color: #1E293B;
    }

    /* ---- SIDEBAR ---- */
    [data-testid="stSidebar"] {
        background: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] * {
        color: #1E293B;
    }

    /* ---- HEADINGS ---- */
    h1 {
        color: #1E293B !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        color: #1E293B !important;
        font-weight: 700 !important;
    }

    /* ---- BODY TEXT ---- */
    p, span, label, div {
        color: #1E293B;
    }

    /* ---- BUTTONS ---- */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
    }
    .stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.25);
    }

    /* ---- CHAT MESSAGES ---- */
    [data-testid="stChatMessage"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #EEF2FF;
        border-color: #C7D2FE;
    }
    [data-testid="stChatMessage"] * {
        color: #1E293B !important;
    }

    /* ---- CHAT INPUT ---- */
    [data-testid="stChatInput"] {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
    }
    [data-testid="stChatInput"] textarea {
        color: #1E293B !important;
        background: #FFFFFF !important;
    }

    /* ---- METRICS ---- */
    [data-testid="stMetric"] {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        color: #1E293B !important;
        font-weight: 700;
    }

    /* ---- FORMS AND INPUTS ---- */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px;
    }

    /* ---- SLIDERS ---- */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #4F46E5 !important;
    }
    .stSlider [data-testid="stTickBar"] {
        color: #64748B;
    }

    /* ---- ALERTS (info, success, warning, error) ---- */
    [data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    [data-testid="stAlert"] * {
        color: #1E293B !important;
    }

    /* ---- EXPANDERS ---- */
    [data-testid="stExpander"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        background: #FFFFFF;
    }
    [data-testid="stExpander"] summary {
        color: #1E293B !important;
        font-weight: 600;
    }

    /* ---- PROGRESS BAR ---- */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4F46E5, #6366F1) !important;
    }

    /* ---- DIVIDER ---- */
    hr {
        border-color: #E2E8F0 !important;
    }

    /* ============================================================
       CUSTOM CHIPS — light background, dark text for readability
       ============================================================ */
    .chip {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px 6px 2px 0;
        border: 1px solid transparent;
    }
    .strategy-chip {
        background: #DBEAFE;
        color: #1E40AF;
        border-color: #93C5FD;
    }
    .prompt-chip {
        background: #D1FAE5;
        color: #065F46;
        border-color: #6EE7B7;
    }
    .algo-chip {
        background: #EDE9FE;
        color: #5B21B6;
        border-color: #C4B5FD;
    }

    /* ---- CARDS ---- */
    .card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        margin: 12px 0;
        color: #1E293B;
    }
    .card-info {
        background: #EFF6FF;
        border-color: #BFDBFE;
    }
    .card-success {
        background: #ECFDF5;
        border-color: #A7F3D0;
    }
    .card-warning {
        background: #FFFBEB;
        border-color: #FDE68A;
    }

    /* ---- CODE ---- */
    code {
        background: #F1F5F9;
        color: #1E293B;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9em;
    }

    /* ---- HIDE STREAMLIT CHROME ---- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
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
if "posttest_topic_name" not in st.session_state:
    st.session_state.posttest_topic_name = None
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

    # ---- Profile ----
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

    # ---- Cognitive state ----
    st.subheader("2️⃣ Cognitive State")
    st.caption("**SIMULATED** — from Components 1 & 2 (EEG + CV) in the full system.")
    attention = st.slider("🎯 Attention", 0.0, 1.0, 0.65, 0.05)
    fatigue   = st.slider("😴 Fatigue",   0.0, 1.0, 0.70, 0.05)
    confusion = st.slider("🤔 Confusion", 0.0, 1.0, 0.80, 0.05)
    readiness = st.slider("📚 Readiness", 0.0, 1.0, 0.40, 0.05)
    workload  = st.slider("📊 Workload",  0.0, 1.0, 0.55, 0.05)

    st.divider()

    # ---- Session status + post-test trigger ----
    st.subheader("3️⃣ Session Status")
    st.metric("💬 Interactions", st.session_state.interactions)

    current_topic = detect_topic(st.session_state.messages)
    if current_topic != "general":
        st.caption(f"Detected topic: **{TOPIC_QUESTION_BANK[current_topic]['name']}**")

    if st.session_state.interactions > 0 and not st.session_state.posttest_open:
        st.markdown("**Finished studying?**")
        if st.button("✅ Take the post-test"):
            topic = detect_topic(st.session_state.messages)
            if topic != "general":
                st.session_state.posttest_topic = topic
                st.session_state.posttest_questions = TOPIC_QUESTION_BANK[topic]["questions"]
                st.session_state.posttest_topic_name = TOPIC_QUESTION_BANK[topic]["name"]
            else:
                with st.spinner("Preparing post-test for your session topic..."):
                    generated = generate_posttest_with_llm(st.session_state.messages)
                st.session_state.posttest_topic = "custom"
                st.session_state.posttest_questions = generated["questions"]
                st.session_state.posttest_topic_name = generated["name"]
            st.session_state.posttest_open = True
            st.rerun()
    elif st.session_state.posttest_open:
        st.info("Post-test is open — see the main panel.")

    st.divider()

    # ---- Algorithm comparison ----
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
        st.session_state.posttest_topic = None
        st.session_state.posttest_topic_name = None
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
                st.markdown(m["rationale"])

# Chat input (unless post-test is open)
if not st.session_state.posttest_open:
    user_input = st.chat_input("Ask a question, e.g. 'Explain red blood cells'")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        state = CognitiveState(
            attention=attention, fatigue=fatigue,
            confusion=confusion, readiness=readiness, workload=workload)
        st.session_state.last_state = state

        with st.chat_message("assistant"):
            with st.spinner("Selecting strategy + prompt..."):
                decision = st.session_state.svc.decide(state, st.session_state.traits)
            st.session_state.last_decision = decision

            history = st.session_state.messages[:-1]
            topic_anchor = session_topic_hint(history)

            # If the current message is a short follow-up, anchor it to the topic
            current_user = user_input
            if topic_anchor and len(user_input.split()) <= 6:
                current_user = (
                    f"{user_input}\n\n"
                    f"(Note: this is a follow-up to our conversation that started with: "
                    f"\"{topic_anchor}\". Stay on that topic.)"
                )

            messages = build_messages(
                state, st.session_state.traits,
                decision["strategy"], decision["prompt"],
                topic=current_user,
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
                    st.markdown(decision["rationale"])
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

# ============================================================
# POST-TEST
# ============================================================
if st.session_state.posttest_open:
    st.divider()
    topic_name = st.session_state.posttest_topic_name or "General biology"
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

        st.session_state.posttest_open = False
        st.session_state.posttest_questions = None
        st.session_state.posttest_topic = None
        st.session_state.posttest_topic_name = None
        
        if post_score >= 66:
            st.balloons()
            st.success("🎉 Great job on the post-test!")
        elif post_score > 0:
            st.info("👍 Good effort. The tutor will adapt to help you better next time.")
        else:
            st.warning("📚 It looks like the topic was tough. The system has logged this to improve future sessions.")

    if skip:
        st.session_state.posttest_open = False
        st.session_state.posttest_questions = None
        st.session_state.posttest_topic = None
        st.session_state.posttest_topic_name = None
        st.rerun()

# ============================================================
# DECISION TRACE
# ============================================================
with st.expander("🔍 Decision trace (last 10)"):
    if st.session_state.svc and st.session_state.svc.trace:
        st.json(st.session_state.svc.trace[-10:])
    else:
        st.caption("No decisions logged yet.")