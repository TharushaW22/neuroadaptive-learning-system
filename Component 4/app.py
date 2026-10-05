"""
app.py — MD-AP2L Streamlit Chat Tutor
Run: streamlit run app.py
"""
import os
import streamlit as st

from schemas import (CognitiveState, StudentTraits, LearningStyle, Interest,
                     Goal, Prior, Motivation, Emotion, EvaluationSignal)
from policy_service import PolicyService
from llm.prompts import build_messages
from llm.client import chat, LLMUnavailable
from evaluation.sus import score as sus_score
from comparison import run as run_comparison
from config import Config

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
        display: inline-block;
        background: rgba(96, 165, 250, 0.18);
        color: #93c5fd;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .prompt-chip {
        display: inline-block;
        background: rgba(52, 211, 153, 0.18);
        color: #6ee7b7;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .algo-chip {
        display: inline-block;
        background: rgba(168, 85, 247, 0.18);
        color: #d8b4fe;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
</style>
""", unsafe_allow_html=True)

if "svc" not in st.session_state:
    st.session_state.svc = PolicyService()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "traits" not in st.session_state:
    st.session_state.traits = None
if "pre_score" not in st.session_state:
    st.session_state.pre_score = None
if "awaiting_post" not in st.session_state:
    st.session_state.awaiting_post = False
if "last_decision" not in st.session_state:
    st.session_state.last_decision = None
if "last_state" not in st.session_state:
    st.session_state.last_state = None
if "comparison_done" not in st.session_state:
    st.session_state.comparison_done = False
if "winner_name" not in st.session_state:
    st.session_state.winner_name = None

with st.sidebar:
    st.markdown("## 🧠 MD-AP2L")
    st.caption("Multi-Dimensional Adaptive Pedagogical Policy Learning · J26-DS-333")
    st.caption(f"`{Config.summary()}`")
    st.divider()

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
        st.success("Profile saved.")
    if st.session_state.traits is None:
        st.warning("⚠️ Save your profile before chatting.")

    st.divider()
    st.subheader("2️⃣ Cognitive State")
    st.caption("**SIMULATED** — from Components 1 & 2 (EEG + CV) in the full system.")
    attention = st.slider("🎯 Attention", 0.0, 1.0, 0.65, 0.05)
    fatigue   = st.slider("😴 Fatigue",   0.0, 1.0, 0.70, 0.05)
    confusion = st.slider("🤔 Confusion", 0.0, 1.0, 0.80, 0.05)
    readiness = st.slider("📚 Readiness", 0.0, 1.0, 0.40, 0.05)
    workload  = st.slider("📊 Workload",  0.0, 1.0, 0.55, 0.05)

    st.divider()
    st.subheader("3️⃣ Pre-Test")
    pre_input = st.number_input("Pre-test score (0–100)", 0, 100, 45, 1)
    if st.button("Set pre-test score"):
        st.session_state.pre_score = float(pre_input)
        st.success(f"Pre-test set to {pre_input}.")

    st.divider()
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
        st.session_state.awaiting_post = False
        st.rerun()

st.title("🧬 MD-AP2L — Adaptive Tutor")
st.caption("Chat with a tutor that adapts its teaching strategy to *how you feel* and *who you are*.")

if st.session_state.traits is None:
    st.info("👈 Complete the questionnaire in the sidebar to begin.")
    st.stop()

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

if not st.session_state.awaiting_post:
    user_input = st.chat_input("Ask a question, e.g. 'Explain mitosis'")
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

            messages = build_messages(
                state, st.session_state.traits,
                decision["strategy"], decision["prompt"], user_input)
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
                st.session_state.awaiting_post = True
                st.rerun()

if st.session_state.awaiting_post:
    st.divider()
    st.subheader("📊 Post-Test & Learning Gain")
    if st.session_state.pre_score is None:
        st.warning("Set the pre-test score in the sidebar first.")
    else:
        post_input = st.number_input("Post-test score (0–100)", 0, 100, 85, 1)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Submit result"):
                sig = EvaluationSignal(
                    pre_score=st.session_state.pre_score,
                    post_score=float(post_input)).compute()
                st.success(
                    f"Learning gain = **{sig.learning_gain:.2f}** "
                    f"| normalized gain = **{sig.normalized_gain:.2f}**")
                d = st.session_state.last_decision
                s = st.session_state.last_state
                st.session_state.svc.feedback(
                    s, st.session_state.traits,
                    d["strategy"], d["prompt"], sig.learning_gain)
                st.info("Policy updated — the system will adapt better next time.")
                st.session_state.awaiting_post = False
                st.rerun()
        with c2:
            if st.button("⏭ Skip (no update)"):
                st.session_state.awaiting_post = False
                st.rerun()

with st.expander("🔍 Decision trace (last 10)"):
    st.json(st.session_state.svc.trace[-10:])