"""
Streamlit UI for the Placement Preparation & Quiz Agent.

Run:
    streamlit run app.py
"""

import os

import streamlit as st

from rag_agent import TOPICS, PlacementAgent


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Placement Preparation & Quiz Agent",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Placement Preparation & Quiz Agent")

st.caption("RAG + TF-IDF + Groq AI")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Leave empty to use offline mode."
    )

    model_name = st.text_input(
        "Groq Model",
        value=os.getenv(
            "LLM_MODEL",
            "openai/gpt-oss-20b"
        )
    )


# =========================================================
# CREATE AGENT
# =========================================================

@st.cache_resource(show_spinner=False)
def get_agent(key: str, model: str) -> PlacementAgent:

    return PlacementAgent(
        "college_info.txt",
        api_key=key or None,
        model=model
    )


agent = get_agent(
    api_key,
    model_name
)


# =========================================================
# AI STATUS
# =========================================================

if agent.online:

    st.sidebar.success(
        "🟢 Groq AI Mode: ON"
    )

else:

    st.sidebar.warning(
        "🟡 Offline Mode"
    )


# =========================================================
# MAIN TABS
# =========================================================

tab_chat, tab_quiz = st.tabs(
    [
        "💬 Ask About Placements",
        "📝 Practice Quiz"
    ]
)


# =========================================================
# TAB 1 — PLACEMENT CHAT
# =========================================================

with tab_chat:

    st.header("💬 Ask About Placements")

    st.write(
        "Ask questions about placement eligibility, "
        "training, interviews, technical preparation, "
        "and other placement information."
    )

    # Store conversation
    st.session_state.setdefault(
        "messages",
        []
    )

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # =====================================================
    # CHAT INPUT
    # =====================================================

    question = st.chat_input(
        "Ask about eligibility, recruiters, interview tips..."
    )

    if question:

        # Display user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching placement information..."
            ):

                result = agent.ask(
                    question
                )

            st.markdown(
                result["answer"]
            )

            # Show retrieved sources
            if result["sources"]:

                with st.expander(
                    "📚 Retrieved Sources"
                ):

                    for source in result["sources"]:

                        st.info(source)

        # Store AI response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"]
            }
        )


# =========================================================
# TAB 2 — QUIZ
# =========================================================

with tab_quiz:

    st.header(
        "📝 Placement Practice Quiz"
    )

    st.write(
        "Generate AI-powered placement questions "
        "and test your knowledge."
    )

    # Quiz controls
    c1, c2, c3 = st.columns(3)

    topic = c1.selectbox(
        "📚 Topic",
        TOPICS
    )

    n_q = c2.slider(
        "🔢 Questions",
        3,
        10,
        5
    )

    difficulty = c3.selectbox(
        "🎯 Difficulty",
        [
            "easy",
            "medium",
            "hard"
        ],
        index=1
    )

    # Generate quiz
    if st.button(
        "🚀 Generate Quiz",
        type="primary"
    ):

        with st.spinner(
            "🤖 Groq is generating your quiz..."
        ):

            st.session_state.quiz = (
                agent.generate_quiz(
                    topic,
                    n_q,
                    difficulty
                )
            )

            st.session_state.quiz_topic = topic

            st.session_state.submitted = False

    # Get generated quiz
    quiz = st.session_state.get(
        "quiz"
    )

    if quiz:

        with st.form(
            "quiz_form"
        ):

            answers = []

            for i, q in enumerate(
                quiz
            ):

                answer = st.radio(
                    f"Q{i + 1}. {q['question']}",
                    q["options"],
                    index=None,
                    key=f"question_{i}"
                )

                answers.append(
                    answer
                )

            submitted = st.form_submit_button(
                "✅ Submit Answers"
            )

            if submitted:

                st.session_state.submitted = True

                st.session_state.answers = answers

        # =================================================
        # RESULTS
        # =================================================

        if st.session_state.get(
            "submitted"
        ):

            answers = st.session_state.answers

            score = 0

            missed = []

            st.divider()

            st.subheader(
                "📊 Results"
            )

            for i, (q, answer) in enumerate(
                zip(quiz, answers)
            ):

                correct = (
                    q["options"][
                        q["answer_index"]
                    ]
                )

                if answer == correct:

                    score += 1

                    st.success(
                        f"Q{i + 1}: Correct ✅ — "
                        f"{correct}"
                    )

                else:

                    missed.append(
                        q["question"]
                    )

                    st.error(
                        f"Q{i + 1}: "
                        f"Your answer: "
                        f"{answer or 'Not answered'} | "
                        f"Correct: {correct}"
                    )

                st.caption(
                    "💡 " + q["explanation"]
                )

            total = len(quiz)

            percentage = round(
                100 * score / total
            )

            st.subheader(
                f"🏆 Score: {score}/{total}"
            )

            st.progress(
                percentage / 100
            )

            st.write(
                f"Percentage: **{percentage}%**"
            )

            # =================================================
            # AI FEEDBACK
            # =================================================

            st.subheader(
                "🤖 AI Feedback"
            )

            feedback = agent.feedback(
                st.session_state.quiz_topic,
                score,
                total,
                missed
            )

            st.info(
                feedback
            )