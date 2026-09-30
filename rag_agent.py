"""
Placement Preparation & Quiz Agent - core logic.

Pipeline (RAG):
    college_info.txt
        ↓
    chunks
        ↓
    TF-IDF index
        ↓
    top-k retrieval
        ↓
    Groq LLM
        ↓
    Answer

Quiz generation uses Groq and returns strict JSON.

Without a Groq API key, the agent still works
in offline mode using retrieval and fallback quizzes.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv

    load_dotenv()

except ImportError:
    pass


# =========================================================
# GROQ CONFIGURATION
# =========================================================

DEFAULT_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b"
)


# =========================================================
# PLACEMENT TOPICS
# =========================================================

TOPICS = [
    "Aptitude",
    "Data Structures & Algorithms",
    "DBMS & SQL",
    "Operating Systems",
    "Computer Networks",
    "OOP",
    "HR & Behavioural"
]


# =========================================================
# FALLBACK QUIZ
# =========================================================

FALLBACK_QUIZ = [

    {
        "topic": "Aptitude",
        "question":
            "A train 150 m long passes a pole in 15 seconds. "
            "Its speed is:",

        "options": [
            "10 m/s",
            "15 m/s",
            "20 m/s",
            "25 m/s"
        ],

        "answer_index": 0,

        "explanation":
            "Speed = distance / time = 150 / 15 = 10 m/s."
    },

    {
        "topic": "Aptitude",

        "question":
            "What is 20% of 20% of 500?",

        "options": [
            "10",
            "20",
            "25",
            "50"
        ],

        "answer_index": 1,

        "explanation":
            "20% of 500 = 100; 20% of 100 = 20."
    },

    {
        "topic": "Data Structures & Algorithms",

        "question":
            "Time complexity of binary search on a sorted array?",

        "options": [
            "O(n)",
            "O(log n)",
            "O(n log n)",
            "O(1)"
        ],

        "answer_index": 1,

        "explanation":
            "Each step halves the search space."
    },

    {
        "topic": "Data Structures & Algorithms",

        "question":
            "Which data structure is used for BFS traversal?",

        "options": [
            "Stack",
            "Queue",
            "Heap",
            "Tree"
        ],

        "answer_index": 1,

        "explanation":
            "BFS explores level by level using a FIFO queue."
    },

    {
        "topic": "DBMS & SQL",

        "question":
            "Which SQL clause filters groups after aggregation?",

        "options": [
            "WHERE",
            "GROUP BY",
            "HAVING",
            "ORDER BY"
        ],

        "answer_index": 2,

        "explanation":
            "HAVING filters groups after aggregation."
    },

    {
        "topic": "DBMS & SQL",

        "question":
            "Which normal form removes partial dependencies?",

        "options": [
            "1NF",
            "2NF",
            "3NF",
            "BCNF"
        ],

        "answer_index": 1,

        "explanation":
            "2NF removes partial dependency."
    },

    {
        "topic": "Operating Systems",

        "question":
            "Which condition is NOT required for a deadlock?",

        "options": [
            "Mutual exclusion",
            "Hold and wait",
            "Preemption",
            "Circular wait"
        ],

        "answer_index": 2,

        "explanation":
            "Deadlock requires NO preemption."
    },

    {
        "topic": "Computer Networks",

        "question":
            "Which protocol is connectionless?",

        "options": [
            "TCP",
            "UDP",
            "HTTP",
            "FTP"
        ],

        "answer_index": 1,

        "explanation":
            "UDP is a connectionless protocol."
    },

    {
        "topic": "OOP",

        "question":
            "Which concept lets a subclass provide its own "
            "version of a parent method?",

        "options": [
            "Encapsulation",
            "Overriding",
            "Overloading",
            "Abstraction"
        ],

        "answer_index": 1,

        "explanation":
            "Method overriding allows a subclass to redefine a method."
    },

    {
        "topic": "HR & Behavioural",

        "question":
            "What does the STAR method stand for?",

        "options": [
            "Skill, Task, Action, Review",
            "Situation, Task, Action, Result",
            "Situation, Team, Approach, Result",
            "Strategy, Task, Action, Result"
        ],

        "answer_index": 1,

        "explanation":
            "STAR means Situation, Task, Action, Result."
    }
]


# =========================================================
# CHUNKING
# =========================================================

def chunk_text(
    text: str,
    max_words: int = 120
) -> list[str]:

    chunks = []

    for block in re.split(
        r"\n\s*\n",
        text
    ):

        block = block.strip()

        if not block:
            continue

        if block.startswith("# NOTE"):
            continue

        words = block.split()

        if len(words) <= max_words:

            chunks.append(block)

        else:

            for i in range(
                0,
                len(words),
                max_words
            ):

                chunks.append(
                    " ".join(
                        words[
                            i:i + max_words
                        ]
                    )
                )

    return chunks


# =========================================================
# RETRIEVER
# =========================================================

class Retriever:

    def __init__(
        self,
        chunks: list[str]
    ):

        if not chunks:

            raise ValueError(
                "No text chunks to index."
            )

        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.matrix = (
            self.vectorizer.fit_transform(
                chunks
            )
        )


    def search(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.05
    ):

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).ravel()

        top = (
            scores.argsort()[::-1][:k]
        )

        return [
            (
                self.chunks[i],
                float(scores[i])
            )

            for i in top

            if scores[i] >= min_score
        ]


# =========================================================
# PLACEMENT AGENT
# =========================================================

class PlacementAgent:

    def __init__(
        self,
        info_path: str = "college_info.txt",
        api_key: str | None = None,
        model: str = DEFAULT_MODEL
    ):

        # ---------------------------------------------
        # Load knowledge base
        # ---------------------------------------------

        text = Path(
            info_path
        ).read_text(
            encoding="utf-8"
        )

        chunks = chunk_text(text)

        self.retriever = Retriever(
            chunks
        )

        # ---------------------------------------------
        # Groq configuration
        # ---------------------------------------------

        self.model = model

        self.client = None

        key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )

        if key:

            try:

                from groq import Groq

                self.client = Groq(
                    api_key=key
                )

            except Exception as e:

                print(
                    "Groq initialization failed:",
                    e
                )

                self.client = None


    # =====================================================
    # ONLINE STATUS
    # =====================================================

    @property
    def online(self) -> bool:

        return self.client is not None


    # =====================================================
    # GROQ LLM
    # =====================================================

    def _llm(
        self,
        system: str,
        user: str,
        max_tokens: int = 1000
    ) -> str:

        if not self.client:

            raise RuntimeError(
                "Groq client is not initialized."
            )

        response = (
            self.client.chat.completions.create(

                model=self.model,

                messages=[
                    {
                        "role": "system",
                        "content": system
                    },

                    {
                        "role": "user",
                        "content": user
                    }
                ],

                temperature=0.2,

                max_tokens=max_tokens
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


    # =====================================================
    # RAG QUESTION ANSWERING
    # =====================================================

    def ask(
        self,
        question: str,
        k: int = 3
    ) -> dict:

        # ---------------------------------------------
        # Retrieve relevant chunks
        # ---------------------------------------------

        hits = self.retriever.search(
            question,
            k=k
        )

        sources = [
            hit[0]
            for hit in hits
        ]

        # ---------------------------------------------
        # Nothing relevant found
        # ---------------------------------------------

        if not sources:

            return {
                "answer":
                    "I couldn't find that information "
                    "in the college placement knowledge base. "
                    "Please contact the Training & Placement Cell.",

                "sources": []
            }

        # ---------------------------------------------
        # Offline mode
        # ---------------------------------------------

        if not self.online:

            return {
                "answer":
                    "(Offline mode - top matching information)\n\n"
                    + "\n\n".join(sources),

                "sources": sources
            }

        # ---------------------------------------------
        # Build RAG context
        # ---------------------------------------------

        context = "\n\n".join(
            f"[Source {i + 1}] {source}"

            for i, source in enumerate(
                sources
            )
        )

        # ---------------------------------------------
        # System prompt
        # ---------------------------------------------

        system = """
You are a placement preparation assistant for college students.

Answer ONLY using the provided context.

Rules:

1. Do not invent college-specific information.
2. Do not invent recruiter names.
3. Do not invent salary information.
4. Do not invent eligibility criteria.
5. If the answer is not present in the context,
   clearly say that the information is not available.
6. Suggest contacting the Training & Placement Cell
   when appropriate.
7. Keep answers concise and student-friendly.
"""

        # ---------------------------------------------
        # User prompt
        # ---------------------------------------------

        user = f"""
Context:

{context}


Student Question:

{question}
"""

        # ---------------------------------------------
        # Call Groq
        # ---------------------------------------------

        try:

            answer = self._llm(
                system,
                user,
                max_tokens=1000
            )

        except Exception as e:

            print(
                "Groq Q&A error:",
                e
            )

            answer = (
                "The AI service could not generate "
                "a response at the moment.\n\n"
                "Relevant placement information:\n\n"
                + "\n\n".join(sources)
            )

        return {
            "answer": answer,
            "sources": sources
        }


    # =====================================================
    # QUIZ JSON PARSER
    # =====================================================

    @staticmethod
    def parse_quiz_json(
        text: str
    ) -> list[dict]:

        cleaned = text.strip()

        cleaned = re.sub(
            r"^```(?:json)?",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

        cleaned = re.sub(
            r"```$",
            "",
            cleaned
        ).strip()

        data = json.loads(
            cleaned
        )

        if isinstance(
            data,
            dict
        ):

            data = data.get(
                "questions",
                []
            )

        quiz = []

        for q in data:

            options = q["options"]

            answer_index = int(
                q["answer_index"]
            )

            if (
                len(options) != 4
                or not 0 <= answer_index < 4
            ):

                raise ValueError(
                    "Each question needs 4 options "
                    "and a valid answer_index."
                )

            quiz.append(
                {
                    "question":
                        str(q["question"]),

                    "options":
                        [
                            str(option)
                            for option in options
                        ],

                    "answer_index":
                        answer_index,

                    "explanation":
                        str(
                            q.get(
                                "explanation",
                                ""
                            )
                        )
                }
            )

        if not quiz:

            raise ValueError(
                "Empty quiz."
            )

        return quiz


    # =====================================================
    # FALLBACK QUIZ
    # =====================================================

    def _fallback_quiz(
        self,
        topic: str,
        n: int
    ) -> list[dict]:

        pool = [
            q

            for q in FALLBACK_QUIZ

            if q["topic"] == topic
        ]

        if not pool:

            pool = FALLBACK_QUIZ

        return random.sample(
            pool,
            min(
                n,
                len(pool)
            )
        )


    # =====================================================
    # GROQ QUIZ GENERATION
    # =====================================================

    def generate_quiz(
        self,
        topic: str,
        n: int = 5,
        difficulty: str = "medium"
    ) -> list[dict]:

        if not self.online:

            return self._fallback_quiz(
                topic,
                n
            )

        system = """
You are an expert campus placement
examination question generator.

Generate technically correct multiple-choice
questions.

Return ONLY valid JSON.

Do not use markdown.
Do not write anything outside JSON.
"""

        user = f"""
Create {n} {difficulty}-difficulty
multiple-choice questions on:

{topic}

These questions are for
campus placement preparation.

Return exactly this structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer_index": 0,
            "explanation": "Short explanation"
        }}
    ]
}}

Rules:

- Exactly four options.
- answer_index must be 0, 1, 2 or 3.
- Questions must be technically correct.
- Avoid duplicate questions.
"""

        try:

            response = self._llm(
                system,
                user,
                max_tokens=2500
            )

            return self.parse_quiz_json(
                response
            )

        except Exception as e:

            print(
                "Quiz generation failed:",
                e
            )

            return self._fallback_quiz(
                topic,
                n
            )


    # =====================================================
    # QUIZ FEEDBACK
    # =====================================================

    def feedback(
        self,
        topic: str,
        score: int,
        total: int,
        missed: list[str]
    ) -> str:

        percentage = (
            round(
                100 * score / total
            )
            if total
            else 0
        )

        # ---------------------------------------------
        # Offline feedback
        # ---------------------------------------------

        if not self.online:

            if percentage >= 80:

                level = "Excellent!"

            elif percentage >= 50:

                level = "Good effort."

            else:

                level = "Keep practising."

            return (
                f"{level} "
                f"You scored {score}/{total} "
                f"({percentage}%) in {topic}."
            )

        # ---------------------------------------------
        # Groq feedback
        # ---------------------------------------------

        prompt = f"""
A student completed a placement preparation quiz.

Topic:
{topic}

Score:
{score}/{total}

Percentage:
{percentage}%

Missed questions:
{missed}

Give:

1. A short performance summary.
2. Three specific study tips.
3. One practice recommendation.

Keep the response concise and encouraging.
"""

        try:

            return self._llm(
                "You are a supportive placement coach.",
                prompt,
                max_tokens=500
            )

        except Exception as e:

            print(
                "Groq feedback error:",
                e
            )

            return (
                f"You scored {score}/{total} "
                f"({percentage}%). "
                f"Continue practising {topic}."
            )