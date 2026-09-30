"""Run with:  pytest -q
Tests use offline mode only - no API key or network needed."""
import pytest

from rag_agent import PlacementAgent, Retriever, chunk_text


@pytest.fixture(scope="module")
def agent():
    return PlacementAgent("college_info.txt", api_key=None)


def test_chunking_splits_paragraphs_and_skips_note():
    text = "# NOTE: ignore\n\nFirst para.\n\nSecond para."
    assert chunk_text(text) == ["First para.", "Second para."]


def test_long_paragraph_is_windowed():
    chunks = chunk_text(" ".join(["word"] * 300), max_words=120)
    assert len(chunks) == 3


def test_retriever_finds_relevant_chunk():
    r = Retriever(["Bananas are yellow fruit.", "The CGPA cutoff for placements is 6.5."])
    top = r.search("What is the CGPA cutoff?", k=1)
    assert "CGPA" in top[0][0]


def test_agent_is_offline_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert PlacementAgent("college_info.txt", api_key=None).online is False


def test_ask_returns_sources(agent):
    res = agent.ask("What is the minimum CGPA eligibility?")
    assert res["sources"] and "6.5" in res["answer"]


def test_ask_unknown_topic(agent):
    res = agent.ask("zzzz qqqq xxxx")
    assert res["sources"] == []


def test_quiz_fallback_shape(agent):
    quiz = agent.generate_quiz("Aptitude", n=2)
    assert len(quiz) == 2
    for q in quiz:
        assert len(q["options"]) == 4 and 0 <= q["answer_index"] < 4


def test_parse_quiz_json_handles_code_fences():
    raw = '```json\n{"questions":[{"question":"Q?","options":["a","b","c","d"],"answer_index":2}]}\n```'
    assert PlacementAgent.parse_quiz_json(raw)[0]["answer_index"] == 2


def test_parse_quiz_json_rejects_bad_data():
    with pytest.raises(Exception):
        PlacementAgent.parse_quiz_json('{"questions":[{"question":"Q","options":["a"],"answer_index":5}]}')


def test_feedback_offline(agent):
    assert "4/5" in agent.feedback("Aptitude", 4, 5, [])
