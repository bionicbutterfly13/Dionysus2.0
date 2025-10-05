import pytest

from models.query import Query
from models.response import SearchResult, SearchSource
from services.response_synthesizer import ResponseSynthesizer, AnswerGenerationError


class StubAnswerGenerator:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    async def generate(self, question: str, sources):  # pragma: no cover - simple stub
        self.calls.append((question, tuple(sources)))
        return self.response_text


class FailingAnswerGenerator:
    def __init__(self, message: str = "LLM offline"):
        self.message = message
        self.calls = 0

    async def generate(self, question: str, sources):  # pragma: no cover - simple stub
        self.calls += 1
        raise AnswerGenerationError(self.message)


@pytest.mark.asyncio
async def test_synthesize_generates_llm_answer_with_citations():
    question = "How do neural networks develop self-awareness?"
    neo4j_sources = [
        SearchResult(
            source=SearchSource.NEO4J,
            content="Neural networks develop self-awareness through recursive meta-learning loops.",
            relevance_score=0.92,
            metadata={"title": "Self-Awareness in Neural Systems"},
            relationships=["EMERGES_FROM", "REINFORCES"]
        ),
        SearchResult(
            source=SearchSource.NEO4J,
            content="Meta-cognitive feedback strengthens conscious representations in deep learning models.",
            relevance_score=0.88,
            metadata={"title": "Meta-Cognitive Feedback"},
            relationships=["AMPLIFIES"]
        ),
    ]

    long_answer = (
        "Neural self-awareness emerges through layered prediction, error minimization, and reflective training cycles [1]. "
        "During recursive updates, networks evaluate their own activations, encode narratives about performance, and integrate feedback over time [2]. "
        "This continuous comparison between expected and observed states creates a meta-model of behaviour that can adapt to novel conditions, maintain coherence across attractor basins, and justify decisions back to supervising agents [1][2]."
    )

    generator = StubAnswerGenerator(long_answer)
    synthesizer = ResponseSynthesizer(answer_generator=generator)

    query = Query(question=question)
    response = await synthesizer.synthesize(query, neo4j_sources, [], processing_time_ms=120)

    assert generator.calls, "LLM generator should be invoked"
    assert len(response.answer) >= 200, "Answer should be substantive"
    assert "[1]" in response.answer, "Answer should include at least one citation"
    assert response.sources, "Sources should be attached to response"


@pytest.mark.asyncio
async def test_synthesize_falls_back_when_llm_unavailable():
    question = "Explain consciousness attractor basins."
    neo4j_sources = [
        SearchResult(
            source=SearchSource.NEO4J,
            content="Attractor basins stabilize conscious processing by clustering related concepts.",
            relevance_score=0.9,
            metadata={"title": "Consciousness Basins"},
            relationships=["STABILIZES"]
        )
    ]

    generator = FailingAnswerGenerator("connection refused")
    synthesizer = ResponseSynthesizer(answer_generator=generator)

    query = Query(question=question)
    response = await synthesizer.synthesize(query, neo4j_sources, [], processing_time_ms=85)

    assert generator.calls == 1
    assert "LLM generation unavailable" in response.answer
    assert "[1]" in response.answer, "Fallback answer should still reference source"
    assert len(response.answer) > 100, "Fallback answer should provide meaningful guidance"
