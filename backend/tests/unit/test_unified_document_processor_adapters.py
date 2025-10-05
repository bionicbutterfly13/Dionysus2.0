import asyncio
import sys
import types

import pytest


def ensure_package(name: str):  # pragma: no cover - test utility
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return sys.modules[name]


def stub_module(name: str, attrs: dict):  # pragma: no cover - test utility
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


# Stub legacy context isolator
ensure_package("legacy")
ensure_package("legacy.daedalus_bridge")
stub_module("legacy.daedalus_bridge.context_isolator", {"ContextIsolatedAgent": object})

# Stub Dionysus source packages
ensure_package("dionysus_source")
ensure_package("dionysus_source.agents")
ensure_package("dionysus_source.kggen")


class DummyAlgorithmExtractor:  # pragma: no cover - stub
    async def extract(self, document_path: str):
        return {"algorithms": []}


class DummyLangExtractAdapter:  # pragma: no cover - stub
    async def extract(self, document_path: str):
        return {"sections": [], "entities": []}


class DummyThoughtseedNetwork:  # pragma: no cover - stub
    async def process_packet(self, packet):
        return {"layers": []}


class DummyNeuronalPacket:  # pragma: no cover - stub
    def __init__(self, id: str, content: dict, activation_level: float = 0.0):
        self.id = id
        self.content = content
        self.activation_level = activation_level


stub_module(
    "dionysus_source.agents.advanced_algorithm_extractor",
    {"AdvancedAlgorithmExtractor": DummyAlgorithmExtractor}
)
stub_module(
    "dionysus_source.agents.langextract_adapter",
    {
        "LangExtractAdapter": DummyLangExtractAdapter,
        "LANGEXTRACT_AVAILABLE": False
    }
)
stub_module(
    "dionysus_source.agents.thoughtseed_core",
    {
        "ThoughtseedNetwork": DummyThoughtseedNetwork,
        "NeuronalPacket": DummyNeuronalPacket
    }
)
stub_module(
    "dionysus_source.kggen.core_extractor",
    {"KGGenExtractor": object}
)
stub_module(
    "dionysus_source.constitutional_document_gateway",
    {"ConstitutionalDocumentGateway": type("Gateway", (), {})}
)

from services import unified_document_processor as udp
from services.unified_document_processor import UnifiedDocumentProcessor

# Ensure tests run on fallback paths to avoid external dependencies
udp.LANGEXTRACT_AVAILABLE = False
udp.PYMUPDF_AVAILABLE = False
udp.FITZ_AVAILABLE = False


class LightweightProcessor(UnifiedDocumentProcessor):
    """Processor that skips heavy external initialisation for tests."""

    def _initialize_processors(self):  # pragma: no cover - testing convenience
        return {}


@pytest.fixture()
def sample_processor():
    return LightweightProcessor()


@pytest.fixture()
def simple_text(tmp_path):
    text = (
        "Introduction to Consciousness\n\n"
        "Consciousness research explores recursive self-models and attractor basins.\n"
        "Algorithms:\n"
        "```python\n"
        "def meta_loop(state):\n"
        "    for layer in range(3):\n"
        "        state = update(state, layer)\n"
        "    return state\n"
        "```\n\n"
        "Conclusion:\n"
        "Emergent behaviour aligns with predictive processing."
    )
    file_path = tmp_path / "sample.txt"
    file_path.write_text(text)
    return str(file_path), text


@pytest.mark.asyncio
async def test_extract_text_pymupdf_provides_content(sample_processor, simple_text):
    path, original_text = simple_text

    result = await sample_processor._extract_text_pymupdf(path)

    assert result["status"] != "placeholder"
    assert "content" in result and original_text.split("\n\n")[0] in result["content"]
    assert "metadata" in result
    assert result["metadata"]["total_words"] >= 5


@pytest.mark.asyncio
async def test_extract_structured_langextract_has_sections(sample_processor, simple_text):
    path, _ = simple_text

    result = await sample_processor._extract_structured_langextract(path)

    assert result["status"] != "placeholder"
    assert result["sections"], "Structured extraction should return at least one section"
    first_section = result["sections"][0]
    assert "title" in first_section and first_section["title"], "Section title should not be empty"
    assert "content" in first_section and first_section["content"], "Section content should not be empty"


@pytest.mark.asyncio
async def test_extract_algorithms_detects_code_blocks(sample_processor, simple_text):
    path, _ = simple_text

    result = await sample_processor._extract_algorithms(path)

    assert result["status"] != "placeholder"
    assert result["algorithms"], "Algorithm extractor should return detected algorithms"
    algo = result["algorithms"][0]
    assert "snippet" in algo and "meta_loop" in algo["snippet"], "Algorithm snippet should contain code"


@pytest.mark.asyncio
async def test_extract_knowledge_graph_returns_entities(sample_processor, simple_text):
    path, _ = simple_text

    result = await sample_processor._extract_knowledge_graph(path)

    assert result["status"] != "placeholder"
    assert result["entities"], "Knowledge graph extractor should identify entities"
    assert result["relationships"] is not None
