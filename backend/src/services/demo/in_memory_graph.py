"""
In-Memory Graph Database (Demo Mode)

Provides a simple in-memory graph for testing CLAUSE Phase 2 without Neo4j.
Pre-populated with climate change knowledge graph.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class InMemoryGraph:
    """In-memory graph database for demo mode"""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

        # Initialize with climate change demo data
        self._initialize_demo_data()

    def _initialize_demo_data(self):
        """Create demo knowledge graph about climate change"""

        # Add nodes
        demo_nodes = [
            {
                "id": "climate_change",
                "text": "Climate change refers to long-term shifts in global temperatures and weather patterns",
                "type": "concept",
            },
            {
                "id": "greenhouse_gases",
                "text": "Greenhouse gases trap heat in Earth's atmosphere",
                "type": "concept",
            },
            {
                "id": "CO2",
                "text": "Carbon dioxide (CO2) is the primary greenhouse gas from human activity",
                "type": "concept",
            },
            {
                "id": "fossil_fuels",
                "text": "Fossil fuels release CO2 when burned for energy",
                "type": "concept",
            },
            {
                "id": "global_warming",
                "text": "Global warming is the increase in Earth's average surface temperature",
                "type": "concept",
            },
            {
                "id": "sea_level_rise",
                "text": "Rising sea levels result from melting ice and thermal expansion",
                "type": "effect",
            },
            {
                "id": "extreme_weather",
                "text": "Climate change increases frequency of extreme weather events",
                "type": "effect",
            },
            {
                "id": "renewable_energy",
                "text": "Renewable energy sources don't emit greenhouse gases",
                "type": "solution",
            },
        ]

        for node in demo_nodes:
            self.nodes[node["id"]] = node

        # Add edges
        demo_edges = [
            {"from": "greenhouse_gases", "to": "climate_change", "relation": "causes"},
            {"from": "CO2", "to": "greenhouse_gases", "relation": "is_a"},
            {"from": "fossil_fuels", "to": "CO2", "relation": "produces"},
            {"from": "climate_change", "to": "global_warming", "relation": "includes"},
            {"from": "climate_change", "to": "sea_level_rise", "relation": "causes"},
            {"from": "climate_change", "to": "extreme_weather", "relation": "causes"},
            {"from": "renewable_energy", "to": "greenhouse_gases", "relation": "reduces"},
        ]

        self.edges = demo_edges

        logger.info(f"Initialized demo graph: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def add_document_concepts(self, document_text: str) -> List[str]:
        """
        Extract concepts from document and add to graph

        Automatically creates new nodes for important words/phrases.
        Returns list of concept IDs added.
        """
        import re

        # Extract potential concepts: capitalized words, important terms
        # 1. Find capitalized words (likely proper nouns/concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', document_text)

        # 2. Find multi-word phrases in title case
        phrases = re.findall(r'\b[A-Z][a-z]+\s+[a-z]+(?:\s+[a-z]+)?\b', document_text)

        # 3. Extract words that appear multiple times (important terms)
        words = re.findall(r'\b[a-z]{4,}\b', document_text.lower())
        word_freq = {}
        for word in words:
            # Skip common words
            if word in ['that', 'this', 'with', 'from', 'have', 'been', 'were', 'they', 'their', 'would', 'could', 'should']:
                continue
            word_freq[word] = word_freq.get(word, 0) + 1

        # Get top frequent words (appear 3+ times)
        frequent_words = [w for w, count in word_freq.items() if count >= 3]

        # Combine all candidate concepts
        all_candidates = capitalized + phrases + frequent_words[:10]

        # Create nodes for new concepts
        created_concepts = []
        for concept_text in all_candidates[:15]:  # Limit to top 15 concepts
            # Create node ID from text
            concept_id = concept_text.lower().replace(' ', '_')

            # Skip if already exists
            if concept_id in self.nodes:
                created_concepts.append(concept_id)
                continue

            # Create new node
            self.nodes[concept_id] = {
                "id": concept_id,
                "text": f"{concept_text} (extracted from document)",
                "type": "concept",
                "source": "document_upload"
            }
            created_concepts.append(concept_id)
            logger.info(f"Created new node: {concept_id}")

        # Create edges between co-occurring concepts
        for i, concept1 in enumerate(created_concepts[:5]):
            for concept2 in created_concepts[i+1:i+3]:
                # Check if edge already exists
                edge_exists = any(
                    (e["from"] == concept1 and e["to"] == concept2) or
                    (e["from"] == concept2 and e["to"] == concept1)
                    for e in self.edges
                )
                if not edge_exists:
                    self.edges.append({
                        "from": concept1,
                        "to": concept2,
                        "relation": "related_to"
                    })

        return created_concepts

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID"""
        return self.nodes.get(node_id)

    def get_node_text(self, node_id: str) -> str:
        """Get node text content"""
        node = self.nodes.get(node_id)
        return node["text"] if node else node_id

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get 1-hop neighbors"""
        neighbors = []
        for edge in self.edges:
            if edge["from"] == node_id:
                neighbors.append(edge["to"])
            elif edge["to"] == node_id:
                neighbors.append(edge["from"])
        return list(set(neighbors))

    def get_node_degree(self, node_id: str) -> int:
        """Get node degree"""
        return len(self.get_neighbors(node_id))

    def get_candidate_hops(self, node_id: str) -> List[Dict[str, Any]]:
        """Get candidate next hops from current node"""
        candidates = []
        for edge in self.edges:
            if edge["from"] == node_id:
                candidates.append({
                    "node": edge["to"],
                    "relation": edge["relation"],
                    "edge": edge,
                })
        return candidates


class SimpleEmbedder:
    """Simple embedding service using hash-based vectors (demo mode)"""

    def __init__(self, dim: int = 384):
        self.dim = dim

    async def embed(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from text hash"""
        # Use hash for deterministic but varied embeddings
        hash_val = hash(text)
        np.random.seed(hash_val % (2**31))
        embedding = np.random.randn(self.dim)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


# Global demo instances
_demo_graph: Optional[InMemoryGraph] = None
_demo_embedder: Optional[SimpleEmbedder] = None


def get_demo_graph() -> InMemoryGraph:
    """Get or create demo graph instance"""
    global _demo_graph
    if _demo_graph is None:
        _demo_graph = InMemoryGraph()
    return _demo_graph


def get_demo_embedder() -> SimpleEmbedder:
    """Get or create demo embedder instance"""
    global _demo_embedder
    if _demo_embedder is None:
        _demo_embedder = SimpleEmbedder()
    return _demo_embedder
