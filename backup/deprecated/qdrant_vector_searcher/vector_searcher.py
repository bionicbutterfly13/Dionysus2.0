"""
Vector Searcher - Semantic search using Qdrant vector database
Per Spec 006 FR-002: Search vector database for semantic similarity
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, SearchParams
import logging
import numpy as np

from ...models.response import SearchResult, SearchSource
from ...config.settings import settings

logger = logging.getLogger(__name__)


class VectorSearcher:
    """
    Search Qdrant vector database for semantically similar content.

    Uses embedding-based semantic search to find relevant documents
    and concepts based on meaning rather than keyword matching.
    """

    def __init__(self, client: Optional[QdrantClient] = None, collection_name: str = "documents"):
        """Initialize with Qdrant client."""
        self._client = client
        self._client_initialized = client is not None
        self.collection_name = collection_name
        self.vector_dimensions = settings.VECTOR_DIMENSIONS

    @property
    def client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if not self._client_initialized:
            try:
                self._client = QdrantClient(url=settings.QDRANT_URL)
                self._client_initialized = True
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise
        return self._client

    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search Qdrant for semantically similar results.

        Args:
            query: Natural language search query
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects from Qdrant
        """
        try:
            # Generate embedding for query
            query_vector = await self._generate_embedding(query)

            # Perform vector search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            # Convert to SearchResult objects
            results = []
            for hit in search_results:
                results.append(SearchResult(
                    result_id=str(hit.id),
                    source=SearchSource.QDRANT,
                    content=hit.payload.get("content", hit.payload.get("text", "")),
                    relevance_score=float(hit.score),
                    metadata={
                        "document_id": hit.payload.get("document_id"),
                        "filename": hit.payload.get("filename"),
                        "content_type": hit.payload.get("content_type"),
                        "chunk_index": hit.payload.get("chunk_index"),
                        "timestamp": hit.payload.get("timestamp")
                    },
                    relationships=[]  # Qdrant doesn't have graph relationships
                ))

            return results

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    async def search_with_filter(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search with metadata filters.

        Args:
            query: Natural language search query
            filters: Metadata filters (e.g., {"document_type": "pdf"})
            limit: Maximum results

        Returns:
            Filtered search results
        """
        try:
            query_vector = await self._generate_embedding(query)

            # Build Qdrant filter
            qdrant_filter = None
            if filters:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(key=key, match={"value": value})
                        for key, value in filters.items()
                    ]
                )

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True
            )

            results = []
            for hit in search_results:
                results.append(SearchResult(
                    result_id=str(hit.id),
                    source=SearchSource.QDRANT,
                    content=hit.payload.get("content", hit.payload.get("text", "")),
                    relevance_score=float(hit.score),
                    metadata=hit.payload,
                    relationships=[]
                ))

            return results

        except Exception as e:
            logger.error(f"Qdrant filtered search failed: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Combine semantic and keyword search for best results.

        Args:
            query: Natural language query
            semantic_weight: Weight for vector similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            limit: Maximum results

        Returns:
            Hybrid search results
        """
        # For now, just do semantic search
        # TODO: Implement keyword component when full-text index available
        return await self.search(query, limit)

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Uses simple hash-based embedding for now - in production,
        would use OpenAI embeddings or local model.
        """
        try:
            # Simple hash-based embedding for testing/demo
            # In production: use OpenAI API or local sentence-transformers
            embedding = self._hash_based_embedding(text)
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * self.vector_dimensions

    def _hash_based_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic hash-based embedding.

        This is for testing only - production should use real embeddings.
        """
        # Use text hash to seed random generator for deterministic results
        text_hash = hash(text.lower())
        np.random.seed(abs(text_hash) % (2**31))

        # Generate random vector
        embedding = np.random.randn(self.vector_dimensions)

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    async def collection_exists(self) -> bool:
        """Check if collection exists in Qdrant."""
        try:
            collections = self.client.get_collections()
            return any(col.name == self.collection_name for col in collections.collections)
        except Exception as e:
            logger.error(f"Failed to check collection: {e}")
            return False

    async def create_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            if not await self.collection_exists():
                from qdrant_client.models import Distance, VectorParams

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimensions,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
