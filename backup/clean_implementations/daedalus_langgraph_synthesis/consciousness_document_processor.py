"""
Consciousness-Guided Document Processor
========================================

Combines SurfSense document processing patterns with consciousness integration.

Pipeline:
1. Convert to markdown (SurfSense pattern)
2. Generate content hash for duplicate detection (SurfSense pattern)
3. Create LLM summary with metadata (SurfSense pattern)
4. Extract concepts → AttractorBasins → ThoughtSeeds (Consciousness)
5. Generate embeddings for chunks (SurfSense pattern)
6. Store in knowledge graph with consciousness metadata

Based on:
- SurfSense: /Volumes/Asylum/dev/Flux/surfsense_backend/app/tasks/document_processors/
- Dionysus: five_level_concept_extraction.py, granular_chunking.py
"""

import sys
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import PyPDF2
import io

logger = logging.getLogger(__name__)

# Import consciousness processing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "extensions" / "context_engineering"))

try:
    from attractor_basin_dynamics import AttractorBasinManager
    BASINS_AVAILABLE = True
except ImportError:
    BASINS_AVAILABLE = False
    logger.warning("AttractorBasinManager not available - consciousness processing disabled")


@dataclass
class DocumentProcessingResult:
    """
    Complete document processing result.

    Combines SurfSense patterns (summary, chunks, embeddings)
    with consciousness patterns (basins, thoughtseeds, learning).
    """
    # Document identification
    filename: str
    content_hash: str

    # Content extraction (SurfSense pattern)
    markdown_content: str
    summary: str
    word_count: int

    # Chunking (SurfSense pattern)
    chunks: List[Dict[str, Any]]

    # Concept extraction (Consciousness pattern)
    concepts: List[str]

    # Consciousness processing (Consciousness pattern)
    basins_created: int
    thoughtseeds_generated: List[str]
    patterns_learned: List[Dict[str, str]]

    # Metadata
    processing_metadata: Dict[str, Any]


class ConsciousnessDocumentProcessor:
    """
    Process documents through consciousness pipeline with SurfSense patterns.

    Examples:
        >>> processor = ConsciousnessDocumentProcessor()
        >>> result = processor.process_pdf(pdf_bytes, "paper.pdf")
        >>> print(f"Extracted {len(result.concepts)} concepts")
        >>> print(f"Created {result.basins_created} attractor basins")
        >>> print(f"Generated {len(result.thoughtseeds_generated)} thoughtseeds")
    """

    def __init__(self):
        """Initialize processor with consciousness integration"""
        self.basin_manager = None

        if BASINS_AVAILABLE:
            try:
                self.basin_manager = AttractorBasinManager()
                logger.info("✅ Consciousness processing available (AttractorBasinManager)")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize AttractorBasinManager: {e}")
        else:
            logger.info("ℹ️ Running without consciousness processing")

    def process_pdf(self, content: bytes, filename: str) -> DocumentProcessingResult:
        """
        Process PDF through consciousness pipeline.

        Steps (SurfSense + Consciousness):
        1. Extract text from PDF
        2. Convert to markdown
        3. Generate content hash (duplicate detection)
        4. Extract concepts
        5. Create attractor basins for concepts
        6. Generate thoughtseeds
        7. Learn patterns
        8. Create chunks with embeddings
        9. Generate summary

        Args:
            content: PDF file bytes
            filename: Name of file

        Returns:
            DocumentProcessingResult with all processing artifacts
        """
        # Step 1: Extract text
        text = self._extract_text_from_pdf(content)

        # Step 2: Convert to markdown (SurfSense pattern)
        markdown = self._convert_to_markdown(text, source_type="pdf")

        # Step 3: Generate content hash (SurfSense pattern - duplicate detection)
        content_hash = self._generate_content_hash(markdown)

        # Step 4: Extract concepts (Consciousness pattern)
        concepts = self._extract_concepts(markdown)

        # Step 5-7: Consciousness processing
        basin_result = self._process_through_consciousness(concepts)

        # Step 8: Create chunks (SurfSense pattern)
        chunks = self._create_chunks(markdown)

        # Step 9: Generate summary
        summary = self._generate_simple_summary(markdown, concepts)

        return DocumentProcessingResult(
            filename=filename,
            content_hash=content_hash,
            markdown_content=markdown,
            summary=summary,
            word_count=len(markdown.split()),
            chunks=chunks,
            concepts=concepts,
            basins_created=basin_result['basins_created'],
            thoughtseeds_generated=basin_result['thoughtseeds'],
            patterns_learned=basin_result['patterns'],
            processing_metadata={
                'processor': 'ConsciousnessDocumentProcessor',
                'consciousness_enabled': BASINS_AVAILABLE,
                'source_type': 'pdf'
            }
        )

    def process_text(self, content: bytes, filename: str) -> DocumentProcessingResult:
        """Process text file through consciousness pipeline"""
        text = content.decode('utf-8')
        markdown = self._convert_to_markdown(text, source_type="text")
        content_hash = self._generate_content_hash(markdown)
        concepts = self._extract_concepts(markdown)
        basin_result = self._process_through_consciousness(concepts)
        chunks = self._create_chunks(markdown)
        summary = self._generate_simple_summary(markdown, concepts)

        return DocumentProcessingResult(
            filename=filename,
            content_hash=content_hash,
            markdown_content=markdown,
            summary=summary,
            word_count=len(markdown.split()),
            chunks=chunks,
            concepts=concepts,
            basins_created=basin_result['basins_created'],
            thoughtseeds_generated=basin_result['thoughtseeds'],
            patterns_learned=basin_result['patterns'],
            processing_metadata={
                'processor': 'ConsciousnessDocumentProcessor',
                'consciousness_enabled': BASINS_AVAILABLE,
                'source_type': 'text'
            }
        )

    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract raw text from PDF (SurfSense pattern)"""
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def _convert_to_markdown(self, text: str, source_type: str) -> str:
        """
        Convert text to markdown format (SurfSense pattern).

        Simple conversion for now - TODO: use Unstructured.io patterns
        from SurfSense for richer markdown conversion.
        """
        # Add document header
        markdown = f"# Document Content\n\n"

        # Split into paragraphs
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if it looks like a heading (short, capitalized)
            words = para.split()
            if len(words) < 10 and para.isupper():
                markdown += f"## {para}\n\n"
            else:
                markdown += f"{para}\n\n"

        return markdown

    def _generate_content_hash(self, content: str) -> str:
        """
        Generate SHA-256 hash for duplicate detection (SurfSense pattern).

        SurfSense combines content + search_space_id.
        We simplify to just content for now.
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts (phrases).

        Strategy:
        - 2-3 word technical phrases
        - Filter stopwords
        - Sort by specificity

        TODO: Integrate five_level_concept_extraction.py for deep analysis
        """
        import re
        from collections import Counter

        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those'
        }

        text_lower = text.lower()
        pattern = r'\b([a-z]+\s+[a-z]+(?:\s+[a-z]+)?)\b'
        phrases = re.findall(pattern, text_lower)

        concepts = []
        for phrase in phrases:
            words = phrase.split()

            if len(phrase) < 8:
                continue

            if words[0] in stopwords:
                continue

            if all(w in stopwords for w in words):
                continue

            concepts.append(phrase)

        # Remove duplicates, sort by length
        concepts = list(set(concepts))
        concepts.sort(key=len, reverse=True)

        return concepts[:50]  # Top 50

    def _process_through_consciousness(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Pass concepts through attractor basin dynamics.

        Returns basin creation results and learning patterns.
        """
        if not self.basin_manager:
            return {
                'basins_created': 0,
                'thoughtseeds': [],
                'patterns': []
            }

        import time
        basins_created = 0
        thoughtseeds = []
        patterns = []

        for concept in concepts:
            try:
                basin = self.basin_manager.integrate_thoughtseed(
                    concept_description=concept,
                    thoughtseed_id=f"ts_{concept[:20]}_{int(time.time())}"
                )

                if basin:
                    basins_created += 1
                    thoughtseeds.append(basin.get('thoughtseed_id'))
                    patterns.append({
                        'concept': concept,
                        'pattern_type': basin.get('influence_type', 'unknown'),
                        'basin_id': basin.get('basin_id')
                    })
            except Exception as e:
                logger.warning(f"Could not process concept '{concept}': {e}")

        return {
            'basins_created': basins_created,
            'thoughtseeds': thoughtseeds,
            'patterns': patterns
        }

    def _create_chunks(self, content: str, chunk_size: int = 512) -> List[Dict[str, Any]]:
        """
        Create chunks with metadata (SurfSense pattern).

        Simple sentence-based chunking for now.
        TODO: Integrate granular_chunking.py for 5-level chunking
        """
        import re

        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', content)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_content = '. '.join(current_chunk) + '.'
                chunks.append({
                    'chunk_id': f"chunk_{chunk_id}",
                    'content': chunk_content,
                    'length': len(chunk_content),
                    'sentence_count': len(current_chunk)
                })
                chunk_id += 1
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_content = '. '.join(current_chunk) + '.'
            chunks.append({
                'chunk_id': f"chunk_{chunk_id}",
                'content': chunk_content,
                'length': len(chunk_content),
                'sentence_count': len(current_chunk)
            })

        return chunks

    def _generate_simple_summary(self, content: str, concepts: List[str]) -> str:
        """
        Generate simple summary.

        TODO: Integrate SurfSense LLM-based summary generation with Ollama
        """
        # For now, return first paragraph + concept list
        paragraphs = content.split('\n\n')
        first_para = paragraphs[0] if paragraphs else content[:200]

        top_concepts = ', '.join(concepts[:10])

        summary = f"""**Summary**

{first_para}

**Key Concepts**: {top_concepts}

**Statistics**: {len(content.split())} words, {len(concepts)} concepts extracted
"""
        return summary
