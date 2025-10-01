"""
Ultra-Granular Chunking System
==============================

Implements five levels of granular text chunking for consciousness-guided analysis:
- Level 1: Sentence-level chunks for atomic concept extraction
- Level 2: 2-3 sentence chunks for relationship mapping
- Level 3: Paragraph-level chunks for composite concepts
- Level 4: Section-level chunks for contextual frameworks
- Level 5: Multi-section chunks for narrative analysis

Uses LangChain text splitters with configurable overlap management.
Implements Spec-022 Task 1.3 requirements.
"""

import asyncio
import re
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import hashlib

# LangChain imports for text splitting
try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        SentenceTransformersTokenTextSplitter,
        TokenTextSplitter
    )
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# NLTK for sentence tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class ChunkingLevel(Enum):
    """Five levels of granular chunking"""
    SENTENCE = 1          # Individual sentences for atomic concepts
    RELATIONSHIP = 2      # 2-3 sentences for relationship mapping
    PARAGRAPH = 3         # Paragraph-level for composite concepts
    SECTION = 4          # Section-level for contextual frameworks
    NARRATIVE = 5         # Multi-section for narrative analysis

class ChunkType(Enum):
    """Types of content chunks"""
    ATOMIC = "atomic"                    # Single concept chunks
    RELATIONAL = "relational"           # Relationship-focused chunks
    COMPOSITE = "composite"             # Multi-concept chunks
    CONTEXTUAL = "contextual"           # Context-preserving chunks
    NARRATIVE = "narrative"             # Story/flow chunks

@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: str
    level: ChunkingLevel
    chunk_type: ChunkType
    start_position: int
    end_position: int
    sentence_count: int
    word_count: int
    token_count: int
    overlap_info: Dict[str, Any] = field(default_factory=dict)
    structure_info: Dict[str, Any] = field(default_factory=dict)
    processing_hints: List[str] = field(default_factory=list)

@dataclass
class ProcessedChunk:
    """A processed text chunk with metadata"""
    content: str
    metadata: ChunkMetadata
    context_chunks: List[str] = field(default_factory=list)  # Related chunks for context
    processing_priority: int = 1  # Higher = more important
    consciousness_domains: List[str] = field(default_factory=list)  # Target domains

@dataclass
class ChunkingConfig:
    """Configuration for chunking process"""
    level: ChunkingLevel
    chunk_size: int = 512
    chunk_overlap: int = 100
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", "? ", "! "])
    preserve_structure: bool = True
    include_context: bool = True
    min_chunk_size: int = 50
    max_chunk_size: int = 2048

@dataclass
class ChunkingResult:
    """Result of the chunking process"""
    success: bool
    chunks: List[ProcessedChunk] = field(default_factory=list)
    level_stats: Dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

class SentenceChunker:
    """Level 1: Sentence-level chunking for atomic concepts"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.config.chunk_size = 200  # Typical sentence length
        self.config.chunk_overlap = 50  # Word overlap between sentences
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[ProcessedChunk]:
        """Chunk text into individual sentences"""
        chunks = []
        
        # Use NLTK for sentence tokenization if available
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
            except LookupError:
                # Download punkt if not available
                nltk.download('punkt', quiet=True)
                sentences = sent_tokenize(text)
        else:
            # Fallback to regex-based sentence splitting
            sentences = self._regex_sentence_split(text)
        
        # Process each sentence
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < self.config.min_chunk_size:
                continue
            
            # Create chunk metadata
            chunk_id = hashlib.md5(f"sent_{i}_{sentence}".encode()).hexdigest()[:12]
            
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                level=ChunkingLevel.SENTENCE,
                chunk_type=ChunkType.ATOMIC,
                start_position=text.find(sentence),
                end_position=text.find(sentence) + len(sentence),
                sentence_count=1,
                word_count=len(sentence.split()),
                token_count=len(sentence.split()),
                structure_info={"sentence_index": i, "is_complete": sentence.endswith(('.', '!', '?'))},
                processing_hints=["atomic_concept_extraction", "entity_recognition"]
            )
            
            # Add context from neighboring sentences
            context_chunks = []
            if i > 0:
                context_chunks.append(sentences[i-1])
            if i < len(sentences) - 1:
                context_chunks.append(sentences[i+1])
            
            chunk = ProcessedChunk(
                content=sentence,
                metadata=chunk_metadata,
                context_chunks=context_chunks,
                processing_priority=2,  # High priority for atomic concepts
                consciousness_domains=["atomic_concepts", "entity_detection"]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _regex_sentence_split(self, text: str) -> List[str]:
        """Fallback regex-based sentence splitting"""
        # Simple sentence splitting pattern
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

class RelationshipChunker:
    """Level 2: 2-3 sentence chunks for relationship mapping"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.config.chunk_size = 400  # 2-3 sentences
        self.config.chunk_overlap = 100  # Overlap with neighboring chunks
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[ProcessedChunk]:
        """Chunk text into 2-3 sentence groups for relationship detection"""
        chunks = []
        
        # Get sentences first
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
            except LookupError:
                nltk.download('punkt', quiet=True)
                sentences = sent_tokenize(text)
        else:
            sentences = self._regex_sentence_split(text)
        
        # Group sentences into 2-3 sentence chunks with overlap
        chunk_size = 3  # sentences per chunk
        overlap = 1     # sentence overlap
        
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_sentences = sentences[i:i + chunk_size]
            if not chunk_sentences:
                continue
            
            chunk_text = ' '.join(chunk_sentences)
            if len(chunk_text) < self.config.min_chunk_size:
                continue
            
            # Create chunk metadata
            chunk_id = hashlib.md5(f"rel_{i}_{chunk_text[:50]}".encode()).hexdigest()[:12]
            
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                level=ChunkingLevel.RELATIONSHIP,
                chunk_type=ChunkType.RELATIONAL,
                start_position=text.find(chunk_sentences[0]),
                end_position=text.find(chunk_sentences[-1]) + len(chunk_sentences[-1]),
                sentence_count=len(chunk_sentences),
                word_count=len(chunk_text.split()),
                token_count=len(chunk_text.split()),
                overlap_info={
                    "prev_overlap": 1 if i > 0 else 0,
                    "next_overlap": 1 if i + chunk_size < len(sentences) else 0
                },
                structure_info={"sentence_range": [i, i + len(chunk_sentences)]},
                processing_hints=["relationship_mapping", "causality_detection", "concept_connections"]
            )
            
            # Add broader context
            context_chunks = []
            if i > chunk_size:
                context_chunks.append(' '.join(sentences[i-chunk_size:i]))
            if i + 2*chunk_size < len(sentences):
                context_chunks.append(' '.join(sentences[i+chunk_size:i+2*chunk_size]))
            
            chunk = ProcessedChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                context_chunks=context_chunks,
                processing_priority=3,  # High priority for relationships
                consciousness_domains=["relationship_mapping", "causal_inference"]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _regex_sentence_split(self, text: str) -> List[str]:
        """Fallback regex-based sentence splitting"""
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

class ParagraphChunker:
    """Level 3: Paragraph-level chunks for composite concepts"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.config.chunk_size = 1024  # Typical paragraph length
        self.config.chunk_overlap = 200  # Overlap for context preservation
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[ProcessedChunk]:
        """Chunk text into paragraphs for composite concept extraction"""
        chunks = []
        
        # Split by paragraphs (double newlines)
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < self.config.min_chunk_size:
                continue
            
            # Handle very large paragraphs by splitting further
            if len(paragraph) > self.config.max_chunk_size:
                sub_chunks = await self._split_large_paragraph(paragraph, i)
                chunks.extend(sub_chunks)
                continue
            
            # Create chunk metadata
            chunk_id = hashlib.md5(f"para_{i}_{paragraph[:50]}".encode()).hexdigest()[:12]
            
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                level=ChunkingLevel.PARAGRAPH,
                chunk_type=ChunkType.COMPOSITE,
                start_position=text.find(paragraph),
                end_position=text.find(paragraph) + len(paragraph),
                sentence_count=len(paragraph.split('. ')),
                word_count=len(paragraph.split()),
                token_count=len(paragraph.split()),
                structure_info={
                    "paragraph_index": i,
                    "has_headers": bool(re.search(r'^#+\s', paragraph, re.MULTILINE)),
                    "has_lists": bool(re.search(r'^\s*[-*+]\s', paragraph, re.MULTILINE))
                },
                processing_hints=["composite_concepts", "theme_detection", "argument_structure"]
            )
            
            # Add context from neighboring paragraphs
            context_chunks = []
            if i > 0:
                context_chunks.append(paragraphs[i-1][:200] + "...")
            if i < len(paragraphs) - 1:
                context_chunks.append(paragraphs[i+1][:200] + "...")
            
            chunk = ProcessedChunk(
                content=paragraph,
                metadata=chunk_metadata,
                context_chunks=context_chunks,
                processing_priority=2,
                consciousness_domains=["composite_concepts", "thematic_analysis"]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    async def _split_large_paragraph(self, paragraph: str, para_index: int) -> List[ProcessedChunk]:
        """Split very large paragraphs into manageable chunks"""
        chunks = []
        
        # Use sentence-based splitting for large paragraphs
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(paragraph)
            except LookupError:
                nltk.download('punkt', quiet=True)
                sentences = sent_tokenize(paragraph)
        else:
            sentences = paragraph.split('. ')
        
        # Group sentences into chunks
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk + sentence) > self.config.chunk_size and current_chunk:
                # Create chunk from current content
                chunk_id = hashlib.md5(f"para_{para_index}_{chunk_index}_{current_chunk[:50]}".encode()).hexdigest()[:12]
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    level=ChunkingLevel.PARAGRAPH,
                    chunk_type=ChunkType.COMPOSITE,
                    start_position=0,  # Would need more complex calculation
                    end_position=len(current_chunk),
                    sentence_count=len(current_chunk.split('. ')),
                    word_count=len(current_chunk.split()),
                    token_count=len(current_chunk.split()),
                    structure_info={"sub_paragraph": True, "parent_paragraph": para_index},
                    processing_hints=["composite_concepts", "theme_detection"]
                )
                
                chunk = ProcessedChunk(
                    content=current_chunk.strip(),
                    metadata=chunk_metadata,
                    processing_priority=2,
                    consciousness_domains=["composite_concepts"]
                )
                
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.config.chunk_overlap > 0:
                    overlap_words = current_chunk.split()[-self.config.chunk_overlap//10:]  # Rough word overlap
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence
                else:
                    current_chunk = sentence
                
                chunk_index += 1
            else:
                current_chunk += ' ' + sentence
        
        # Add final chunk if any content remains
        if current_chunk.strip():
            chunk_id = hashlib.md5(f"para_{para_index}_{chunk_index}_{current_chunk[:50]}".encode()).hexdigest()[:12]
            
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                level=ChunkingLevel.PARAGRAPH,
                chunk_type=ChunkType.COMPOSITE,
                start_position=0,
                end_position=len(current_chunk),
                sentence_count=len(current_chunk.split('. ')),
                word_count=len(current_chunk.split()),
                token_count=len(current_chunk.split()),
                structure_info={"sub_paragraph": True, "parent_paragraph": para_index},
                processing_hints=["composite_concepts"]
            )
            
            chunk = ProcessedChunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                processing_priority=2,
                consciousness_domains=["composite_concepts"]
            )
            
            chunks.append(chunk)
        
        return chunks

class SectionChunker:
    """Level 4: Section-level chunks for contextual frameworks"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.config.chunk_size = 2048  # Section-level content
        self.config.chunk_overlap = 300
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[ProcessedChunk]:
        """Chunk text into sections for contextual framework extraction"""
        chunks = []
        
        # Detect sections by headers or major breaks
        sections = self._detect_sections(text)
        
        for i, section in enumerate(sections):
            if len(section["content"]) < self.config.min_chunk_size:
                continue
            
            # Handle very large sections
            if len(section["content"]) > self.config.max_chunk_size:
                sub_chunks = await self._split_large_section(section, i)
                chunks.extend(sub_chunks)
                continue
            
            chunk_id = hashlib.md5(f"sect_{i}_{section['title'][:30]}".encode()).hexdigest()[:12]
            
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                level=ChunkingLevel.SECTION,
                chunk_type=ChunkType.CONTEXTUAL,
                start_position=section.get("start_pos", 0),
                end_position=section.get("end_pos", len(section["content"])),
                sentence_count=len(section["content"].split('. ')),
                word_count=len(section["content"].split()),
                token_count=len(section["content"].split()),
                structure_info={
                    "section_title": section["title"],
                    "section_level": section.get("level", 1),
                    "has_subsections": section.get("has_subsections", False)
                },
                processing_hints=["contextual_frameworks", "domain_knowledge", "section_analysis"]
            )
            
            # Add context from neighboring sections
            context_chunks = []
            if i > 0:
                context_chunks.append(f"Previous: {sections[i-1]['title']}")
            if i < len(sections) - 1:
                context_chunks.append(f"Next: {sections[i+1]['title']}")
            
            chunk = ProcessedChunk(
                content=section["content"],
                metadata=chunk_metadata,
                context_chunks=context_chunks,
                processing_priority=1,  # Medium priority
                consciousness_domains=["contextual_frameworks", "domain_modeling"]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect sections in text based on headers and structure"""
        sections = []
        
        # Look for markdown headers
        lines = text.split('\n')
        current_section = {"title": "Introduction", "content": "", "level": 0, "start_pos": 0}
        
        for line_num, line in enumerate(lines):
            # Check for markdown headers
            if line.strip().startswith('#'):
                # Save previous section if it has content
                if current_section["content"].strip():
                    current_section["end_pos"] = len(current_section["content"])
                    sections.append(current_section)
                
                # Start new section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                current_section = {
                    "title": title,
                    "content": "",
                    "level": level,
                    "start_pos": sum(len(l) + 1 for l in lines[:line_num]),
                    "has_subsections": False
                }
            else:
                current_section["content"] += line + '\n'
        
        # Add final section
        if current_section["content"].strip():
            current_section["end_pos"] = len(current_section["content"])
            sections.append(current_section)
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections = [{
                "title": "Full Document",
                "content": text,
                "level": 1,
                "start_pos": 0,
                "end_pos": len(text),
                "has_subsections": False
            }]
        
        return sections
    
    async def _split_large_section(self, section: Dict[str, Any], section_index: int) -> List[ProcessedChunk]:
        """Split large sections into manageable chunks"""
        chunks = []
        content = section["content"]
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) > self.config.chunk_size and current_chunk:
                # Create chunk
                chunk_id = hashlib.md5(f"sect_{section_index}_{chunk_index}_{current_chunk[:50]}".encode()).hexdigest()[:12]
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    level=ChunkingLevel.SECTION,
                    chunk_type=ChunkType.CONTEXTUAL,
                    start_position=0,
                    end_position=len(current_chunk),
                    sentence_count=len(current_chunk.split('. ')),
                    word_count=len(current_chunk.split()),
                    token_count=len(current_chunk.split()),
                    structure_info={
                        "parent_section": section["title"],
                        "sub_section": True,
                        "chunk_index": chunk_index
                    },
                    processing_hints=["contextual_frameworks"]
                )
                
                chunk = ProcessedChunk(
                    content=current_chunk.strip(),
                    metadata=chunk_metadata,
                    processing_priority=1,
                    consciousness_domains=["contextual_frameworks"]
                )
                
                chunks.append(chunk)
                current_chunk = paragraph
                chunk_index += 1
            else:
                current_chunk += '\n\n' + paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = hashlib.md5(f"sect_{section_index}_{chunk_index}_{current_chunk[:50]}".encode()).hexdigest()[:12]
            
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                level=ChunkingLevel.SECTION,
                chunk_type=ChunkType.CONTEXTUAL,
                start_position=0,
                end_position=len(current_chunk),
                sentence_count=len(current_chunk.split('. ')),
                word_count=len(current_chunk.split()),
                token_count=len(current_chunk.split()),
                structure_info={
                    "parent_section": section["title"],
                    "sub_section": True,
                    "chunk_index": chunk_index
                },
                processing_hints=["contextual_frameworks"]
            )
            
            chunk = ProcessedChunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                processing_priority=1,
                consciousness_domains=["contextual_frameworks"]
            )
            
            chunks.append(chunk)
        
        return chunks

class NarrativeChunker:
    """Level 5: Multi-section chunks for narrative analysis"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.config.chunk_size = 4096  # Large narrative chunks
        self.config.chunk_overlap = 500
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[ProcessedChunk]:
        """Chunk text into narrative flows for story/argument analysis"""
        chunks = []
        
        # Detect narrative flows and major thematic breaks
        narratives = self._detect_narratives(text)
        
        for i, narrative in enumerate(narratives):
            if len(narrative["content"]) < self.config.min_chunk_size:
                continue
            
            chunk_id = hashlib.md5(f"narr_{i}_{narrative['theme'][:30]}".encode()).hexdigest()[:12]
            
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                level=ChunkingLevel.NARRATIVE,
                chunk_type=ChunkType.NARRATIVE,
                start_position=narrative.get("start_pos", 0),
                end_position=narrative.get("end_pos", len(narrative["content"])),
                sentence_count=len(narrative["content"].split('. ')),
                word_count=len(narrative["content"].split()),
                token_count=len(narrative["content"].split()),
                structure_info={
                    "narrative_theme": narrative["theme"],
                    "narrative_type": narrative.get("type", "general"),
                    "flow_direction": narrative.get("flow", "linear")
                },
                processing_hints=["narrative_analysis", "argument_flow", "story_structure", "methodology"]
            )
            
            # Add full document context for narrative analysis
            context_chunks = [
                f"Document theme: {narrative.get('document_theme', 'unknown')}",
                f"Narrative position: {i+1}/{len(narratives)}"
            ]
            
            chunk = ProcessedChunk(
                content=narrative["content"],
                metadata=chunk_metadata,
                context_chunks=context_chunks,
                processing_priority=0,  # Lowest priority, but important for holistic understanding
                consciousness_domains=["narrative_structure", "argument_analysis", "methodology_detection"]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _detect_narratives(self, text: str) -> List[Dict[str, Any]]:
        """Detect narrative flows in text"""
        # For now, implement simple narrative detection
        # In future, could use more sophisticated NLP techniques
        
        narratives = []
        
        # Split by major sections first
        sections = text.split('\n\n\n')  # Triple newline for major breaks
        if len(sections) == 1:
            sections = text.split('\n\n')  # Fallback to paragraph breaks
        
        # Group sections into narrative flows
        chunk_size = max(1, len(sections) // 3)  # Aim for 3-4 narrative chunks
        
        for i in range(0, len(sections), chunk_size):
            narrative_sections = sections[i:i + chunk_size]
            content = '\n\n'.join(narrative_sections)
            
            if len(content.strip()) < self.config.min_chunk_size:
                continue
            
            # Extract theme from first few words or section headers
            theme = self._extract_theme(content)
            
            narrative = {
                "theme": theme,
                "content": content,
                "type": self._classify_narrative_type(content),
                "flow": "linear",
                "start_pos": 0,  # Could calculate actual positions
                "end_pos": len(content)
            }
            
            narratives.append(narrative)
        
        # If no narratives found, treat whole text as single narrative
        if not narratives:
            theme = self._extract_theme(text)
            narratives = [{
                "theme": theme,
                "content": text,
                "type": "complete_document",
                "flow": "linear",
                "start_pos": 0,
                "end_pos": len(text)
            }]
        
        return narratives
    
    def _extract_theme(self, content: str) -> str:
        """Extract main theme from content"""
        # Look for headers first
        lines = content.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            if line.strip().startswith('#'):
                return line.lstrip('#').strip()
        
        # Fallback to first meaningful words
        words = content.split()[:10]
        meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
        
        if meaningful_words:
            return ' '.join(meaningful_words[:3])
        else:
            return "Unknown Theme"
    
    def _classify_narrative_type(self, content: str) -> str:
        """Classify the type of narrative"""
        content_lower = content.lower()
        
        # Check for different narrative types
        if any(word in content_lower for word in ["method", "procedure", "algorithm", "step"]):
            return "methodology"
        elif any(word in content_lower for word in ["result", "finding", "conclusion", "evidence"]):
            return "findings"
        elif any(word in content_lower for word in ["background", "introduction", "literature", "previous"]):
            return "background"
        elif any(word in content_lower for word in ["discussion", "interpretation", "implication", "significance"]):
            return "discussion"
        else:
            return "general"

class UltraGranularChunker:
    """Main service for ultra-granular text chunking at all five levels"""
    
    def __init__(self):
        self.chunkers = {}
        self._initialize_chunkers()
    
    def _initialize_chunkers(self):
        """Initialize all chunker levels"""
        # Create default configs for each level
        configs = {
            ChunkingLevel.SENTENCE: ChunkingConfig(
                level=ChunkingLevel.SENTENCE,
                chunk_size=200,
                chunk_overlap=50,
                separators=[". ", "! ", "? "],
                min_chunk_size=20,
                max_chunk_size=400
            ),
            ChunkingLevel.RELATIONSHIP: ChunkingConfig(
                level=ChunkingLevel.RELATIONSHIP,
                chunk_size=400,
                chunk_overlap=100,
                separators=[". ", "\n"],
                min_chunk_size=100,
                max_chunk_size=800
            ),
            ChunkingLevel.PARAGRAPH: ChunkingConfig(
                level=ChunkingLevel.PARAGRAPH,
                chunk_size=1024,
                chunk_overlap=200,
                separators=["\n\n", "\n"],
                min_chunk_size=200,
                max_chunk_size=2048
            ),
            ChunkingLevel.SECTION: ChunkingConfig(
                level=ChunkingLevel.SECTION,
                chunk_size=2048,
                chunk_overlap=300,
                separators=["\n\n\n", "\n\n"],
                min_chunk_size=500,
                max_chunk_size=4096
            ),
            ChunkingLevel.NARRATIVE: ChunkingConfig(
                level=ChunkingLevel.NARRATIVE,
                chunk_size=4096,
                chunk_overlap=500,
                separators=["\n\n\n\n", "\n\n\n"],
                min_chunk_size=1000,
                max_chunk_size=8192
            )
        }
        
        # Initialize chunkers
        self.chunkers = {
            ChunkingLevel.SENTENCE: SentenceChunker(configs[ChunkingLevel.SENTENCE]),
            ChunkingLevel.RELATIONSHIP: RelationshipChunker(configs[ChunkingLevel.RELATIONSHIP]),
            ChunkingLevel.PARAGRAPH: ParagraphChunker(configs[ChunkingLevel.PARAGRAPH]),
            ChunkingLevel.SECTION: SectionChunker(configs[ChunkingLevel.SECTION]),
            ChunkingLevel.NARRATIVE: NarrativeChunker(configs[ChunkingLevel.NARRATIVE])
        }
    
    async def chunk_document(self, 
                           text: str,
                           levels: List[ChunkingLevel] = None,
                           metadata: Dict[str, Any] = None) -> Dict[ChunkingLevel, ChunkingResult]:
        """
        Chunk a document at specified granularity levels
        
        Args:
            text: Input text to chunk
            levels: List of chunking levels to apply (default: all levels)
            metadata: Additional metadata for processing
            
        Returns:
            Dictionary mapping each level to its chunking results
        """
        if levels is None:
            levels = list(ChunkingLevel)
        
        results = {}
        
        for level in levels:
            start_time = datetime.now()
            
            try:
                chunker = self.chunkers[level]
                chunks = await chunker.chunk_text(text, metadata)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate statistics
                level_stats = {
                    "total_chunks": len(chunks),
                    "avg_chunk_size": sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
                    "total_words": sum(c.metadata.word_count for c in chunks),
                    "avg_words_per_chunk": sum(c.metadata.word_count for c in chunks) / len(chunks) if chunks else 0
                }
                
                results[level] = ChunkingResult(
                    success=True,
                    chunks=chunks,
                    level_stats=level_stats,
                    processing_time=processing_time
                )
                
                logger.info(f"Level {level.value} chunking completed: {len(chunks)} chunks in {processing_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Chunking failed for level {level.value}: {e}")
                results[level] = ChunkingResult(
                    success=False,
                    error=str(e),
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
        
        return results
    
    async def get_chunks_for_consciousness_processing(self, 
                                                    chunking_results: Dict[ChunkingLevel, ChunkingResult],
                                                    domain_focus: List[str] = None) -> List[ProcessedChunk]:
        """
        Get chunks organized for consciousness processing pipeline
        
        Args:
            chunking_results: Results from chunk_document
            domain_focus: Focus domains for filtering chunks
            
        Returns:
            List of chunks sorted by processing priority and domain relevance
        """
        all_chunks = []
        
        # Collect all chunks from successful results
        for level, result in chunking_results.items():
            if result.success:
                all_chunks.extend(result.chunks)
        
        # Filter by domain focus if specified
        if domain_focus:
            filtered_chunks = []
            for chunk in all_chunks:
                if any(domain in chunk.consciousness_domains for domain in domain_focus):
                    filtered_chunks.append(chunk)
            all_chunks = filtered_chunks
        
        # Sort by processing priority (higher = more important)
        all_chunks.sort(key=lambda x: x.processing_priority, reverse=True)
        
        return all_chunks
    
    def get_chunking_stats(self, results: Dict[ChunkingLevel, ChunkingResult]) -> Dict[str, Any]:
        """Get comprehensive statistics about chunking results"""
        stats = {
            "total_levels_processed": len([r for r in results.values() if r.success]),
            "total_chunks": sum(len(r.chunks) for r in results.values() if r.success),
            "total_processing_time": sum(r.processing_time for r in results.values()),
            "level_breakdown": {},
            "average_chunk_sizes": {},
            "success_rate": len([r for r in results.values() if r.success]) / len(results) if results else 0
        }
        
        for level, result in results.items():
            if result.success:
                stats["level_breakdown"][level.value] = result.level_stats
                stats["average_chunk_sizes"][level.value] = result.level_stats.get("avg_chunk_size", 0)
        
        return stats

# Global service instance
granular_chunker = UltraGranularChunker()

# Test function
async def test_granular_chunking():
    """Test the ultra-granular chunking system"""
    print("🧪 Testing Ultra-Granular Chunking System")
    print("=" * 50)
    
    # Test text with multiple levels of structure
    test_text = """# Neural Networks and Synaptic Plasticity

## Introduction

Artificial neural networks draw inspiration from biological neural systems. These computational models consist of interconnected nodes that process information through weighted connections.

Synaptic plasticity represents the ability of synapses to strengthen or weaken over time. This mechanism underlies learning and memory formation in biological systems.

## Mechanisms of Plasticity

### Long-term Potentiation

Long-term potentiation (LTP) is a persistent strengthening of synapses. It occurs when synaptic connections are repeatedly activated. This process involves NMDA receptor activation and calcium influx.

The molecular cascades triggered by LTP include:
- Protein kinase activation
- CREB-mediated gene expression  
- New protein synthesis
- Structural synaptic modifications

### Long-term Depression

In contrast, long-term depression (LTD) weakens synaptic connections. This mechanism is equally important for learning. It provides a counterbalance to LTP and enables memory refinement.

## Artificial Neural Networks

Modern deep learning architectures implement plasticity-like mechanisms. Backpropagation adjusts connection weights based on error signals. This process mirrors biological learning principles.

### Hebbian Learning

Donald Hebb proposed that neurons that fire together wire together. This principle guides both biological and artificial learning algorithms. It explains how associative memories form.

## Conclusion

Understanding synaptic plasticity informs artificial intelligence development. The parallels between biological and artificial learning continue to drive innovation in neural network design.
"""
    
    chunker = UltraGranularChunker()
    
    # Test all chunking levels
    print("🔄 Processing document at all granularity levels...")
    results = await chunker.chunk_document(test_text)
    
    # Display results for each level
    for level, result in results.items():
        print(f"\n📊 Level {level.value} ({level.name}):")
        if result.success:
            print(f"  ✅ Success: {len(result.chunks)} chunks")
            print(f"  ⏱️  Processing time: {result.processing_time:.3f}s")
            print(f"  📈 Stats: {result.level_stats}")
            
            # Show first chunk as example
            if result.chunks:
                first_chunk = result.chunks[0]
                print(f"  📝 Example chunk:")
                print(f"    ID: {first_chunk.metadata.chunk_id}")
                print(f"    Type: {first_chunk.metadata.chunk_type.value}")
                print(f"    Words: {first_chunk.metadata.word_count}")
                print(f"    Content preview: {first_chunk.content[:100]}...")
                print(f"    Domains: {first_chunk.consciousness_domains}")
        else:
            print(f"  ❌ Failed: {result.error}")
    
    # Get chunks for consciousness processing
    print(f"\n🧠 Organizing for consciousness processing...")
    consciousness_chunks = await chunker.get_chunks_for_consciousness_processing(results)
    print(f"  📋 Total chunks for processing: {len(consciousness_chunks)}")
    
    # Show priority ordering
    priority_groups = {}
    for chunk in consciousness_chunks:
        priority = chunk.processing_priority
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(chunk)
    
    for priority in sorted(priority_groups.keys(), reverse=True):
        chunks = priority_groups[priority]
        print(f"  🎯 Priority {priority}: {len(chunks)} chunks")
    
    # Get overall statistics
    stats = chunker.get_chunking_stats(results)
    print(f"\n📊 Overall Statistics:")
    print(f"  📈 Success rate: {stats['success_rate']:.1%}")
    print(f"  🔢 Total chunks: {stats['total_chunks']}")
    print(f"  ⏱️  Total processing time: {stats['total_processing_time']:.3f}s")
    
    print("\n🎉 Granular chunking test completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_granular_chunking())