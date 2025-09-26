"""
DocumentArtifact Model - T016
Flux Self-Teaching Consciousness Emulator

Represents documents, files, and digital artifacts with semantic analysis
and consciousness-aware processing capabilities.
Constitutional compliance: mock data transparency, evaluation feedback integration.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class DocumentType(str, Enum):
    """Document type classification"""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    RESEARCH_PAPER = "research_paper"
    NOTE = "note"
    CONVERSATION = "conversation"


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class SemanticAnalysis(BaseModel):
    """Semantic analysis results for document"""
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Analysis identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

    # Content Analysis
    key_concepts: List[str] = Field(default_factory=list, description="Extracted key concepts")
    topics: List[str] = Field(default_factory=list, description="Identified topics")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Named entities")

    # Semantic Structure
    semantic_embedding: Optional[List[float]] = Field(None, description="Document embedding vector")
    concept_graph: Dict[str, Any] = Field(default_factory=dict, description="Concept relationship graph")

    # Content Metrics
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Content complexity score")
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Content coherence score")
    information_density: float = Field(default=0.0, ge=0.0, le=1.0, description="Information density measure")

    # Consciousness-Relevant Features
    consciousness_indicators: List[str] = Field(default_factory=list, description="Consciousness-related content indicators")
    meta_cognitive_elements: List[str] = Field(default_factory=list, description="Meta-cognitive content elements")
    learning_potential: float = Field(default=0.0, ge=0.0, le=1.0, description="Educational/learning potential score")


class DocumentArtifact(BaseModel):
    """
    Document artifact model for the Flux consciousness system.

    Represents digital documents and files with semantic analysis,
    consciousness-aware processing, and integration capabilities.
    """

    # Core Identity
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique artifact identifier")
    user_id: str = Field(..., description="Associated user ID")
    journey_id: Optional[str] = Field(None, description="Associated autobiographical journey")

    # Document Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Artifact creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last artifact update")
    modified_at: Optional[datetime] = Field(None, description="Original document modification time")

    # Document Information
    title: str = Field(..., description="Document title or name")
    description: Optional[str] = Field(None, description="Document description")
    document_type: DocumentType = Field(..., description="Type of document")
    file_path: Optional[str] = Field(None, description="File system path")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type")

    # Content Data
    content: Optional[str] = Field(None, description="Extracted text content")
    content_hash: Optional[str] = Field(None, description="Content hash for deduplication")
    raw_content: Optional[bytes] = Field(None, description="Raw binary content")

    # Processing Status
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Current processing status")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    error_log: List[str] = Field(default_factory=list, description="Processing error history")

    # Semantic Analysis
    semantic_analysis: Optional[SemanticAnalysis] = Field(None, description="Semantic analysis results")

    # Version Control
    version: int = Field(default=1, description="Document version number")
    parent_artifact_id: Optional[str] = Field(None, description="Parent artifact for versioning")
    version_history: List[Dict[str, Any]] = Field(default_factory=list, description="Version history")

    # Relationships
    related_artifact_ids: List[str] = Field(default_factory=list, description="Related artifact IDs")
    source_references: List[Dict[str, Any]] = Field(default_factory=list, description="Source references")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Citations from this document")

    # Tags and Classification
    tags: List[str] = Field(default_factory=list, description="User-defined tags")
    categories: List[str] = Field(default_factory=list, description="Auto-classified categories")
    priority: int = Field(default=3, ge=1, le=5, description="Document priority (1-5)")

    # Access Control
    access_permissions: Dict[str, Any] = Field(default_factory=dict, description="Access control settings")
    sharing_settings: Dict[str, Any] = Field(default_factory=dict, description="Sharing configuration")

    # Constitutional Compliance
    mock_data_enabled: bool = Field(default=True, description="Mock data mode for development")
    evaluation_feedback_enabled: bool = Field(default=True, description="Evaluation feedback collection enabled")
    privacy_level: str = Field(default="private", description="Document privacy level")

    # Integration Points
    thoughtseed_references: List[str] = Field(default_factory=list, description="Associated thoughtseed trace IDs")
    event_references: List[str] = Field(default_factory=list, description="Associated event IDs")
    concept_activations: Dict[str, float] = Field(default_factory=dict, description="Activated concepts and strengths")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            bytes: lambda v: v.decode('utf-8', errors='replace') if v else None
        }

    def update_processing_status(self, status: ProcessingStatus, metadata: Dict[str, Any] = None) -> None:
        """Update document processing status"""
        self.processing_status = status
        if metadata:
            self.processing_metadata.update(metadata)
        self.updated_at = datetime.utcnow()

    def add_semantic_analysis(self, key_concepts: List[str], topics: List[str],
                            embedding: List[float] = None, analysis_metadata: Dict[str, Any] = None) -> str:
        """Add semantic analysis results"""
        analysis = SemanticAnalysis(
            key_concepts=key_concepts,
            topics=topics,
            semantic_embedding=embedding,
            **(analysis_metadata or {})
        )

        self.semantic_analysis = analysis
        self.processing_status = ProcessingStatus.COMPLETED
        self.updated_at = datetime.utcnow()

        return analysis.analysis_id

    def add_consciousness_indicators(self, indicators: List[str], meta_cognitive_elements: List[str] = None) -> None:
        """Add consciousness-related content indicators"""
        if not self.semantic_analysis:
            self.semantic_analysis = SemanticAnalysis()

        self.semantic_analysis.consciousness_indicators.extend(indicators)
        if meta_cognitive_elements:
            self.semantic_analysis.meta_cognitive_elements.extend(meta_cognitive_elements)

        self.updated_at = datetime.utcnow()

    def create_version(self, changes_description: str) -> str:
        """Create new version of document"""
        version_record = {
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat(),
            "changes": changes_description,
            "content_hash": self.content_hash
        }

        self.version_history.append(version_record)
        self.version += 1
        self.updated_at = datetime.utcnow()

        return str(self.version)

    def add_source_reference(self, reference_type: str, reference_data: Dict[str, Any]) -> None:
        """Add source reference to document"""
        reference = {
            "reference_id": str(uuid.uuid4()),
            "reference_type": reference_type,  # "url", "file", "citation", "api"
            "created_at": datetime.utcnow().isoformat(),
            "data": reference_data
        }

        self.source_references.append(reference)
        self.updated_at = datetime.utcnow()

    def add_citation(self, cited_work: str, citation_context: str, page_number: int = None) -> None:
        """Add citation from this document"""
        citation = {
            "citation_id": str(uuid.uuid4()),
            "cited_work": cited_work,
            "context": citation_context,
            "page_number": page_number,
            "created_at": datetime.utcnow().isoformat()
        }

        self.citations.append(citation)
        self.updated_at = datetime.utcnow()

    def associate_thoughtseed(self, trace_id: str, relevance_score: float = 1.0) -> None:
        """Associate document with thoughtseed trace"""
        if trace_id not in self.thoughtseed_references:
            self.thoughtseed_references.append(trace_id)

        # Add to concept activations
        self.concept_activations[f"thoughtseed_{trace_id}"] = relevance_score
        self.updated_at = datetime.utcnow()

    def calculate_learning_potential(self) -> float:
        """Calculate educational/learning potential of document"""
        if not self.semantic_analysis:
            return 0.0

        factors = []

        # Complexity factor (moderate complexity is best for learning)
        complexity = self.semantic_analysis.complexity_score
        complexity_factor = 1.0 - abs(complexity - 0.6)  # Peak at 0.6 complexity
        factors.append(complexity_factor)

        # Information density
        factors.append(self.semantic_analysis.information_density)

        # Coherence (higher is better for learning)
        factors.append(self.semantic_analysis.coherence_score)

        # Consciousness indicators boost
        consciousness_boost = min(len(self.semantic_analysis.consciousness_indicators) * 0.1, 0.3)

        # Calculate weighted average
        if factors:
            base_potential = sum(factors) / len(factors)
            self.semantic_analysis.learning_potential = min(base_potential + consciousness_boost, 1.0)
        else:
            self.semantic_analysis.learning_potential = 0.0

        return self.semantic_analysis.learning_potential

    def get_content_summary(self) -> Dict[str, Any]:
        """Get summary of document content and analysis"""
        summary = {
            "artifact_id": self.artifact_id,
            "title": self.title,
            "document_type": self.document_type.value,
            "processing_status": self.processing_status.value,
            "content_length": len(self.content) if self.content else 0,
            "tags": self.tags,
            "categories": self.categories,
            "version": self.version
        }

        if self.semantic_analysis:
            summary.update({
                "key_concepts": self.semantic_analysis.key_concepts[:5],  # Top 5
                "topics": self.semantic_analysis.topics[:3],  # Top 3
                "complexity_score": self.semantic_analysis.complexity_score,
                "learning_potential": self.semantic_analysis.learning_potential,
                "consciousness_indicators": len(self.semantic_analysis.consciousness_indicators)
            })

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()

    @classmethod
    def create_mock_document(cls, user_id: str, title: str, document_type: DocumentType,
                           journey_id: str = None, content: str = None) -> "DocumentArtifact":
        """
        Create mock document artifact for development/testing.
        Constitutional compliance: clearly marked as mock data.
        """
        mock_content = content or f"Mock {document_type.value} content for: {title}"

        artifact = cls(
            user_id=user_id,
            journey_id=journey_id,
            title=title,
            document_type=document_type,
            content=mock_content,
            content_hash=f"mock_hash_{hash(mock_content)}",
            mock_data_enabled=True,
            processing_status=ProcessingStatus.COMPLETED,
            tags=["mock", "development", document_type.value],
            categories=["development", "consciousness_research"],
            priority=3
        )

        # Add mock semantic analysis
        semantic_analysis = SemanticAnalysis(
            key_concepts=["consciousness", "learning", "development", "mock_data"],
            topics=["consciousness_research", "development"],
            complexity_score=0.6,
            coherence_score=0.7,
            information_density=0.5,
            consciousness_indicators=["self_reference", "meta_cognition"],
            meta_cognitive_elements=["reflection", "awareness"],
            learning_potential=0.65
        )

        artifact.semantic_analysis = semantic_analysis

        return artifact


# Type aliases for convenience
DocumentArtifactDict = Dict[str, Any]
DocumentArtifactList = List[DocumentArtifact]