#!/usr/bin/env python3
"""
DocumentSource Model: Document processing and ThoughtSeed integration
"""

from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid
from pathlib import Path


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    DOCX = "docx"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ExtractionQuality(BaseModel):
    """Document extraction quality metrics"""
    overall_quality: float = Field(..., ge=0.0, le=1.0, description="Overall extraction quality")
    text_clarity: float = Field(..., ge=0.0, le=1.0, description="Text clarity score")
    structure_preservation: float = Field(..., ge=0.0, le=1.0, description="Structure preservation score")
    metadata_completeness: float = Field(..., ge=0.0, le=1.0, description="Metadata completeness")
    
    # Quality breakdown by section
    section_qualities: Dict[str, float] = Field(default_factory=dict, description="Quality by document section")
    
    # Issue tracking
    extraction_issues: List[str] = Field(default_factory=list, description="Extraction issues encountered")
    confidence_intervals: Dict[str, List[float]] = Field(default_factory=dict, description="Confidence intervals")


class NarrativeElements(BaseModel):
    """Extracted narrative elements from document"""
    themes: List[str] = Field(default_factory=list, description="Identified themes")
    motifs: List[str] = Field(default_factory=list, description="Recurring motifs")
    story_structures: List[Dict[str, Any]] = Field(default_factory=list, description="Story structure elements")
    
    # Character and entity analysis
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Named entities")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Entity relationships")
    
    # Narrative flow
    narrative_flow: List[Dict[str, Any]] = Field(default_factory=list, description="Narrative progression")
    emotional_arc: List[Dict[str, float]] = Field(default_factory=list, description="Emotional progression")
    
    # Meta-narrative analysis
    narrative_techniques: List[str] = Field(default_factory=list, description="Narrative techniques used")
    perspective_shifts: List[Dict[str, Any]] = Field(default_factory=list, description="Perspective changes")


class DocumentMetadata(BaseModel):
    """Document metadata and properties"""
    # File properties
    filename: str = Field(..., min_length=1, description="Original filename")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    file_hash: Optional[str] = Field(None, description="File content hash (SHA-256)")
    
    # Document properties
    page_count: Optional[int] = Field(None, ge=0, description="Number of pages (if applicable)")
    word_count: Optional[int] = Field(None, ge=0, description="Estimated word count")
    character_count: Optional[int] = Field(None, ge=0, description="Character count")
    language: Optional[str] = Field(None, description="Detected language")
    
    # Creation and modification
    creation_date: Optional[datetime] = Field(None, description="Document creation date")
    modification_date: Optional[datetime] = Field(None, description="Last modification date")
    author: Optional[str] = Field(None, description="Document author")
    title: Optional[str] = Field(None, description="Document title")
    
    # Processing metadata
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    processing_timestamp: Optional[datetime] = Field(None, description="Processing start timestamp")
    
    # Additional properties
    custom_properties: Dict[str, Any] = Field(default_factory=dict, description="Custom document properties")
    tags: Set[str] = Field(default_factory=set, description="User-assigned tags")


class ContentSection(BaseModel):
    """Document content section"""
    section_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Section identifier")
    section_type: str = Field(..., description="Section type (header, paragraph, list, etc.)")
    title: Optional[str] = Field(None, description="Section title")
    content: str = Field(..., description="Section content")
    
    # Hierarchical structure
    level: int = Field(default=0, ge=0, description="Hierarchical level")
    parent_section: Optional[str] = Field(None, description="Parent section ID")
    child_sections: List[str] = Field(default_factory=list, description="Child section IDs")
    
    # Position and layout
    start_position: Optional[int] = Field(None, description="Start position in document")
    end_position: Optional[int] = Field(None, description="End position in document")
    page_number: Optional[int] = Field(None, description="Page number (if applicable)")
    
    # Processing results
    extracted_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Entities in section")
    semantic_summary: Optional[str] = Field(None, description="Semantic summary of section")
    complexity_score: float = Field(default=0.0, ge=0.0, description="Content complexity score")


class DocumentSource(BaseModel):
    """
    DocumentSource: Represents a document processed through ThoughtSeed hierarchy
    
    Handles document ingestion, processing, and integration with:
    - ThoughtSeed 5-layer processing
    - Pattern extraction
    - Narrative element identification
    - Context Engineering integration
    """
    
    # Core identification
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document identifier")
    document_name: Optional[str] = Field(None, description="Human-readable document name")
    description: Optional[str] = Field(None, description="Document description")
    
    # Document type and metadata
    document_type: DocumentType = Field(..., description="Document type classification")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    
    # Processing state
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Processing status")
    extraction_quality: ExtractionQuality = Field(..., description="Extraction quality metrics")
    processing_time_ms: int = Field(default=0, ge=0, description="Total processing time")
    
    # Content structure
    raw_content: Optional[str] = Field(None, description="Raw extracted text content")
    structured_content: List[ContentSection] = Field(default_factory=list, description="Structured content sections")
    content_summary: Optional[str] = Field(None, description="Content summary")
    
    # ThoughtSeed integration
    thoughtseed_workspace_id: Optional[str] = Field(None, description="Associated ThoughtSeed workspace")
    thoughtseed_layers_processed: List[str] = Field(default_factory=list, description="Processed ThoughtSeed layers")
    thoughtseed_traces: List[str] = Field(default_factory=list, description="Generated ThoughtSeed trace IDs")
    
    # Pattern extraction
    patterns_extracted: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted patterns")
    pattern_confidence: Dict[str, float] = Field(default_factory=dict, description="Pattern confidence scores")
    pattern_relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Pattern relationships")
    
    # Narrative analysis
    narrative_elements: NarrativeElements = Field(default_factory=NarrativeElements, description="Narrative elements")
    narrative_complexity: float = Field(default=0.0, ge=0.0, description="Narrative complexity score")
    
    # Consciousness and intelligence metrics
    consciousness_indicators: List[str] = Field(default_factory=list, description="Consciousness indicators found")
    intelligence_patterns: List[str] = Field(default_factory=list, description="Intelligence pattern IDs")
    emergence_signals: Dict[str, float] = Field(default_factory=dict, description="Emergence signal strengths")
    
    # Context Engineering integration
    attractor_basins_activated: List[str] = Field(default_factory=list, description="Activated basin IDs")
    neural_field_influences: Dict[str, float] = Field(default_factory=dict, description="Neural field influences")
    
    # Semantic and vector representations
    semantic_embeddings: Optional[List[float]] = Field(None, description="Document semantic embeddings")
    section_embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="Section embeddings")
    
    # Database integration
    vector_database_id: Optional[str] = Field(None, description="Qdrant vector database ID")
    graph_database_id: Optional[str] = Field(None, description="Neo4j graph database ID")
    knowledge_graph_nodes: List[str] = Field(default_factory=list, description="Knowledge graph node IDs")
    
    # Processing configuration
    processing_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "extract_narratives": True,
            "enable_consciousness_detection": True,
            "thoughtseed_layers": ["sensory", "perceptual", "conceptual"],
            "pattern_extraction_threshold": 0.7,
            "narrative_analysis_depth": "standard"
        },
        description="Processing configuration"
    )
    
    # Temporal tracking
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Document creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
    
    # Access and usage
    access_count: int = Field(default=0, ge=0, description="Number of times accessed")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    
    # Relationships
    related_documents: List[str] = Field(default_factory=list, description="Related document IDs")
    source_research_queries: List[str] = Field(default_factory=list, description="Source research query IDs")
    generated_insights: List[str] = Field(default_factory=list, description="Generated insight IDs")
    
    # Quality and validation
    validation_status: str = Field(default="unvalidated", description="Validation status")
    quality_scores: Dict[str, float] = Field(default_factory=dict, description="Various quality scores")
    review_comments: List[Dict[str, Any]] = Field(default_factory=list, description="Review comments")
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate document metadata"""
        if v.file_size_bytes < 0:
            raise ValueError("File size cannot be negative")
        return v
    
    @validator('structured_content')
    def validate_content_hierarchy(cls, v):
        """Validate content section hierarchy"""
        section_ids = {section.section_id for section in v}
        
        for section in v:
            # Check parent references
            if section.parent_section and section.parent_section not in section_ids:
                raise ValueError(f"Parent section {section.parent_section} not found")
            
            # Check child references
            for child_id in section.child_sections:
                if child_id not in section_ids:
                    raise ValueError(f"Child section {child_id} not found")
        
        return v
    
    @validator('semantic_embeddings')
    def validate_semantic_embeddings(cls, v):
        """Validate semantic embeddings"""
        if v is not None:
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Semantic embeddings must be list of numbers")
            if len(v) == 0:
                raise ValueError("Semantic embeddings cannot be empty")
        return v
    
    def add_content_section(self, section_type: str, content: str, 
                           title: Optional[str] = None, level: int = 0) -> ContentSection:
        """Add a new content section"""
        section = ContentSection(
            section_type=section_type,
            content=content,
            title=title,
            level=level
        )
        
        self.structured_content.append(section)
        self.updated_at = datetime.utcnow()
        
        return section
    
    def update_processing_status(self, status: ProcessingStatus, 
                               processing_time_ms: Optional[int] = None) -> None:
        """Update document processing status"""
        self.processing_status = status
        self.updated_at = datetime.utcnow()
        
        if processing_time_ms is not None:
            self.processing_time_ms = processing_time_ms
        
        if status == ProcessingStatus.COMPLETED:
            self.processed_at = datetime.utcnow()
    
    def add_pattern(self, pattern_data: Dict[str, Any], confidence: float) -> None:
        """Add extracted pattern"""
        pattern_id = pattern_data.get("pattern_id", str(uuid.uuid4()))
        
        self.patterns_extracted.append(pattern_data)
        self.pattern_confidence[pattern_id] = confidence
        self.updated_at = datetime.utcnow()
    
    def calculate_overall_quality(self) -> float:
        """Calculate overall document quality score"""
        extraction_quality = self.extraction_quality.overall_quality
        
        # Factor in processing success
        processing_factor = 1.0 if self.processing_status == ProcessingStatus.COMPLETED else 0.5
        
        # Factor in content richness
        content_factor = min(len(self.structured_content) / 10, 1.0)
        
        # Factor in pattern extraction success
        pattern_factor = min(len(self.patterns_extracted) / 5, 1.0)
        
        overall_quality = (
            extraction_quality * 0.4 +
            processing_factor * 0.3 +
            content_factor * 0.2 +
            pattern_factor * 0.1
        )
        
        return min(overall_quality, 1.0)
    
    def get_sections_by_type(self, section_type: str) -> List[ContentSection]:
        """Get content sections by type"""
        return [section for section in self.structured_content 
                if section.section_type == section_type]
    
    def get_section_hierarchy(self) -> Dict[str, List[str]]:
        """Get section hierarchy mapping"""
        hierarchy = {}
        
        for section in self.structured_content:
            parent_id = section.parent_section or "root"
            if parent_id not in hierarchy:
                hierarchy[parent_id] = []
            hierarchy[parent_id].append(section.section_id)
        
        return hierarchy
    
    def increment_access_count(self) -> None:
        """Increment access count and update last accessed time"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def is_processing_complete(self) -> bool:
        """Check if document processing is complete"""
        return self.processing_status == ProcessingStatus.COMPLETED
    
    def get_content_preview(self, max_length: int = 500) -> str:
        """Get content preview"""
        if self.content_summary:
            return self.content_summary[:max_length]
        elif self.raw_content:
            return self.raw_content[:max_length] + "..." if len(self.raw_content) > max_length else self.raw_content
        elif self.structured_content:
            content_parts = [section.content for section in self.structured_content[:3]]
            combined = " ".join(content_parts)
            return combined[:max_length] + "..." if len(combined) > max_length else combined
        else:
            return f"Document: {self.metadata.filename}"
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            set: lambda v: list(v)
        }
        validate_assignment = True
        extra = "forbid"