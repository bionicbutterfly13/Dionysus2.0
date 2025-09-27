#!/usr/bin/env python3
"""
Document Model: Document upload and processing management
"""

from typing import Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class DocumentStatus(str, Enum):
    """Document processing status states"""
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class Document(BaseModel):
    """
    Document: Represents uploaded research documents for processing

    Purpose: Manages document lifecycle from upload through processing completion
    Storage: Primary in Neo4j, metadata cached in Redis
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document identifier")
    filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    file_size: int = Field(..., gt=0, le=500_000_000, description="File size in bytes (max 500MB)")

    # Processing relationship
    batch_id: str = Field(..., description="Reference to ProcessingBatch")

    # Content and extraction
    extracted_text: Optional[str] = Field(None, description="Extracted text content")
    text_extraction_method: Optional[str] = Field(None, description="Method used for text extraction")

    # Status tracking
    processing_status: DocumentStatus = Field(default=DocumentStatus.UPLOADED, description="Current processing status")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Document upload timestamp")
    validation_timestamp: Optional[datetime] = Field(None, description="Validation completion timestamp")
    processing_started_timestamp: Optional[datetime] = Field(None, description="Processing start timestamp")
    processing_completed_timestamp: Optional[datetime] = Field(None, description="Processing completion timestamp")

    # File validation
    file_hash: Optional[str] = Field(None, description="SHA-256 hash of file content")
    virus_scan_status: Optional[str] = Field(None, description="Virus scan result")

    # Processing metadata
    thoughtseed_processing_enabled: bool = Field(default=True, description="Enable ThoughtSeed processing")
    attractor_modification_enabled: bool = Field(default=True, description="Enable attractor basin modification")
    neural_field_evolution_enabled: bool = Field(default=True, description="Enable neural field evolution")
    memory_integration_enabled: bool = Field(default=True, description="Enable memory formation")

    # Processing results tracking
    thoughtseed_ids: list[str] = Field(default_factory=list, description="Generated ThoughtSeed IDs")
    attractor_basin_ids: list[str] = Field(default_factory=list, description="Modified attractor basin IDs")
    neural_field_ids: list[str] = Field(default_factory=list, description="Evolved neural field IDs")
    memory_formation_ids: list[str] = Field(default_factory=list, description="Created memory formation IDs")

    # Error tracking
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")
    retry_count: int = Field(default=0, ge=0, description="Number of processing retries")
    max_retries: int = Field(default=3, ge=0, description="Maximum allowed retries")

    # TTL and cleanup
    expires_at: Optional[datetime] = Field(None, description="Document expiration timestamp")
    cleanup_scheduled: bool = Field(default=False, description="Whether cleanup is scheduled")

    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate supported content types"""
        supported_types = {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/markdown"
        }
        if v not in supported_types:
            raise ValueError(f"Unsupported content type: {v}. Supported types: {supported_types}")
        return v

    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename doesn't contain invalid characters"""
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Filename contains invalid characters: {invalid_chars}")
        return v

    @validator('file_size')
    def validate_file_size(cls, v):
        """Validate file size limits"""
        if v <= 0:
            raise ValueError("File size must be greater than 0")
        if v > 500_000_000:  # 500MB
            raise ValueError("File size exceeds 500MB limit")
        return v

    def transition_status(self, new_status: DocumentStatus, error_message: Optional[str] = None) -> bool:
        """
        Transition document to new status with validation

        Args:
            new_status: Target status
            error_message: Optional error message for FAILED status

        Returns:
            bool: True if transition was valid and completed
        """
        valid_transitions = {
            DocumentStatus.UPLOADED: [DocumentStatus.VALIDATING, DocumentStatus.FAILED],
            DocumentStatus.VALIDATING: [DocumentStatus.QUEUED, DocumentStatus.FAILED],
            DocumentStatus.QUEUED: [DocumentStatus.PROCESSING, DocumentStatus.FAILED, DocumentStatus.EXPIRED],
            DocumentStatus.PROCESSING: [DocumentStatus.COMPLETED, DocumentStatus.FAILED],
            DocumentStatus.COMPLETED: [DocumentStatus.EXPIRED],
            DocumentStatus.FAILED: [DocumentStatus.QUEUED],  # Allow retry
            DocumentStatus.EXPIRED: []  # Terminal state
        }

        current_status = self.processing_status
        if new_status not in valid_transitions.get(current_status, []):
            return False

        # Update status and timestamps
        self.processing_status = new_status
        now = datetime.utcnow()

        if new_status == DocumentStatus.VALIDATING:
            pass  # No specific timestamp
        elif new_status == DocumentStatus.QUEUED:
            self.validation_timestamp = now
        elif new_status == DocumentStatus.PROCESSING:
            self.processing_started_timestamp = now
        elif new_status == DocumentStatus.COMPLETED:
            self.processing_completed_timestamp = now
        elif new_status == DocumentStatus.FAILED:
            if error_message:
                self.error_message = error_message
            self.retry_count += 1

        return True

    def can_retry(self) -> bool:
        """Check if document can be retried after failure"""
        return (
            self.processing_status == DocumentStatus.FAILED and
            self.retry_count < self.max_retries
        )

    def get_processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds"""
        if self.processing_started_timestamp and self.processing_completed_timestamp:
            delta = self.processing_completed_timestamp - self.processing_started_timestamp
            return delta.total_seconds()
        return None

    def is_expired(self) -> bool:
        """Check if document has expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    def add_thoughtseed_result(self, thoughtseed_id: str) -> None:
        """Add a generated ThoughtSeed ID to results"""
        if thoughtseed_id not in self.thoughtseed_ids:
            self.thoughtseed_ids.append(thoughtseed_id)

    def add_attractor_result(self, attractor_basin_id: str) -> None:
        """Add a modified attractor basin ID to results"""
        if attractor_basin_id not in self.attractor_basin_ids:
            self.attractor_basin_ids.append(attractor_basin_id)

    def add_neural_field_result(self, neural_field_id: str) -> None:
        """Add an evolved neural field ID to results"""
        if neural_field_id not in self.neural_field_ids:
            self.neural_field_ids.append(neural_field_id)

    def add_memory_result(self, memory_formation_id: str) -> None:
        """Add a created memory formation ID to results"""
        if memory_formation_id not in self.memory_formation_ids:
            self.memory_formation_ids.append(memory_formation_id)

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True
        extra = "forbid"