"""
Document Models
Constitutional compliance: mock data transparency, evaluative feedback
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentArtifact(BaseModel):
    """Document artifact with constitutional compliance"""
    
    # Core fields
    id: str = Field(..., description="Unique document identifier")
    user_id: str = Field(..., description="User who uploaded the document")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    content_hash: str = Field(..., description="SHA-256 hash of content")
    
    # Processing status
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Processing status")
    batch_id: Optional[str] = Field(default=None, description="Batch processing identifier")
    
    # Metadata
    source_type: str = Field(default="file", description="Source type: file, url, text")
    mime_type: Optional[str] = Field(default=None, description="MIME type")
    language: Optional[str] = Field(default=None, description="Detected language")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    evaluation_frame_id: Optional[str] = Field(default=None, description="Associated evaluation frame")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)
    processed_at: Optional[datetime] = Field(default=None)
    
    # Processing results
    concepts_extracted: List[str] = Field(default_factory=list, description="Extracted concepts")
    thoughtseed_traces: List[str] = Field(default_factory=list, description="Associated ThoughtSeed IDs")
    knowledge_graph_nodes: List[str] = Field(default_factory=list, description="Created KG node IDs")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DocumentUploadRequest(BaseModel):
    """Document upload request with constitutional compliance"""
    
    source_type: str = Field(default="file", description="Source type: file, url, text")
    payload: Optional[str] = Field(default=None, description="Content for text/url sources")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    
    class Config:
        schema_extra = {
            "example": {
                "source_type": "file",
                "metadata": {
                    "title": "Research Paper",
                    "author": "Dr. Smith",
                    "tags": ["AI", "consciousness"]
                },
                "mock_data": True
            }
        }

class DocumentUploadResponse(BaseModel):
    """Document upload response with constitutional compliance"""
    
    batch_id: str = Field(..., description="Batch processing identifier")
    document_id: str = Field(..., description="Document identifier")
    status: DocumentStatus = Field(..., description="Current processing status")
    
    # Constitutional compliance
    mock_data: bool = Field(..., description="Constitutional requirement: mock data flag")
    evaluation_frame_id: str = Field(..., description="Associated evaluation frame")
    
    # Processing info
    estimated_processing_time: Optional[int] = Field(default=None, description="Estimated seconds")
    concepts_preview: List[str] = Field(default_factory=list, description="Preview of extracted concepts")
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_123",
                "document_id": "doc_456",
                "status": "processing",
                "mock_data": True,
                "evaluation_frame_id": "eval_789",
                "estimated_processing_time": 30,
                "concepts_preview": ["active inference", "consciousness"]
            }
        }
