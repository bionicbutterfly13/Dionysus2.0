"""
Document Ingestion API Routes - T029 (Updated with Daedalus Gateway per Spec 021)
Flux Self-Evolving Consciousness Emulator

Handles document upload through Daedalus perceptual information gateway.
Constitutional compliance: mock data transparency, evaluation framework.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import List, Optional, Dict, Any
import logging
import os
import json
from pathlib import Path
from datetime import datetime
import io

# Daedalus Gateway Integration (Spec 021)
from services.daedalus import Daedalus

logger = logging.getLogger(__name__)

router = APIRouter()

# Simple in-memory storage for uploaded documents (for demo purposes)
uploaded_documents = []
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load existing documents from disk on startup
def load_existing_documents():
    """Load documents from disk into memory on startup."""
    global uploaded_documents
    try:
        for file_path in UPLOAD_DIR.glob("doc_*"):
            # Parse filename: doc_{hash}_{filename}
            parts = file_path.name.split("_", 2)
            if len(parts) >= 3:
                doc_id = f"doc_{parts[1]}"
                filename = parts[2]

                # Check if already loaded
                if any(d.get("document_id") == doc_id for d in uploaded_documents):
                    continue

                # Add basic metadata
                uploaded_documents.append({
                    "document_id": doc_id,
                    "filename": filename,
                    "size": file_path.stat().st_size,
                    "content_type": "application/pdf" if filename.endswith(".pdf") else "text/plain",
                    "uploaded_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "file_path": str(file_path),
                    "extraction": {"concepts": []},  # Will be populated on re-processing
                    "status": "completed"
                })
        logger.info(f"Loaded {len(uploaded_documents)} existing documents from disk")
    except Exception as e:
        logger.error(f"Failed to load existing documents: {str(e)}")

# Load documents on module import
load_existing_documents()

# Initialize Daedalus gateway
daedalus = Daedalus()

@router.get("/documents")
async def list_documents(topic: Optional[str] = None):
    """
    List all uploaded documents with metadata.
    Returns documents sorted by upload time (newest first).

    Args:
        topic: Optional topic/concept to filter by
    """
    try:
        docs_to_return = uploaded_documents

        # Filter by topic if provided
        if topic:
            docs_to_return = [
                doc for doc in uploaded_documents
                if topic.lower() in [
                    c.lower() for c in doc.get("extraction", {}).get("concepts", [])
                ]
            ]

        # Sort by upload time
        sorted_docs = sorted(
            docs_to_return,
            key=lambda d: d.get('uploaded_at', ''),
            reverse=True
        )

        return {
            "documents": [
                {
                    "id": doc.get("document_id"),
                    "title": doc.get("filename"),
                    "type": "file" if doc.get("content_type", "").startswith("application") or doc.get("content_type", "").startswith("text") else "web",
                    "uploaded_at": doc.get("uploaded_at"),
                    "extraction": doc.get("extraction", {}),
                    "quality": doc.get("quality", {}),
                    "consciousness": doc.get("consciousness", {}),
                }
                for doc in sorted_docs
            ],
            "total": len(sorted_docs),
            "filter": {"topic": topic} if topic else None
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        return {"documents": [], "total": 0}

@router.post("/documents")
async def ingest_documents(
    files: List[UploadFile] = File(...),
    tags: Optional[str] = Form(None)
):
    """
    Ingest documents into Flux consciousness system via Daedalus Gateway.

    Updated per Spec 021: Uses simplified Daedalus as perceptual information gateway.
    Daedalus receives uploads and creates LangGraph agents for processing.
    """
    try:
        results = []
        for file in files:
            # Read file content
            content = await file.read()

            # Basic validation
            if len(content) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is empty"
                )

            # Convert to file-like object for Daedalus
            file_obj = io.BytesIO(content)
            file_obj.name = file.filename

            # Pass through Daedalus gateway with LangGraph workflow
            # Updated 2025-10-01: Now uses full consciousness processing pipeline
            tag_list = tags.split(",") if tags else []
            daedalus_response = daedalus.receive_perceptual_information(
                data=file_obj,
                tags=tag_list,
                max_iterations=3,
                quality_threshold=0.7
            )

            # Handle Daedalus response
            if daedalus_response.get('status') == 'error':
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Daedalus gateway error: {daedalus_response.get('error_message')}"
                )

            # Save file to disk
            document_id = f"doc_{hash(file.filename + str(file.size))}"[:16]
            file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"

            with open(file_path, "wb") as buffer:
                buffer.write(content)

            # Store document metadata with full consciousness processing results
            result = {
                # Basic metadata
                "filename": file.filename,
                "size": file.size,
                "content_type": file.content_type,
                "status": "completed",
                "document_id": document_id,
                "mockData": False,
                "tags": tag_list,
                "uploaded_at": datetime.now().isoformat(),
                "file_path": str(file_path),

                # Consciousness processing results (NEW)
                "extraction": daedalus_response.get('extraction', {}),
                "consciousness": daedalus_response.get('consciousness', {}),
                "research": daedalus_response.get('research', {}),
                "quality": daedalus_response.get('quality', {}),
                "meta_cognitive": daedalus_response.get('meta_cognitive', {}),
                "workflow": daedalus_response.get('workflow', {}),

                # Gateway info
                "daedalus_status": daedalus_response.get('status'),
                "daedalus_source": daedalus_response.get('source')
            }
            results.append(result)
            uploaded_documents.append(result)

            logger.info(
                f"Document processed via Daedalus LangGraph: {file.filename} ({file.size} bytes) "
                f"| Concepts: {len(daedalus_response.get('extraction', {}).get('concepts', []))} "
                f"| Basins: {daedalus_response.get('consciousness', {}).get('basins_created', 0)} "
                f"| Quality: {daedalus_response.get('quality', {}).get('scores', {}).get('overall', 0):.2f}"
            )

        return {
            "message": f"Successfully ingested {len(files)} documents via Daedalus gateway",
            "documents": results,
            "gateway_info": {
                "gateway_used": "daedalus",
                "spec_version": "021-remove-all-that",
                "gateway_responsibility": "perceptual_information_reception"
            },
            "constitutional_compliance": {
                "mock_data_transparency": False,
                "evaluation_framework": "active"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )

@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get full document details by ID."""
    try:
        # Find document in memory
        doc = next((d for d in uploaded_documents if d.get("document_id") == document_id), None)

        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Return full document details
        return {
            "id": doc.get("document_id"),
            "title": doc.get("filename"),
            "type": "file" if doc.get("content_type", "").startswith("application") or doc.get("content_type", "").startswith("text") else "web",
            "uploaded_at": doc.get("uploaded_at"),
            "size": doc.get("size"),
            "content_type": doc.get("content_type"),
            "extraction": doc.get("extraction", {}),
            "quality": doc.get("quality", {}),
            "research": doc.get("research", {}),
            "consciousness": doc.get("consciousness", {}),
            "meta_cognitive": doc.get("meta_cognitive", {}),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}"
        )

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete document by ID, removing from memory and disk."""
    global uploaded_documents
    try:
        # Find document in memory
        doc = next((d for d in uploaded_documents if d.get("document_id") == document_id), None)

        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Delete file from disk
        file_path = doc.get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file from disk: {file_path}")

        # Remove from memory
        uploaded_documents = [d for d in uploaded_documents if d.get("document_id") != document_id]

        logger.info(f"Document deleted: {doc.get('filename')} ({document_id})")

        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully",
            "deleted": {
                "id": document_id,
                "title": doc.get("filename")
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )