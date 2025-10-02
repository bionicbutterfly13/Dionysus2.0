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
from src.services.daedalus import Daedalus

logger = logging.getLogger(__name__)

router = APIRouter()

# Simple in-memory storage for uploaded documents (for demo purposes)
uploaded_documents = []
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize Daedalus gateway
daedalus = Daedalus()

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
    """Get document by ID."""
    return {
        "document_id": document_id,
        "status": "placeholder"
    }

@router.get("/documents")
async def list_documents():
    """List all documents."""
    return {
        "documents": uploaded_documents,
        "total": len(uploaded_documents),
        "status": "active"
    }