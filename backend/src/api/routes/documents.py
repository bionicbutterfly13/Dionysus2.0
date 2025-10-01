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

            # Pass through Daedalus gateway (Spec 021: Single responsibility - receive perceptual information)
            daedalus_response = daedalus.receive_perceptual_information(file_obj)

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

            # Store document metadata with Daedalus integration info
            result = {
                "filename": file.filename,
                "size": file.size,
                "content_type": file.content_type,
                "status": "completed",
                "document_id": document_id,
                "mockData": False,
                "tags": tags.split(",") if tags else [],
                "uploaded_at": datetime.now().isoformat(),
                "file_path": str(file_path),
                "daedalus_reception": daedalus_response.get('status'),
                "agents_created": daedalus_response.get('agents_created', [])
            }
            results.append(result)
            uploaded_documents.append(result)

            logger.info(
                f"Document processed via Daedalus: {file.filename} ({file.size} bytes) "
                f"| Agents: {len(daedalus_response.get('agents_created', []))}"
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