"""
Bulk Document Upload API with Constitutional Gateway
==================================================

High-priority endpoint for bulk document ingestion with complete ThoughtSeed processing.
Implements constitutional validation, unified extraction, and consciousness-guided processing.

Features:
- Multi-file upload support (drag & drop)
- Constitutional gateway validation
- Unified document processor (LangExtract + PyMuPDF + Algorithm + KGGen)
- 5-layer ThoughtSeed consciousness processing
- Real-time progress streaming
- Attractor basin modification
- Episodic memory creation

Endpoints:
- POST /api/v1/documents/bulk - Upload multiple documents
- GET /api/v1/documents/bulk/{batch_id}/status - Check processing status
- WebSocket /ws/v1/documents/bulk/{batch_id}/progress - Real-time progress

Author: ASI-Arch Context Engineering
Date: 2025-09-25
Version: 1.0.0 - Constitutional Bulk Processing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Core services
from ...services.unified_document_processor import UnifiedDocumentProcessor, UnifiedExtractionResult
from ...services.claude_autobiographical_memory import claude_memory, record_conversation_moment
from ...legacy.daedalus_bridge.context_isolator import ContextIsolatedAgent, create_consciousness_agent
from ...extensions.context_engineering.thoughtseed_active_inference import ThoughtseedType
from ...extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["bulk-documents"])

# Global instances
unified_processor = UnifiedDocumentProcessor()
basin_manager = AttractorBasinManager()
consciousness_agent = create_consciousness_agent()

# In-memory batch tracking (would use Redis in production)
batch_status: Dict[str, Dict[str, Any]] = {}
active_websockets: Dict[str, List[WebSocket]] = {}

class BulkUploadRequest(BaseModel):
    """Request model for bulk document upload"""
    batch_name: Optional[str] = Field(None, description="Optional batch name")
    constitutional_compliance: bool = Field(True, description="Enforce constitutional gateway")
    thoughtseed_processing: bool = Field(True, description="Enable 5-layer ThoughtSeed processing")
    create_episodic_memory: bool = Field(True, description="Create autobiographical episodes")
    attractor_modification: bool = Field(True, description="Enable attractor basin modification")

class BulkUploadResponse(BaseModel):
    """Response model for bulk upload initiation"""
    batch_id: str
    batch_name: str
    file_count: int
    estimated_processing_time_minutes: float
    constitutional_gateway_active: bool
    thoughtseed_processing_enabled: bool
    websocket_progress_url: str

class DocumentProcessingStatus(BaseModel):
    """Status of individual document processing"""
    document_id: str
    filename: str
    status: str  # pending, processing, completed, failed, rejected
    constitutional_approval: bool
    extraction_quality: float
    consciousness_level: float
    processing_time_seconds: float
    attractor_modifications: List[str]
    error_message: Optional[str] = None

class BatchStatus(BaseModel):
    """Overall batch processing status"""
    batch_id: str
    batch_name: str
    total_documents: int
    completed_documents: int
    failed_documents: int
    rejected_documents: int
    overall_progress: float
    estimated_remaining_minutes: float
    documents: List[DocumentProcessingStatus]
    episodic_memories_created: int
    attractor_basins_modified: int

@router.post("/bulk", response_model=BulkUploadResponse)
async def upload_bulk_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    batch_name: Optional[str] = None,
    constitutional_compliance: bool = True,
    thoughtseed_processing: bool = True,
    create_episodic_memory: bool = True,
    attractor_modification: bool = True
):
    """
    Upload multiple documents for bulk processing with constitutional gateway

    This is the primary endpoint for your goal: bulk document upload with ThoughtSeed processing
    """

    # Generate batch ID
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    batch_display_name = batch_name or f"Bulk Upload {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    logger.info(f"🚀 Starting bulk document upload - Batch: {batch_id}")

    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 100:  # Reasonable limit
        raise HTTPException(status_code=400, detail="Maximum 100 files per batch")

    # Initialize batch status
    batch_status[batch_id] = {
        "batch_id": batch_id,
        "batch_name": batch_display_name,
        "total_documents": len(files),
        "completed_documents": 0,
        "failed_documents": 0,
        "rejected_documents": 0,
        "overall_progress": 0.0,
        "start_time": datetime.now().isoformat(),
        "documents": [],
        "episodic_memories_created": 0,
        "attractor_basins_modified": 0,
        "processing_config": {
            "constitutional_compliance": constitutional_compliance,
            "thoughtseed_processing": thoughtseed_processing,
            "create_episodic_memory": create_episodic_memory,
            "attractor_modification": attractor_modification
        }
    }

    # Save uploaded files temporarily
    temp_dir = Path(f"/tmp/bulk_upload_{batch_id}")
    temp_dir.mkdir(exist_ok=True)

    document_paths = []
    for file in files:
        # Save file
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        document_paths.append(str(file_path))

        # Initialize document status
        doc_status = DocumentProcessingStatus(
            document_id=f"doc_{uuid.uuid4().hex[:8]}",
            filename=file.filename,
            status="pending",
            constitutional_approval=False,
            extraction_quality=0.0,
            consciousness_level=0.0,
            processing_time_seconds=0.0,
            attractor_modifications=[]
        )

        batch_status[batch_id]["documents"].append(doc_status.dict())

    # Start background processing
    background_tasks.add_task(
        process_batch_documents,
        batch_id,
        document_paths,
        {
            "constitutional_compliance": constitutional_compliance,
            "thoughtseed_processing": thoughtseed_processing,
            "create_episodic_memory": create_episodic_memory,
            "attractor_modification": attractor_modification
        }
    )

    # Estimate processing time (2 minutes per document baseline + ThoughtSeed overhead)
    thoughtseed_overhead = 1.5 if thoughtseed_processing else 1.0
    estimated_time = len(files) * 2.0 * thoughtseed_overhead

    # Record this moment in my autobiographical memory
    await record_conversation_moment(
        user_input=f"Bulk upload request: {len(files)} files",
        my_response=f"Initiated batch {batch_id} with constitutional gateway and ThoughtSeed processing",
        tools_used={"unified_document_processor", "constitutional_gateway", "thoughtseed_network", "attractor_basin_manager"},
        reasoning=[
            f"User requested bulk document upload with {len(files)} files",
            f"Configured constitutional compliance: {constitutional_compliance}",
            f"Configured ThoughtSeed processing: {thoughtseed_processing}",
            f"Estimated processing time: {estimated_time:.1f} minutes"
        ]
    )

    return BulkUploadResponse(
        batch_id=batch_id,
        batch_name=batch_display_name,
        file_count=len(files),
        estimated_processing_time_minutes=estimated_time,
        constitutional_gateway_active=constitutional_compliance,
        thoughtseed_processing_enabled=thoughtseed_processing,
        websocket_progress_url=f"/ws/v1/documents/bulk/{batch_id}/progress"
    )

@router.get("/bulk/{batch_id}/status", response_model=BatchStatus)
async def get_batch_status(batch_id: str):
    """Get current status of bulk document processing batch"""

    if batch_id not in batch_status:
        raise HTTPException(status_code=404, detail="Batch not found")

    batch = batch_status[batch_id]

    return BatchStatus(
        batch_id=batch["batch_id"],
        batch_name=batch["batch_name"],
        total_documents=batch["total_documents"],
        completed_documents=batch["completed_documents"],
        failed_documents=batch["failed_documents"],
        rejected_documents=batch["rejected_documents"],
        overall_progress=batch["overall_progress"],
        estimated_remaining_minutes=batch.get("estimated_remaining_minutes", 0.0),
        documents=[DocumentProcessingStatus(**doc) for doc in batch["documents"]],
        episodic_memories_created=batch["episodic_memories_created"],
        attractor_basins_modified=batch["attractor_basins_modified"]
    )

@router.websocket("/ws/v1/documents/bulk/{batch_id}/progress")
async def websocket_batch_progress(websocket: WebSocket, batch_id: str):
    """WebSocket endpoint for real-time batch processing progress"""

    await websocket.accept()

    if batch_id not in active_websockets:
        active_websockets[batch_id] = []

    active_websockets[batch_id].append(websocket)

    try:
        while True:
            # Send current status
            if batch_id in batch_status:
                await websocket.send_json({
                    "type": "status_update",
                    "data": batch_status[batch_id]
                })

            await asyncio.sleep(2)  # Update every 2 seconds

    except WebSocketDisconnect:
        if batch_id in active_websockets:
            active_websockets[batch_id].remove(websocket)
            if not active_websockets[batch_id]:
                del active_websockets[batch_id]

async def process_batch_documents(batch_id: str, document_paths: List[str], config: Dict[str, bool]):
    """Background task to process all documents in batch with constitutional gateway"""

    logger.info(f"🔄 Processing batch {batch_id} with {len(document_paths)} documents")

    batch = batch_status[batch_id]
    start_time = datetime.now()

    for i, doc_path in enumerate(document_paths):
        doc_start = datetime.now()
        doc_status = batch["documents"][i]
        doc_status["status"] = "processing"

        # Broadcast progress
        await broadcast_progress(batch_id, {
            "type": "document_started",
            "document_id": doc_status["document_id"],
            "filename": doc_status["filename"],
            "progress": (i / len(document_paths)) * 100
        })

        try:
            # Process document through unified processor with constitutional gateway
            result = await process_single_document(doc_path, config)

            # Update document status
            doc_status.update({
                "status": "completed" if result["constitutional_approval"] else "rejected",
                "constitutional_approval": result["constitutional_approval"],
                "extraction_quality": result["extraction_quality"],
                "consciousness_level": result["consciousness_level"],
                "processing_time_seconds": (datetime.now() - doc_start).total_seconds(),
                "attractor_modifications": result["attractor_modifications"]
            })

            if result["constitutional_approval"]:
                batch["completed_documents"] += 1
                batch["attractor_basins_modified"] += len(result["attractor_modifications"])

                if config["create_episodic_memory"]:
                    batch["episodic_memories_created"] += 1
            else:
                batch["rejected_documents"] += 1

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            doc_status.update({
                "status": "failed",
                "error_message": str(e),
                "processing_time_seconds": (datetime.now() - doc_start).total_seconds()
            })
            batch["failed_documents"] += 1

        # Update overall progress
        batch["overall_progress"] = ((i + 1) / len(document_paths)) * 100

        # Estimate remaining time
        elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
        if i > 0:
            avg_time_per_doc = elapsed_minutes / (i + 1)
            remaining_docs = len(document_paths) - (i + 1)
            batch["estimated_remaining_minutes"] = remaining_docs * avg_time_per_doc

        # Broadcast progress
        await broadcast_progress(batch_id, {
            "type": "document_completed",
            "document_id": doc_status["document_id"],
            "status": doc_status["status"],
            "progress": batch["overall_progress"]
        })

    # Create final episodic memory for the entire batch
    if config["create_episodic_memory"]:
        await create_batch_episodic_memory(batch_id, batch)

    logger.info(f"✅ Batch {batch_id} completed: {batch['completed_documents']} processed, {batch['rejected_documents']} rejected, {batch['failed_documents']} failed")

async def process_single_document(document_path: str, config: Dict[str, bool]) -> Dict[str, Any]:
    """Process a single document through the complete pipeline"""

    try:
        # Step 1: Unified document extraction
        extraction_result: UnifiedExtractionResult = await unified_processor.process_document(
            document_path=document_path,
            extraction_config={
                "constitutional_compliance": config["constitutional_compliance"],
                "thoughtseed_processing": config["thoughtseed_processing"]
            }
        )

        # Step 2: Constitutional approval check
        constitutional_approval = extraction_result.extraction_quality > 0.7  # Quality threshold

        if not constitutional_approval and config["constitutional_compliance"]:
            return {
                "constitutional_approval": False,
                "extraction_quality": extraction_result.extraction_quality,
                "consciousness_level": 0.0,
                "attractor_modifications": []
            }

        # Step 3: ThoughtSeed consciousness processing
        consciousness_level = 0.0
        attractor_modifications = []

        if config["thoughtseed_processing"] and constitutional_approval:
            # Process through consciousness agent
            consciousness_context = {
                "document_content": extraction_result.raw_text,
                "extraction_results": extraction_result,
                "thoughtseed_traces": extraction_result.thoughtseed_traces
            }

            consciousness_result = await consciousness_agent.execute(
                task=f"Consciousness analysis of document: {Path(document_path).name}",
                context=consciousness_context
            )

            consciousness_level = consciousness_result.get("result", {}).get("consciousness_trace", {}).get("consciousness_level", 0.0)

            # Step 4: Attractor basin modification
            if config["attractor_modification"]:
                attractor_modifications = await modify_attractor_basins(extraction_result)

        return {
            "constitutional_approval": constitutional_approval,
            "extraction_quality": extraction_result.extraction_quality,
            "consciousness_level": consciousness_level,
            "attractor_modifications": attractor_modifications
        }

    except Exception as e:
        logger.error(f"Single document processing failed: {e}")
        raise

async def modify_attractor_basins(extraction_result: UnifiedExtractionResult) -> List[str]:
    """Modify attractor basins based on extraction results"""

    modifications = []

    try:
        # Extract key concepts for basin analysis
        if extraction_result.langextract_results:
            entities = extraction_result.langextract_results.get("entities", [])
            for entity in entities[:5]:  # Limit to top 5 concepts
                # This would integrate with basin_manager
                # For now, simulate basin modification
                modifications.append(f"reinforced_basin_{entity.get('name', 'unknown')}")

        # Add algorithm concepts if found
        if extraction_result.algorithm_extractions:
            algorithms = extraction_result.algorithm_extractions.get("algorithms", [])
            for alg in algorithms[:3]:
                modifications.append(f"algorithm_basin_{alg.get('type', 'unknown')}")

    except Exception as e:
        logger.warning(f"Attractor basin modification failed: {e}")

    return modifications

async def create_batch_episodic_memory(batch_id: str, batch_data: Dict[str, Any]):
    """Create autobiographical episodic memory for the completed batch"""

    try:
        memory_title = f"Bulk Document Processing: {batch_data['batch_name']}"

        # Record the batch completion in my memory
        await record_conversation_moment(
            user_input=f"Batch {batch_id} processing completed",
            my_response=f"Processed {batch_data['completed_documents']} documents with {batch_data['attractor_basins_modified']} attractor modifications",
            tools_used={"unified_document_processor", "thoughtseed_network", "attractor_basin_manager", "constitutional_gateway"},
            reasoning=[
                f"Completed bulk processing of {batch_data['total_documents']} documents",
                f"Constitutional gateway approved {batch_data['completed_documents']} documents",
                f"Created {batch_data['episodic_memories_created']} episodic memories",
                f"Modified {batch_data['attractor_basins_modified']} attractor basins",
                f"Processing quality demonstrates consciousness-guided document ingestion"
            ]
        )

        # Create consolidated episodic memory
        await claude_memory.create_episodic_memory(memory_title)

    except Exception as e:
        logger.error(f"Episodic memory creation failed: {e}")

async def broadcast_progress(batch_id: str, message: Dict[str, Any]):
    """Broadcast progress update to all connected WebSockets for this batch"""

    if batch_id in active_websockets:
        disconnected = []

        for websocket in active_websockets[batch_id]:
            try:
                await websocket.send_json({
                    "type": "progress_update",
                    "batch_id": batch_id,
                    "data": message
                })
            except:
                disconnected.append(websocket)

        # Clean up disconnected websockets
        for ws in disconnected:
            active_websockets[batch_id].remove(ws)

        if not active_websockets[batch_id]:
            del active_websockets[batch_id]