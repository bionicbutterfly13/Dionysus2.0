#!/usr/bin/env python3
"""
Debug Streaming API
====================

Provides a server-sent events (SSE) stream that exposes real-time visibility into
the document processing pipeline. Designed for the debug panel to surface queue
status, LangGraph node transitions, and cognition telemetry without requiring
external tooling.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Dict, Any, List, Optional
import asyncio
import json
import logging
from datetime import datetime
from uuid import uuid4

router = APIRouter(prefix="/api/debug", tags=["debug"])
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Shared state (in-memory for development / single-process usage)
# -----------------------------------------------------------------------------
processing_queue: List[Dict[str, Any]] = []
active_processing: Dict[str, Any] = {}
_connected_clients: List[asyncio.Queue] = []


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _format_timestamp() -> str:
    return datetime.now().isoformat()


def _format_sse(event: Dict[str, Any]) -> str:
    """Convert an event dict into SSE wire format."""
    return f"data: {json.dumps(event)}\n\n"


def _queue_snapshot() -> Dict[str, Any]:
    """Generate a snapshot of current queue/active processing state."""
    return {
        "type": "queue_status",
        "queue": [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "position": idx,
                "status": doc.get("status", "queued"),
                "queued_at": doc.get("queued_at"),
            }
            for idx, doc in enumerate(processing_queue)
        ],
        "active": active_processing,
        "timestamp": _format_timestamp(),
    }


async def _broadcast_event(event: Dict[str, Any]) -> None:
    """Send an event to all connected SSE clients."""
    event.setdefault("timestamp", _format_timestamp())
    if not _connected_clients:
        logger.debug("[DEBUG STREAM] No connected clients; skipping broadcast.")
        return

    for queue in list(_connected_clients):
        try:
            await queue.put(event)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"[DEBUG STREAM] Failed to queue event for client: {exc}")


async def broadcast_queue_status() -> None:
    """Broadcast the latest queue snapshot."""
    await _broadcast_event(_queue_snapshot())


def register_document_in_queue(filename: str, document_id: str) -> Dict[str, Any]:
    """Append a document to the in-memory queue and return the queued record."""
    doc = {
        "id": document_id,
        "filename": filename,
        "status": "queued",
        "queued_at": _format_timestamp(),
    }
    processing_queue.append(doc)
    return doc


# Backward compatibility for internal imports that still reference the private name
_add_document_to_queue = register_document_in_queue


# -----------------------------------------------------------------------------
# SSE Endpoint
# -----------------------------------------------------------------------------
@router.get("/stream")
async def stream_debug_events() -> StreamingResponse:
    """
    Server-Sent Events endpoint delivering real-time pipeline telemetry.

    Event Types:
        - init: initial payload (queue depth)
        - queue_status: queue + active processing snapshot
        - processing_start / node_start / node_complete / processing_complete
        - concept_discovered / basin_activated / thoughtseed_generated
        - quality_metric / insight_discovered / knowledge_graph_created / processing_error
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        client_queue: asyncio.Queue = asyncio.Queue()
        _connected_clients.append(client_queue)

        try:
            # Immediate initialization payload
            await client_queue.put(
                {
                    "type": "init",
                    "queue_size": len(processing_queue),
                    "timestamp": _format_timestamp(),
                }
            )

            while True:
                try:
                    event = await asyncio.wait_for(client_queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    event = _queue_snapshot()

                yield _format_sse(event)

        finally:
            if client_queue in _connected_clients:
                _connected_clients.remove(client_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------
@router.post("/queue-document")
async def queue_document(filename: str, document_id: str) -> Dict[str, Any]:
    """HTTP endpoint to manually append a document to the debug queue."""
    register_document_in_queue(filename, document_id)
    await broadcast_queue_status()
    return {
        "status": "queued",
        "position": len(processing_queue) - 1,
        "document_id": document_id,
    }


@router.post("/process-document")
async def process_document_for_debug(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tags: Optional[str] = None,
    max_iterations: int = 3,
    quality_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Upload a document and process it through the debug pipeline.

    The document is enqueued immediately so it appears in the queue column,
    and processing runs in the background to stream LangGraph events.
    """
    content = await file.read()
    filename = file.filename or "unknown_document"
    document_id = str(uuid4())

    _add_document_to_queue(filename, document_id)
    await broadcast_queue_status()

    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    background_tasks.add_task(
        _run_debug_processing,
        content,
        filename,
        document_id,
        tag_list,
        max_iterations,
        quality_threshold,
    )

    return {
        "status": "queued",
        "document_id": document_id,
        "filename": filename,
        "message": "Document queued for debug processing",
    }


# -----------------------------------------------------------------------------
# Event broadcasting helpers (used by DebugDocumentProcessor)
# -----------------------------------------------------------------------------
async def emit_processing_event(event: Dict[str, Any]) -> None:
    """Emit a processing event to connected debug clients."""
    logger.info("[DEBUG EVENT] %s", json.dumps(event, ensure_ascii=False))
    await _broadcast_event(event)


async def _run_debug_processing(
    content: bytes,
    filename: str,
    document_id: str,
    tags: List[str],
    max_iterations: int,
    quality_threshold: float,
) -> None:
    """
    Background task: execute the debug processor and emit streamed events.
    """
    from ...services.debug_document_processor import DebugDocumentProcessor  # Local import to avoid circular dependency

    processor = DebugDocumentProcessor()

    try:
        async for event in processor.process_with_debug_events(
            content=content,
            filename=filename,
            document_id=document_id,
            tags=tags,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
        ):
            await emit_processing_event(event)

        # Ensure queue snapshot reflects completion
        await broadcast_queue_status()

    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("Debug processing failed: %s", exc)
        await emit_processing_event(
            {
                "type": "processing_error",
                "document_id": document_id,
                "error": str(exc),
            }
        )
        await broadcast_queue_status()
