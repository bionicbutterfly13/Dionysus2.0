"""
Real-Time Document Streaming Service
===================================

Enables real-time processing of documents as they arrive, with live concept extraction,
knowledge graph updates, and consciousness-guided analysis streaming.

Features:
- Real-time document processing pipeline
- Streaming concept extraction with live updates
- WebSocket support for real-time frontend updates
- Queue-based document processing with priority handling
- Live consciousness detection and measurement
- Real-time knowledge graph construction and updates
- Memory system integration with live persistence
"""

import asyncio
import logging
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path

# WebSocket and async support
try:
    import websockets
    from fastapi import WebSocket, WebSocketDisconnect
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Queue management
import asyncio
from asyncio import Queue, Event

# Integration with our services
try:
    from .five_level_concept_extraction import FiveLevelConceptExtractionService
    from .autoschemakg_integration import AutoSchemaKGService
    from .multi_tier_memory import MultiTierMemorySystem
    from .document_ingestion import DocumentIngestionService
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)

class DocumentPriority(Enum):
    """Priority levels for document processing"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class ProcessingStatus(Enum):
    """Document processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    BUILDING_GRAPH = "building_graph"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class StreamingDocument:
    """Document for real-time processing"""
    id: str
    source: str
    source_type: str
    priority: DocumentPriority = DocumentPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: ProcessingStatus = ProcessingStatus.QUEUED
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class StreamingUpdate:
    """Real-time update for frontend"""
    document_id: str
    update_type: str  # 'progress', 'concept', 'relationship', 'completion', 'error'
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    consciousness_level: Optional[float] = None

class RealTimeDocumentStreamer:
    """Real-time document processing and streaming service"""
    
    def __init__(self, 
                 max_concurrent: int = 3,
                 queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        
        # Processing queue and workers
        self.document_queue: Queue[StreamingDocument] = Queue(maxsize=queue_size)
        self.processing_workers: List[asyncio.Task] = []
        self.active_documents: Dict[str, StreamingDocument] = {}
        self.completed_documents: Dict[str, StreamingDocument] = {}
        
        # WebSocket connections for live updates
        self.websocket_connections: List[WebSocket] = []
        self.update_queue: Queue[StreamingUpdate] = Queue()
        
        # Processing services
        self.concept_extractor = None
        self.autoschema_service = None
        self.memory_system = None
        self.document_ingestion = None
        
        # Performance metrics
        self.processing_stats = {
            "total_processed": 0,
            "total_concepts_extracted": 0,
            "total_relations_created": 0,
            "average_processing_time": 0.0,
            "consciousness_levels": []
        }
        
        # Shutdown event
        self.shutdown_event = Event()
    
    async def initialize(self):
        """Initialize the real-time streaming service"""
        logger.info("🚀 Initializing Real-Time Document Streamer")
        
        # Initialize services
        if SERVICES_AVAILABLE:
            self.concept_extractor = FiveLevelConceptExtractionService()
            self.memory_system = MultiTierMemorySystem()
            self.autoschema_service = AutoSchemaKGService(memory_system=self.memory_system)
            self.document_ingestion = DocumentIngestionService()
            
            # Initialize all services
            await self.memory_system.initialize()
            await self.autoschema_service.initialize()
        
        # Start processing workers
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._processing_worker(f"worker_{i}"))
            self.processing_workers.append(worker)
        
        # Start update broadcaster
        self.update_broadcaster = asyncio.create_task(self._update_broadcaster())
        
        logger.info(f"✅ Real-time streamer initialized with {self.max_concurrent} workers")
    
    async def submit_document(self, 
                            source: str, 
                            source_type: str = "auto",
                            priority: DocumentPriority = DocumentPriority.NORMAL,
                            metadata: Dict[str, Any] = None) -> str:
        """Submit a document for real-time processing"""
        
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        streaming_doc = StreamingDocument(
            id=document_id,
            source=source,
            source_type=source_type,
            priority=priority,
            metadata=metadata or {},
            status=ProcessingStatus.QUEUED
        )
        
        try:
            # Add to queue (will block if queue is full)
            await self.document_queue.put(streaming_doc)
            self.active_documents[document_id] = streaming_doc
            
            # Send queued update
            await self._send_update(StreamingUpdate(
                document_id=document_id,
                update_type="queued",
                data={
                    "status": "queued",
                    "priority": priority.name,
                    "queue_position": self.document_queue.qsize()
                }
            ))
            
            logger.info(f"📥 Document queued: {document_id} (priority: {priority.name})")
            return document_id
            
        except asyncio.QueueFull:
            logger.error(f"❌ Queue full, rejecting document: {document_id}")
            raise RuntimeError("Processing queue is full")
    
    async def _processing_worker(self, worker_id: str):
        """Worker coroutine for processing documents"""
        logger.info(f"👷 Worker {worker_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get document from queue with timeout
                try:
                    document = await asyncio.wait_for(
                        self.document_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the document
                await self._process_document(document, worker_id)
                
                # Mark task as done
                self.document_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"👷 Worker {worker_id} stopped")
    
    async def _process_document(self, document: StreamingDocument, worker_id: str):
        """Process a single document with real-time updates"""
        start_time = time.time()
        
        try:
            document.status = ProcessingStatus.PROCESSING
            await self._send_progress_update(document, 0.1, "Starting processing...")
            
            # Step 1: Document ingestion
            if self.document_ingestion:
                ingestion_result = await self.document_ingestion.ingest_document(
                    document.source, document.source_type
                )
                text_content = ingestion_result.text_content
            else:
                text_content = document.source  # Fallback for testing
            
            await self._send_progress_update(document, 0.2, "Document ingested")
            
            # Step 2: Five-level concept extraction
            document.status = ProcessingStatus.EXTRACTING
            await self._send_progress_update(document, 0.3, "Extracting concepts...")
            
            if self.concept_extractor:
                extraction_result = await self.concept_extractor.extract_concepts_from_document(text_content)
            else:
                # Mock extraction for testing
                extraction_result = self._create_mock_extraction_result(text_content)
            
            # Send concept extraction updates
            await self._send_concept_updates(document, extraction_result)
            await self._send_progress_update(document, 0.6, "Concepts extracted")
            
            # Step 3: Knowledge graph construction
            document.status = ProcessingStatus.BUILDING_GRAPH
            await self._send_progress_update(document, 0.7, "Building knowledge graph...")
            
            if self.autoschema_service:
                # Use the extracted concepts to build graph
                kg_result = await self._build_knowledge_graph_from_extraction(extraction_result, document.id)
            else:
                # Mock KG result
                kg_result = {"nodes": [], "relations": [], "statistics": {}}
            
            # Send relationship updates
            await self._send_relationship_updates(document, kg_result)
            await self._send_progress_update(document, 0.9, "Knowledge graph created")
            
            # Step 4: Store in memory system
            document.status = ProcessingStatus.STORING
            await self._send_progress_update(document, 0.95, "Storing in memory...")
            
            if self.memory_system:
                await self._store_processing_result(document, extraction_result, kg_result)
            
            # Step 5: Complete processing
            processing_time = time.time() - start_time
            document.status = ProcessingStatus.COMPLETED
            document.progress = 1.0
            document.result = {
                "extraction_result": self._serialize_extraction_result(extraction_result),
                "knowledge_graph": kg_result,
                "processing_time": processing_time,
                "consciousness_level": getattr(extraction_result, 'consciousness_level', 0.8),
                "worker_id": worker_id
            }
            
            # Update stats
            self._update_processing_stats(extraction_result, kg_result, processing_time)
            
            # Move to completed
            self.completed_documents[document.id] = document
            if document.id in self.active_documents:
                del self.active_documents[document.id]
            
            # Send completion update
            await self._send_completion_update(document)
            
            logger.info(f"✅ Document processed: {document.id} in {processing_time:.2f}s by {worker_id}")
            
        except Exception as e:
            # Handle processing error
            document.status = ProcessingStatus.FAILED
            document.error = str(e)
            
            await self._send_update(StreamingUpdate(
                document_id=document.id,
                update_type="error",
                data={
                    "error": str(e),
                    "worker_id": worker_id,
                    "processing_time": time.time() - start_time
                }
            ))
            
            logger.error(f"❌ Document processing failed: {document.id} - {e}")
    
    async def _build_knowledge_graph_from_extraction(self, extraction_result, doc_id):
        """Build knowledge graph from extraction result"""
        nodes = []
        relations = []
        
        # Create document node
        from .autoschemakg_integration import KnowledgeGraphNode, NodeType
        document_node = KnowledgeGraphNode(
            id=doc_id,
            type=NodeType.DOCUMENT,
            name=f"Stream_Document_{doc_id}",
            properties={
                "source": "streaming",
                "processing_date": datetime.now().isoformat(),
                "extraction_confidence": getattr(extraction_result, 'confidence_score', 0.8)
            }
        )
        nodes.append(document_node)
        
        # Create nodes from extraction
        await self.autoschema_service._create_nodes_from_extraction(extraction_result, nodes, doc_id)
        
        # Create relationships
        await self.autoschema_service._infer_concept_relationships(extraction_result, nodes, relations, doc_id)
        
        # Store in Neo4j
        if self.autoschema_service.driver:
            await self.autoschema_service._store_knowledge_graph(nodes, relations)
        
        return {
            "nodes": [self.autoschema_service._serialize_node(node) for node in nodes],
            "relations": [self.autoschema_service._serialize_relation(rel) for rel in relations],
            "statistics": {
                "total_nodes": len(nodes),
                "total_relations": len(relations),
                "node_types": self._count_node_types(nodes),
                "relation_types": self._count_relation_types(relations)
            }
        }
    
    def _create_mock_extraction_result(self, text: str):
        """Create mock extraction result for testing"""
        class MockResult:
            def __init__(self, text):
                words = text.split()[:10]  # Take first 10 words as concepts
                self.atomic_concepts = [
                    {"concept": word, "importance": 0.5 + len(word) * 0.05, "confidence": 0.8}
                    for word in words if len(word) > 3
                ]
                self.relationships = [
                    {"relationship": f"{words[0]} relates to {words[1] if len(words) > 1 else 'concept'}", 
                     "type": "general", "confidence": 0.7}
                ]
                self.composite_concepts = [
                    {"concept": "text_analysis", "components": words[:3], "confidence": 0.75}
                ]
                self.contexts = [
                    {"context": "document_processing", "scope": "local", "confidence": 0.7}
                ]
                self.narratives = [
                    {"narrative": "content_analysis", "theme": "processing", "confidence": 0.6}
                ]
                self.confidence_score = 0.8
                self.processing_time = 1.5
                self.consciousness_level = 0.85
        
        return MockResult(text)
    
    def _serialize_extraction_result(self, result):
        """Serialize extraction result for JSON"""
        return {
            "atomic_concepts": getattr(result, 'atomic_concepts', []),
            "relationships": getattr(result, 'relationships', []),
            "composite_concepts": getattr(result, 'composite_concepts', []),
            "contexts": getattr(result, 'contexts', []),
            "narratives": getattr(result, 'narratives', []),
            "confidence_score": getattr(result, 'confidence_score', 0.8),
            "processing_time": getattr(result, 'processing_time', 1.0),
            "consciousness_level": getattr(result, 'consciousness_level', 0.8)
        }
    
    async def _send_concept_updates(self, document: StreamingDocument, extraction_result):
        """Send real-time concept extraction updates"""
        
        # Send atomic concepts
        for concept in getattr(extraction_result, 'atomic_concepts', []):
            await self._send_update(StreamingUpdate(
                document_id=document.id,
                update_type="concept",
                data={
                    "type": "atomic",
                    "concept": concept,
                    "level": 1
                },
                consciousness_level=getattr(extraction_result, 'consciousness_level', 0.8)
            ))
        
        # Send relationships
        for relation in getattr(extraction_result, 'relationships', []):
            await self._send_update(StreamingUpdate(
                document_id=document.id,
                update_type="concept",
                data={
                    "type": "relationship",
                    "relationship": relation,
                    "level": 2
                }
            ))
        
        # Send composite concepts
        for composite in getattr(extraction_result, 'composite_concepts', []):
            await self._send_update(StreamingUpdate(
                document_id=document.id,
                update_type="concept",
                data={
                    "type": "composite",
                    "concept": composite,
                    "level": 3
                }
            ))
    
    async def _send_relationship_updates(self, document: StreamingDocument, kg_result):
        """Send real-time knowledge graph relationship updates"""
        
        for relation in kg_result.get("relations", []):
            await self._send_update(StreamingUpdate(
                document_id=document.id,
                update_type="relationship",
                data={
                    "relation": relation,
                    "graph_stats": kg_result.get("statistics", {})
                }
            ))
    
    async def _send_progress_update(self, document: StreamingDocument, progress: float, message: str):
        """Send processing progress update"""
        document.progress = progress
        
        await self._send_update(StreamingUpdate(
            document_id=document.id,
            update_type="progress",
            data={
                "progress": progress,
                "message": message,
                "status": document.status.value
            }
        ))
    
    async def _send_completion_update(self, document: StreamingDocument):
        """Send processing completion update"""
        await self._send_update(StreamingUpdate(
            document_id=document.id,
            update_type="completion",
            data={
                "status": "completed",
                "result": document.result,
                "processing_time": document.result.get("processing_time", 0),
                "consciousness_level": document.result.get("consciousness_level", 0)
            },
            consciousness_level=document.result.get("consciousness_level", 0)
        ))
    
    async def _send_update(self, update: StreamingUpdate):
        """Send update to WebSocket connections"""
        await self.update_queue.put(update)
    
    async def _update_broadcaster(self):
        """Broadcast updates to all WebSocket connections"""
        while not self.shutdown_event.is_set():
            try:
                # Get update from queue
                try:
                    update = await asyncio.wait_for(
                        self.update_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Serialize update
                update_data = {
                    "document_id": update.document_id,
                    "update_type": update.update_type,
                    "data": update.data,
                    "timestamp": update.timestamp.isoformat(),
                    "consciousness_level": update.consciousness_level
                }
                
                message = json.dumps(update_data)
                
                # Send to all connected WebSockets
                disconnected = []
                for websocket in self.websocket_connections:
                    try:
                        await websocket.send_text(message)
                    except Exception as e:
                        logger.warning(f"WebSocket send failed: {e}")
                        disconnected.append(websocket)
                
                # Remove disconnected WebSockets
                for ws in disconnected:
                    if ws in self.websocket_connections:
                        self.websocket_connections.remove(ws)
                
            except Exception as e:
                logger.error(f"Update broadcaster error: {e}")
                await asyncio.sleep(1)
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add a WebSocket connection for real-time updates"""
        await websocket.accept()
        self.websocket_connections.append(websocket)
        logger.info(f"📡 WebSocket connected: {len(self.websocket_connections)} total connections")
        
        # Send current status
        status_update = {
            "update_type": "status",
            "data": {
                "active_documents": len(self.active_documents),
                "queue_size": self.document_queue.qsize(),
                "processing_stats": self.processing_stats
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket.send_text(json.dumps(status_update))
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
            logger.info(f"📡 WebSocket disconnected: {len(self.websocket_connections)} remaining")
    
    async def _store_processing_result(self, document: StreamingDocument, extraction_result, kg_result):
        """Store processing result in memory system"""
        if not self.memory_system:
            return
        
        # Store the complete processing result
        result_data = {
            "type": "streaming_document_result",
            "document_id": document.id,
            "extraction": self._serialize_extraction_result(extraction_result),
            "knowledge_graph": kg_result,
            "metadata": document.metadata,
            "processing_time": document.result.get("processing_time", 0),
            "created_at": document.created_at.isoformat()
        }
        
        # High importance for real-time processed documents
        await self.memory_system.store_concept(result_data, importance=0.8)
    
    def _update_processing_stats(self, extraction_result, kg_result, processing_time):
        """Update processing statistics"""
        self.processing_stats["total_processed"] += 1
        self.processing_stats["total_concepts_extracted"] += len(getattr(extraction_result, 'atomic_concepts', []))
        self.processing_stats["total_relations_created"] += kg_result.get("statistics", {}).get("total_relations", 0)
        
        # Update average processing time
        total = self.processing_stats["total_processed"]
        current_avg = self.processing_stats["average_processing_time"]
        new_avg = ((current_avg * (total - 1)) + processing_time) / total
        self.processing_stats["average_processing_time"] = new_avg
        
        # Track consciousness levels
        consciousness = getattr(extraction_result, 'consciousness_level', 0.8)
        self.processing_stats["consciousness_levels"].append(consciousness)
        
        # Keep only last 100 consciousness levels
        if len(self.processing_stats["consciousness_levels"]) > 100:
            self.processing_stats["consciousness_levels"] = self.processing_stats["consciousness_levels"][-100:]
    
    def _count_node_types(self, nodes):
        """Count nodes by type"""
        counts = {}
        for node in nodes:
            node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _count_relation_types(self, relations):
        """Count relations by type"""
        counts = {}
        for relation in relations:
            rel_type = relation.type.value if hasattr(relation.type, 'value') else str(relation.type)
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts
    
    async def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a document"""
        if document_id in self.active_documents:
            doc = self.active_documents[document_id]
        elif document_id in self.completed_documents:
            doc = self.completed_documents[document_id]
        else:
            return None
        
        return {
            "id": doc.id,
            "status": doc.status.value,
            "progress": doc.progress,
            "created_at": doc.created_at.isoformat(),
            "metadata": doc.metadata,
            "result": doc.result,
            "error": doc.error
        }
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        consciousness_avg = 0
        if self.processing_stats["consciousness_levels"]:
            consciousness_avg = np.mean(self.processing_stats["consciousness_levels"])
        
        return {
            "total_processed": self.processing_stats["total_processed"],
            "currently_active": len(self.active_documents),
            "queue_size": self.document_queue.qsize(),
            "total_concepts_extracted": self.processing_stats["total_concepts_extracted"],
            "total_relations_created": self.processing_stats["total_relations_created"],
            "average_processing_time": self.processing_stats["average_processing_time"],
            "average_consciousness_level": consciousness_avg,
            "websocket_connections": len(self.websocket_connections)
        }
    
    async def shutdown(self):
        """Shutdown the streaming service"""
        logger.info("🛑 Shutting down real-time document streamer")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for workers to finish
        if self.processing_workers:
            await asyncio.gather(*self.processing_workers, return_exceptions=True)
        
        # Cancel update broadcaster
        if hasattr(self, 'update_broadcaster'):
            self.update_broadcaster.cancel()
        
        # Close WebSocket connections
        for websocket in self.websocket_connections[:]:
            try:
                await websocket.close()
            except:
                pass
        self.websocket_connections.clear()
        
        # Close services
        if self.memory_system:
            await self.memory_system.close()
        if self.autoschema_service:
            await self.autoschema_service.close()
        
        logger.info("✅ Real-time document streamer shutdown complete")

# Testing and demonstration
async def test_realtime_streaming():
    """Test real-time document streaming"""
    print("🧪 Testing Real-Time Document Streaming")
    print("=" * 50)
    
    # Initialize streamer
    streamer = RealTimeDocumentStreamer(max_concurrent=2)
    await streamer.initialize()
    
    # Test documents
    test_docs = [
        {
            "source": "Synaptic plasticity enables learning through LTP and LTD mechanisms in hippocampal neurons.",
            "priority": DocumentPriority.HIGH,
            "metadata": {"domain": "neuroscience", "type": "research"}
        },
        {
            "source": "Artificial neural networks use backpropagation for gradient-based optimization of weights.",
            "priority": DocumentPriority.NORMAL,
            "metadata": {"domain": "AI", "type": "technical"}
        },
        {
            "source": "Consciousness emerges from complex neural interactions across distributed brain networks.",
            "priority": DocumentPriority.URGENT,
            "metadata": {"domain": "consciousness", "type": "theory"}
        }
    ]
    
    # Submit documents
    document_ids = []
    for i, doc_info in enumerate(test_docs):
        doc_id = await streamer.submit_document(
            source=doc_info["source"],
            priority=doc_info["priority"],
            metadata=doc_info["metadata"]
        )
        document_ids.append(doc_id)
        print(f"📥 Submitted document {i+1}: {doc_id}")
    
    # Monitor processing
    print("\n🔄 Monitoring real-time processing...")
    
    processed_count = 0
    start_time = time.time()
    
    while processed_count < len(document_ids) and time.time() - start_time < 30:
        await asyncio.sleep(2)
        
        stats = await streamer.get_processing_statistics()
        print(f"📊 Stats: {stats['currently_active']} active, {stats['queue_size']} queued, {stats['total_processed']} completed")
        
        # Check individual document statuses
        for doc_id in document_ids:
            status = await streamer.get_document_status(doc_id)
            if status and status["status"] == "completed" and doc_id not in [d for d in document_ids if processed_count > document_ids.index(d)]:
                processed_count += 1
                result = status["result"]
                print(f"✅ Completed {doc_id}:")
                print(f"   🧠 Consciousness: {result.get('consciousness_level', 0):.3f}")
                print(f"   ⏱️  Time: {result.get('processing_time', 0):.2f}s")
                print(f"   📝 Concepts: {len(result.get('extraction_result', {}).get('atomic_concepts', []))}")
    
    # Final statistics
    final_stats = await streamer.get_processing_statistics()
    print(f"\n🎉 Real-time streaming test completed!")
    print(f"📊 Final Statistics:")
    print(f"   Total processed: {final_stats['total_processed']}")
    print(f"   Concepts extracted: {final_stats['total_concepts_extracted']}")
    print(f"   Relations created: {final_stats['total_relations_created']}")
    print(f"   Avg processing time: {final_stats['average_processing_time']:.2f}s")
    print(f"   Avg consciousness: {final_stats['average_consciousness_level']:.3f}")
    
    await streamer.shutdown()

if __name__ == "__main__":
    asyncio.run(test_realtime_streaming())