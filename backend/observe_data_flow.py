#!/usr/bin/env python3
"""
Real-time Data Flow Observer for Flux Backend
Monitors document processing pipeline and displays data transformations
"""

import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging to show all info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Import and patch loggers to observe data flow
from src.services.daedalus import logger as daedalus_logger
from src.services.document_processing_graph import logger as graph_logger
from src.services.consciousness_document_processor import logger as consciousness_logger

# Set all loggers to DEBUG level for maximum observability
daedalus_logger.setLevel(logging.DEBUG)
graph_logger.setLevel(logging.DEBUG)
consciousness_logger.setLevel(logging.DEBUG)

print("=" * 80)
print("🔍 FLUX DATA FLOW OBSERVER")
print("=" * 80)
print("\nMonitoring:")
print("  - Daedalus Gateway (perceptual information reception)")
print("  - Document Processing Graph (LangGraph workflow)")
print("  - Consciousness Document Processor (basin creation)")
print("\nUpload a document to see the complete data flow...")
print("=" * 80)
print()

# Keep script running
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\nObserver stopped.")
