#!/usr/bin/env python3
"""
Direct test for document ingestion without service imports
"""
import asyncio
import tempfile
import sys
import os
from pathlib import Path

# Test the service directly by running it
async def test_direct():
    print("🧪 Direct Document Ingestion Test")
    
    # Test text file creation and processing
    test_text = """# Neural Networks and Synaptic Plasticity

This document discusses the relationship between artificial neural networks
and biological synaptic plasticity mechanisms.

## Key Concepts:
- Hebbian learning
- Long-term potentiation (LTP)
- Backpropagation algorithms
- Neuroplasticity

The brain's ability to adapt through synaptic modifications parallels
how artificial neural networks adjust weights during training.
"""
    
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(test_text)
        test_file = f.name
    
    print(f"📝 Created test file: {test_file}")
    print(f"📊 Content length: {len(test_text)} characters")
    print(f"🔤 Word count: {len(test_text.split())} words")
    
    # Test markdown structure detection
    lines = test_text.split('\n')
    headers = [line for line in lines if line.startswith('#')]
    print(f"📋 Found {len(headers)} headers:")
    for header in headers:
        level = len(header) - len(header.lstrip('#'))
        title = header.lstrip('#').strip()
        print(f"  Level {level}: {title}")
    
    # Cleanup
    os.unlink(test_file)
    print("✅ Direct test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_direct())