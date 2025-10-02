#!/usr/bin/env python3
"""
Comprehensive test for document ingestion service
"""
import asyncio
import tempfile
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.document_ingestion import DocumentIngestionService, DocumentType

async def test_comprehensive_ingestion():
    """Test all document ingestion capabilities"""
    print("🧪 Comprehensive Document Ingestion Test")
    print("=" * 50)
    
    service = DocumentIngestionService()
    
    # Test 1: Plain text file
    print("\n📄 Test 1: Plain Text Processing")
    test_text = """# Neural Networks and Consciousness

Neural networks are computational systems inspired by biological neural networks. 
They consist of interconnected nodes (neurons) that process information through 
weighted connections (synapses).

## Key Concepts:
- Synaptic plasticity
- Backpropagation 
- Activation functions
- Deep learning architectures

This text contains neuroscience and AI concepts for testing domain-specific processing.
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_text)
        temp_file = f.name
    
    try:
        result = await service.ingest_document(temp_file)
        if result.success:
            print(f"✅ Text processing successful")
            print(f"  Document Type: {result.document_type.value}")
            print(f"  Word Count: {result.content.metadata.word_count}")
            print(f"  Processing Time: {result.processing_time:.3f}s")
            print(f"  Title: {result.content.metadata.title}")
        else:
            print(f"❌ Text processing failed: {result.error}")
    finally:
        os.unlink(temp_file)
    
    # Test 2: Markdown file
    print("\n📝 Test 2: Markdown Processing")
    markdown_text = """# Synaptic Plasticity Research

## Introduction
Synaptic plasticity is the ability of synapses to strengthen or weaken over time.

### Types of Plasticity:
1. **Long-term Potentiation (LTP)**
   - Activity-dependent increase in synaptic strength
   - Key mechanism for learning and memory
   
2. **Long-term Depression (LTD)**
   - Activity-dependent decrease in synaptic strength
   - Important for memory consolidation

## Molecular Mechanisms
- NMDA receptor activation
- Calcium influx
- Protein synthesis
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(markdown_text)
        temp_md_file = f.name
    
    try:
        result = await service.ingest_document(temp_md_file)
        if result.success:
            print(f"✅ Markdown processing successful")
            print(f"  Document Type: {result.document_type.value}")
            print(f"  Word Count: {result.content.metadata.word_count}")
            print(f"  Structure Elements: {len(result.content.structured_content)}")
            if result.content.structured_content:
                print(f"  First Header: {result.content.structured_content[0].get('title', 'N/A')}")
        else:
            print(f"❌ Markdown processing failed: {result.error}")
    finally:
        os.unlink(temp_md_file)
    
    # Test 3: Web URL processing
    print("\n🌐 Test 3: Web URL Processing")
    try:
        # Test with a simple neuroscience-related URL
        test_url = "https://en.wikipedia.org/wiki/Neuron"
        result = await service.ingest_document(test_url, source_type="url")
        
        if result.success:
            print(f"✅ Web processing successful")
            print(f"  Document Type: {result.document_type.value}")
            print(f"  Title: {result.content.metadata.title}")
            print(f"  Word Count: {result.content.metadata.word_count}")
            print(f"  Processing Time: {result.processing_time:.3f}s")
            print(f"  Content Preview: {result.content.raw_text[:100]}...")
        else:
            print(f"❌ Web processing failed: {result.error}")
    except Exception as e:
        print(f"❌ Web processing exception: {e}")
    
    # Test 4: Auto-detection
    print("\n🔍 Test 4: Auto-Detection")
    test_cases = [
        ("test.pdf", DocumentType.PDF),
        ("document.txt", DocumentType.PLAIN_TEXT),
        ("readme.md", DocumentType.MARKDOWN),
        ("https://example.com/paper.pdf", DocumentType.PDF),
        ("https://example.com", DocumentType.WEB_URL),
    ]
    
    for test_input, expected_type in test_cases:
        if test_input.startswith('http'):
            detected_type = service.detector.detect_from_url(test_input)
        else:
            detected_type = service.detector.detect_from_path(test_input)
        
        status = "✅" if detected_type == expected_type else "❌"
        print(f"  {status} {test_input} -> {detected_type.value} (expected: {expected_type.value})")
    
    # Test 5: Error handling
    print("\n⚠️ Test 5: Error Handling")
    try:
        result = await service.ingest_document("/nonexistent/file.txt")
        if not result.success:
            print(f"✅ Error handling working: {result.error}")
        else:
            print(f"❌ Error handling failed - should have errored")
    except Exception as e:
        print(f"✅ Exception handling working: {e}")
    
    # Test 6: Check supported formats
    print("\n📋 Test 6: Supported Formats")
    formats = await service.get_supported_formats()
    for fmt, available in formats.items():
        status = "✅" if available else "❌"
        print(f"  {status} {fmt}")
    
    print("\n🎉 Comprehensive ingestion test completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_comprehensive_ingestion())