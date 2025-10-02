#!/usr/bin/env python3
"""
Simple integration test for Ollama that avoids timeouts
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.ollama_integration import OllamaModelManager, OllamaConceptExtractor, ConceptExtractionRequest

async def simple_test():
    """Test Ollama integration with simpler timeout handling"""
    print("🚀 Starting simple Ollama integration test...")
    
    # Initialize manager
    manager = OllamaModelManager()
    
    # Check available models
    print("\n📋 Configured models:")
    for name, config in manager.models.items():
        print(f"  - {name}: {config.type.value} (timeout: {config.timeout}s)")
    
    # Test just one model at a time
    test_models = ["qwen2.5:7b", "nomic-embed-text"]
    
    for model_name in test_models:
        if model_name in manager.models:
            print(f"\n🔍 Testing {model_name}...")
            try:
                health = await manager.check_model_health(model_name)
                print(f"  Status: {health.status.value}")
                print(f"  Response time: {health.response_time:.2f}s")
                
                if health.status.name in ["HEALTHY", "SLOW"]:
                    print(f"  ✅ {model_name} is working")
                else:
                    print(f"  ❌ {model_name} health check failed")
                    
            except Exception as e:
                print(f"  ❌ {model_name} failed: {e}")
    
    # Test concept extraction with healthy model
    print("\n🧠 Testing concept extraction...")
    extractor = OllamaConceptExtractor(manager)
    
    # Simple test request
    request = ConceptExtractionRequest(
        text="Neural networks learn patterns.",
        granularity_level=1,
        domain_focus=["ai"],
        extraction_type="atomic_concepts"
    )
    
    try:
        result = await extractor.extract_atomic_concepts(request)
        if result.success:
            print(f"✅ Concept extraction successful: {len(result.concepts)} concepts")
        else:
            print(f"❌ Concept extraction failed: {result.error}")
    except Exception as e:
        print(f"❌ Concept extraction exception: {e}")
    
    # Cleanup
    await manager.cleanup()
    print("\n🎉 Simple test completed!")

if __name__ == "__main__":
    asyncio.run(simple_test())