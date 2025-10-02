#!/usr/bin/env python3
"""
Standalone Ollama test using direct imports
"""
import asyncio
import sys
import os
import importlib.util
import httpx
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

# Load the ollama_integration module directly
spec = importlib.util.spec_from_file_location(
    "ollama_integration", 
    "src/services/ollama_integration.py"
)
ollama_module = importlib.util.module_from_spec(spec)

# Add required dependencies
sys.modules['ollama_integration'] = ollama_module

class ModelType(Enum):
    PRIMARY = "primary"
    FAST = "fast" 
    EMBEDDING = "embedding"

class ModelStatus(Enum):
    HEALTHY = "healthy"
    SLOW = "slow"
    ERROR = "error"

@dataclass
class ModelConfig:
    name: str
    type: ModelType
    endpoint: str = "http://localhost:11434"
    timeout: float = 60.0

@dataclass 
class ModelHealthStatus:
    model_name: str
    status: ModelStatus
    last_check: str
    response_time: float

async def test_standalone_ollama():
    """Test Ollama without full service imports"""
    print("🧪 Testing standalone Ollama integration...")
    
    client = httpx.AsyncClient(timeout=60.0)
    
    # Test models
    models_to_test = {
        "qwen2.5:7b": ModelType.FAST,
        "nomic-embed-text": ModelType.EMBEDDING
    }
    
    for model_name, model_type in models_to_test.items():
        print(f"\n🔍 Testing {model_name}...")
        start_time = time.time()
        
        try:
            if model_type == ModelType.EMBEDDING:
                # Test embedding endpoint
                response = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={
                        "model": model_name,
                        "prompt": "test neural networks"
                    },
                    timeout=60.0
                )
            else:
                # Test generation endpoint  
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "Extract concepts from: neural networks",
                        "stream": False,
                        "options": {"num_predict": 20}
                    },
                    timeout=60.0
                )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"  ✅ {model_name} working - {response_time:.2f}s")
                if model_type == ModelType.EMBEDDING:
                    result = response.json()
                    print(f"    Embedding dimensions: {len(result.get('embedding', []))}")
                else:
                    result = response.json()
                    print(f"    Response: {result.get('response', '')[:50]}...")
            else:
                print(f"  ❌ {model_name} failed - HTTP {response.status_code}")
                print(f"    Error: {response.text}")
                
        except Exception as e:
            print(f"  ❌ {model_name} exception: {e}")
    
    await client.aclose()
    print("\n🎉 Standalone test completed!")

if __name__ == "__main__":
    asyncio.run(test_standalone_ollama())