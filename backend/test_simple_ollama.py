#!/usr/bin/env python3
"""
Simple Ollama test to debug connection issues
"""
import asyncio
import httpx
import json

async def test_ollama_connection():
    """Test basic Ollama connection"""
    client = httpx.AsyncClient(timeout=30.0)
    
    try:
        # Test list models
        print("🔍 Testing model list...")
        response = await client.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Found {len(models['models'])} models:")
            for model in models['models']:
                print(f"  - {model['name']}")
        else:
            print(f"❌ Failed to list models: {response.status_code}")
            return False
        
        # Test generation
        print("\n🧠 Testing text generation...")
        gen_request = {
            "model": "qwen2.5:7b",
            "prompt": "Extract concepts from: 'Neural networks process information through synaptic connections.'",
            "stream": False,
            "options": {"num_predict": 50}
        }
        
        print(f"Sending request: {json.dumps(gen_request, indent=2)}")
        response = await client.post(
            "http://localhost:11434/api/generate",
            json=gen_request,
            timeout=60.0  # Longer timeout
        )
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generation successful:")
            print(f"  Response: {result['response'][:100]}...")
            print(f"  Duration: {result.get('total_duration', 0) / 1e9:.2f}s")
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
        
        # Test embeddings
        print("\n📊 Testing embeddings...")
        embed_request = {
            "model": "nomic-embed-text",
            "prompt": "neural networks and consciousness"
        }
        
        response = await client.post(
            "http://localhost:11434/api/embeddings",
            json=embed_request,
            timeout=20.0
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get('embedding', [])
            print(f"✅ Embeddings successful:")
            print(f"  Dimensions: {len(embedding)}")
            print(f"  Sample values: {embedding[:5]}")
        else:
            print(f"❌ Embeddings failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
        
        print("\n🎉 All Ollama tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(test_ollama_connection())