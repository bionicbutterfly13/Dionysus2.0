#!/usr/bin/env python3
"""
Ollama Health Check Script
Verifies that Ollama is running and required models are available.
"""

import sys
import os
import requests
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running at http://localhost:11434")
            return True, response.json()
        else:
            print(f"❌ Ollama server returned status {response.status_code}")
            return False, None
    except requests.exceptions.ConnectionError:
        print("❌ Ollama server is not running")
        print("   Start with: ollama serve")
        return False, None
    except Exception as e:
        print(f"❌ Error checking Ollama server: {e}")
        return False, None

def check_required_models(models_data):
    """Check if required models are installed"""
    if not models_data:
        return False

    installed_models = [m['name'] for m in models_data.get('models', [])]

    # Required models (any one of these)
    required_models = ['qwen2.5:7b', 'qwen2.5:14b', 'llama3.2:3b']
    embedding_model = 'nomic-embed-text'

    # Check for at least one reasoning model
    has_reasoning_model = any(model in installed_models for model in required_models)
    has_embedding_model = embedding_model in installed_models

    print("\n📦 Installed Models:")
    for model in installed_models:
        marker = "✅" if model in required_models + [embedding_model] else "ℹ️ "
        print(f"   {marker} {model}")

    print("\n🔍 Required Models Check:")
    if has_reasoning_model:
        print("   ✅ Reasoning model available")
    else:
        print("   ❌ No reasoning model found")
        print(f"      Install with: ollama pull {required_models[0]}")

    if has_embedding_model:
        print("   ✅ Embedding model available")
    else:
        print("   ⚠️  Embedding model not found (optional)")
        print(f"      Install with: ollama pull {embedding_model}")

    return has_reasoning_model

def check_env_config():
    """Check .env configuration"""
    env_path = Path(__file__).parent.parent.parent / ".env"

    if not env_path.exists():
        print("\n❌ .env file not found")
        print("   Copy .env.example to .env and configure")
        return False

    # Read env file
    env_vars = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()

    print("\n⚙️  Environment Configuration:")

    provider = env_vars.get('LLM_PROVIDER', 'not set')
    if provider == 'ollama':
        print(f"   ✅ LLM_PROVIDER={provider}")
    else:
        print(f"   ⚠️  LLM_PROVIDER={provider} (should be 'ollama' for local processing)")

    endpoint = env_vars.get('OLLAMA_ENDPOINT', 'not set')
    print(f"   {'✅' if 'localhost:11434' in endpoint else 'ℹ️ '} OLLAMA_ENDPOINT={endpoint}")

    model = env_vars.get('OLLAMA_MODEL', 'not set')
    print(f"   {'✅' if model in ['qwen2.5:7b', 'qwen2.5:14b', 'llama3.2:3b'] else 'ℹ️ '} OLLAMA_MODEL={model}")

    return provider == 'ollama'

def test_ollama_query():
    """Test a simple query to Ollama"""
    print("\n🧪 Testing Ollama Query...")

    try:
        from dotenv import load_dotenv
        load_dotenv()

        model = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')
        endpoint = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')

        response = requests.post(
            f"{endpoint}/api/generate",
            json={
                "model": model,
                "prompt": "Say 'Hello' in one word.",
                "stream": False,
                "options": {
                    "num_predict": 5,
                    "temperature": 0.1
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            output = result.get('response', '').strip()
            print(f"   ✅ Query successful")
            print(f"   Model: {model}")
            print(f"   Response: {output}")
            return True
        else:
            print(f"   ❌ Query failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Query failed: {e}")
        return False

def main():
    print("=" * 70)
    print("Ollama Health Check")
    print("Dionysus 2.0 - Local LLM Verification")
    print("=" * 70)

    # Check Ollama server
    server_running, models_data = check_ollama_server()

    if not server_running:
        print("\n⚠️  Ollama server is not running. Start it with: ollama serve")
        return 1

    # Check installed models
    has_models = check_required_models(models_data)

    # Check environment configuration
    env_configured = check_env_config()

    # Test query
    query_works = False
    if server_running and has_models and env_configured:
        query_works = test_ollama_query()

    # Summary
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)

    all_good = server_running and has_models and env_configured and query_works

    if all_good:
        print("✅ All checks passed! Ollama is ready for ingestion.")
        print("\nYou can now run:")
        print("   python backend/scripts/ingest_openspec_specs.py --all")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")

        if not server_running:
            print("\n1. Start Ollama: ollama serve")
        if not has_models:
            print("\n2. Install model: ollama pull qwen2.5:7b")
        if not env_configured:
            print("\n3. Set LLM_PROVIDER=ollama in .env")
        if not query_works and server_running and has_models:
            print("\n4. Test manually: ollama run qwen2.5:7b 'Hello'")

        return 1

if __name__ == "__main__":
    sys.exit(main())
