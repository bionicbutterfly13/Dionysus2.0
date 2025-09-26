#!/usr/bin/env python3
"""
RAG API Fix Script
=================

This script fixes the missing opensearch-py dependency for the RAG API
without interfering with the ThoughtSeed trace model work in progress.

Safe workspace: /Volumes/Asylum/devb/ASI-Arch-Thoughtseeds/safe_workspace
"""

import subprocess
import sys
import os

def install_opensearch():
    """Install the missing opensearch-py dependency"""
    try:
        print("🔧 Installing opensearch-py dependency...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "opensearch-py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ opensearch-py installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing opensearch-py: {e}")
        return False

def test_rag_api():
    """Test if RAG API can start without errors"""
    try:
        print("🧪 Testing RAG API import...")
        
        # Change to cognition_base directory
        os.chdir("/Volumes/Asylum/devb/ASI-Arch-Thoughtseeds/cognition_base")
        
        # Test import
        import rag_service
        print("✅ RAG service imports successfully")
        
        # Test API initialization
        from rag_api import init_rag_service
        success = init_rag_service()
        
        if success:
            print("✅ RAG API initialization successful")
            return True
        else:
            print("⚠️ RAG API initialization had issues")
            return False
            
    except Exception as e:
        print(f"❌ RAG API test failed: {e}")
        return False

def main():
    """Main fix process"""
    print("🚀 RAG API Fix Process")
    print("=" * 50)
    
    # Step 1: Install dependency
    if install_opensearch():
        # Step 2: Test RAG API
        test_rag_api()
    
    print("\n✅ RAG API fix process complete")
    print("📍 Safe workspace: /Volumes/Asylum/devb/ASI-Arch-Thoughtseeds/safe_workspace")

if __name__ == "__main__":
    main()
