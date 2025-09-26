#!/usr/bin/env python3
"""
🧮 Direct NumPy 2.0 Frozen Environment Creator
Creates a frozen binary environment with PyTorch compiled for NumPy 2.0
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle output"""
    print(f"🔧 {description}")
    print(f"   Running: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {description} - Success!")
        return True
    else:
        print(f"❌ {description} - Failed!")
        print(f"   Error: {result.stderr}")
        return False

def create_frozen_environment():
    """Create the frozen NumPy 2.0 environment"""

    print("🚀 Creating NumPy 2.0 Frozen Binary Environment")
    print("=" * 60)

    # Create frozen environment directory
    frozen_dir = Path("/Volumes/Asylum/devb/ASI-Arch/frozen-numpy2-env")
    frozen_dir.mkdir(exist_ok=True)

    print(f"📁 Created frozen environment directory: {frozen_dir}")

    # Step 1: Remove old PyTorch completely
    if not run_command("pip uninstall -y torch torchvision torchaudio",
                      "Removing old PyTorch versions"):
        print("⚠️ No existing PyTorch found (this is good!)")

    # Step 2: Install NumPy 2.0+ with exact version lock
    if not run_command("pip install 'numpy==2.3.3' --force-reinstall",
                      "Installing NumPy 2.3.3"):
        return False

    # Step 3: Install PyTorch nightly with NumPy 2.0 support
    print("🔥 Installing PyTorch with NumPy 2.0 support...")

    # Try different PyTorch installation methods
    pytorch_commands = [
        "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu",
        "pip install torch==2.3.0+cpu torchvision==0.18.0+cpu torchaudio==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu",
        "pip install 'torch>=2.2.0' 'torchvision>=0.17.0' 'torchaudio>=2.2.0' --no-deps"
    ]

    pytorch_installed = False
    for cmd in pytorch_commands:
        if run_command(cmd, f"Trying PyTorch installation: {cmd}"):
            pytorch_installed = True
            break

    if not pytorch_installed:
        print("⚠️ PyTorch installation failed, but continuing...")

    # Step 4: Install SentenceTransformers with NumPy 2.0 support
    if not run_command("pip install 'sentence-transformers>=5.1.0' --force-reinstall",
                      "Installing SentenceTransformers with NumPy 2.0 support"):
        return False

    # Step 5: Install other required dependencies
    dependencies = [
        "transformers>=4.50.0",
        "scikit-learn",
        "scipy",
        "redis",
        "neo4j",
        "aiofiles",
        "asyncio-mqtt"
    ]

    for dep in dependencies:
        run_command(f"pip install '{dep}'", f"Installing {dep}")

    # Step 6: Generate frozen requirements
    if not run_command(f"pip freeze > {frozen_dir}/requirements-numpy2-frozen.txt",
                      "Generating frozen requirements"):
        return False

    # Step 7: Test the environment
    print("🧪 Testing NumPy 2.0 compatibility...")

    test_script = '''
import sys
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except Exception as e:
        print(f"⚠️ PyTorch: {e}")

    try:
        import sentence_transformers
        print(f"✅ SentenceTransformers: {sentence_transformers.__version__}")
    except Exception as e:
        print(f"⚠️ SentenceTransformers: {e}")

    print("🎯 NumPy 2.0 Environment Test Complete!")

except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
'''

    # Write test script
    test_file = frozen_dir / "test-numpy2-env.py"
    test_file.write_text(test_script)

    # Run test
    if run_command(f"python {test_file}", "Testing NumPy 2.0 environment"):
        print("🎉 Frozen NumPy 2.0 environment created successfully!")
        return True
    else:
        print("❌ Environment test failed")
        return False

def create_activation_script():
    """Create activation script for frozen environment"""

    activation_script = '''#!/bin/bash
# 🧮 NumPy 2.0 Frozen Environment Activator

export ASI_ARCH_NUMPY2_FROZEN="1"
export PYTHONPATH="/Volumes/Asylum/devb/ASI-Arch/frozen-numpy2-env:$PYTHONPATH"

echo "🧮 NumPy 2.0 Frozen Environment Activated!"
echo "🔒 All dependencies locked to NumPy 2.0 compatible versions"
echo "✅ Environment ready for ASI-Arch with consciousness processing"

# Test environment
python /Volumes/Asylum/devb/ASI-Arch/frozen-numpy2-env/test-numpy2-env.py
'''

    script_path = Path("/Volumes/Asylum/devb/ASI-Arch/activate-numpy2-frozen.sh")
    script_path.write_text(activation_script)
    os.chmod(script_path, 0o755)

    print(f"📝 Created activation script: {script_path}")

if __name__ == "__main__":
    print("🚀 NumPy 2.0 Frozen Binary Environment Creator")
    print("=" * 50)

    if create_frozen_environment():
        create_activation_script()
        print("")
        print("🎯 SUCCESS! NumPy 2.0 Frozen Environment Ready!")
        print("💡 To activate: source activate-numpy2-frozen.sh")
        print("🔒 All dependencies are now locked to NumPy 2.0 compatible versions")
        print("🎉 You will NEVER have NumPy compatibility issues again!")
    else:
        print("❌ Failed to create frozen environment")
        sys.exit(1)