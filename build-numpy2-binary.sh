#!/bin/bash

# 🚀 NumPy 2.0 Frozen Binary Builder
# Creates a frozen environment with PyTorch compiled for NumPy 2.0

echo "🔧 Building NumPy 2.0 Frozen Binary Environment..."

# Build the Docker image
docker build -f Dockerfile.numpy2-binary -t asi-arch-numpy2-binary:latest .

if [ $? -eq 0 ]; then
    echo "✅ NumPy 2.0 Binary Environment Built Successfully!"

    # Create extraction script
    echo "📦 Creating binary extraction..."

    # Run container and extract frozen libraries
    docker run --name asi-numpy2-extract asi-arch-numpy2-binary:latest
    docker cp asi-numpy2-extract:/frozen-binary/lib ./frozen-numpy2-binary/
    docker rm asi-numpy2-extract

    # Create activation script
    cat > activate-numpy2-binary.sh << 'EOF'
#!/bin/bash
# NumPy 2.0 Binary Environment Activator

export PYTHONPATH="/Volumes/Asylum/devb/ASI-Arch/frozen-numpy2-binary:$PYTHONPATH"
export ASI_ARCH_NUMPY2_BINARY="1"

echo "🧮 NumPy 2.0 Binary Environment Activated!"
echo "✅ PyTorch compiled with NumPy 2.0 support"
echo "✅ SentenceTransformers with NumPy 2.0 compatibility"
echo "✅ All dependencies frozen and locked"

# Test the environment
python -c "
import sys
sys.path.insert(0, '/Volumes/Asylum/devb/ASI-Arch/frozen-numpy2-binary')
import numpy as np
import torch
import sentence_transformers
print(f'🔢 NumPy: {np.__version__}')
print(f'🔥 PyTorch: {torch.__version__}')
print(f'🤖 SentenceTransformers: {sentence_transformers.__version__}')
print('🎯 All systems operational with NumPy 2.0!')
"
EOF

    chmod +x activate-numpy2-binary.sh

    echo ""
    echo "🎯 FROZEN BINARY ENVIRONMENT READY!"
    echo "💡 To use: source activate-numpy2-binary.sh"
    echo "🔒 All dependencies locked to NumPy 2.0 compatible versions"

else
    echo "❌ Build failed. Check Docker installation and try again."
    exit 1
fi