#!/bin/bash
# ASI-Arch Frozen Environment Setup
# Constitutional compliance: NumPy < 2.0 MANDATORY
# Version: 1.0.0
# Date: 2025-09-24

set -e  # Exit on any error

echo "🏛️ ASI-Arch Frozen Environment Setup"
echo "======================================"
echo "Constitutional compliance: NumPy < 2.0 MANDATORY"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
if [[ $python_version == 3.11.* ]] || [[ $python_version == 3.10.* ]]; then
    print_success "Python $python_version compatible"
else
    print_warning "Python $python_version detected (3.10+ recommended)"
fi

# Step 2: Create frozen environment
print_status "Creating frozen environment..."
if [ -d "asi-arch-frozen-env" ]; then
    print_warning "Frozen environment already exists, removing..."
    rm -rf asi-arch-frozen-env
fi

python3 -m venv asi-arch-frozen-env
print_success "Frozen environment created"

# Step 3: Activate environment
print_status "Activating frozen environment..."
source asi-arch-frozen-env/bin/activate
print_success "Environment activated"

# Step 4: Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip
print_success "Pip upgraded"

# Step 5: Install NumPy FIRST (Constitutional requirement)
print_status "Installing NumPy < 2.0 (Constitutional requirement)..."
pip install "numpy<2" --force-reinstall
print_success "NumPy < 2.0 installed"

# Step 6: Verify NumPy compliance
print_status "Verifying NumPy compliance..."
python -c "
import numpy
version = numpy.__version__
if version.startswith('2.'):
    print('❌ CONSTITUTION VIOLATION: NumPy 2.x detected')
    exit(1)
else:
    print(f'✅ NumPy {version} compliant')
"
print_success "NumPy compliance verified"

# Step 7: Install frozen requirements
print_status "Installing frozen requirements..."
if [ -f "requirements-frozen.txt" ]; then
    pip install -r requirements-frozen.txt
    print_success "Frozen requirements installed"
else
    print_error "requirements-frozen.txt not found"
    exit 1
fi

# Step 8: Run constitutional compliance check
print_status "Running constitutional compliance check..."
python constitutional_compliance_checker.py
if [ $? -eq 0 ]; then
    print_success "Constitutional compliance verified"
else
    print_error "Constitutional compliance check failed"
    exit 1
fi

# Step 9: Test ThoughtSeed integration
print_status "Testing ThoughtSeed integration..."
python -c "
import sys
sys.path.append('backend/models')
from thoughtseed_trace import ThoughtSeedTrace
print('✅ ThoughtSeed Trace Model import successful')
"
print_success "ThoughtSeed integration verified"

# Step 10: Create activation script
print_status "Creating activation script..."
cat > activate_frozen_env.sh << 'EOF'
#!/bin/bash
# ASI-Arch Frozen Environment Activation
# Constitutional compliance: NumPy < 2.0 MANDATORY

echo "🏛️ Activating ASI-Arch Frozen Environment"
echo "Constitutional compliance: NumPy < 2.0 MANDATORY"
echo ""

# Activate environment
source asi-arch-frozen-env/bin/activate

# Verify compliance
python -c "
import numpy
version = numpy.__version__
if version.startswith('2.'):
    print('❌ CONSTITUTION VIOLATION: NumPy 2.x detected')
    print('Run: pip install \"numpy<2\" --force-reinstall')
    exit(1)
else:
    print(f'✅ NumPy {version} compliant')
"

echo "✅ Frozen environment activated and verified"
echo "🚀 Ready for ASI-Arch operations"
EOF

chmod +x activate_frozen_env.sh
print_success "Activation script created"

# Step 11: Create deactivation script
print_status "Creating deactivation script..."
cat > deactivate_frozen_env.sh << 'EOF'
#!/bin/bash
# ASI-Arch Frozen Environment Deactivation

echo "🏛️ Deactivating ASI-Arch Frozen Environment"
deactivate
echo "✅ Frozen environment deactivated"
EOF

chmod +x deactivate_frozen_env.sh
print_success "Deactivation script created"

# Step 12: Final verification
print_status "Running final verification..."
python -c "
import numpy, torch, transformers
print(f'NumPy: {numpy.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
assert numpy.__version__.startswith('1.'), 'CONSTITUTION VIOLATION'
print('✅ All dependencies constitution-compliant')
"
print_success "Final verification complete"

# Summary
echo ""
echo "🎉 ASI-Arch Frozen Environment Setup Complete!"
echo "============================================="
echo ""
echo "✅ Environment: asi-arch-frozen-env"
echo "✅ NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "✅ Constitutional compliance: VERIFIED"
echo "✅ ThoughtSeed integration: READY"
echo ""
echo "📋 Usage Instructions:"
echo "  Activate:   source activate_frozen_env.sh"
echo "  Deactivate: source deactivate_frozen_env.sh"
echo "  Check:      python constitutional_compliance_checker.py"
echo ""
echo "🛡️ Constitutional Requirements:"
echo "  - NumPy version MUST be < 2.0"
echo "  - Always use frozen environment"
echo "  - Run compliance check before operations"
echo "  - Coordinate with other agents"
echo ""
echo "🚀 Ready for ASI-Arch operations!"
