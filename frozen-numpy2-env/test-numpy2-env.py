
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
