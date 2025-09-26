#!/usr/bin/env python3
"""
Setup script for Pydantic LogFire integration
===========================================

This script installs and configures LogFire for ThoughtSeed observability.
"""

import subprocess
import sys
import os

def install_logfire():
    """Install LogFire and dependencies"""
    print("🔥 Installing Pydantic LogFire...")

    packages = [
        "logfire",
        "pydantic>=2.0.0",
        "opentelemetry-api",
        "opentelemetry-sdk"
    ]

    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("✅ LogFire installation complete!")

def configure_logfire():
    """Configure LogFire for the project"""
    print("⚙️ Configuring LogFire...")

    # Create logfire configuration
    config_code = '''
import logfire

# Configure LogFire for ThoughtSeed project
logfire.configure(
    service_name="thoughtseed-asi-go-2",
    environment="development",
    send_to_logfire=True  # Set to False for local-only logging
)

print("🔥 LogFire configured for ThoughtSeed!")
'''

    with open("logfire_config.py", "w") as f:
        f.write(config_code)

    print("✅ LogFire configuration created!")

def test_logfire():
    """Test LogFire installation"""
    print("🧪 Testing LogFire...")

    try:
        import logfire

        # Quick test
        logfire.configure(service_name="test")
        logfire.info("LogFire test message", test_data={"status": "working"})

        print("✅ LogFire test successful!")
        return True

    except ImportError as e:
        print(f"❌ LogFire import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ LogFire test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Pydantic LogFire for ThoughtSeed")
    print("=" * 50)

    try:
        install_logfire()
        configure_logfire()

        if test_logfire():
            print("\n🎉 LogFire setup complete!")
            print("You can now run the enhanced ThoughtSeed tracing:")
            print("  python demo_logfire_comparison.py")
        else:
            print("\n⚠️ LogFire setup incomplete - check errors above")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)