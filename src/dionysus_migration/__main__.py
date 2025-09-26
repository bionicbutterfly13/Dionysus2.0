"""
Main entry point for Dionysus Migration CLI

Allows the package to be executed as a module:
python -m dionysus_migration
"""

from .cli.main import main

if __name__ == "__main__":
    main()