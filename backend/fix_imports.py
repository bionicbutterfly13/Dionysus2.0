#!/usr/bin/env python3
"""
Fix absolute imports to relative imports in src/ directory.
Converts 'from models.X import Y' → 'from ..models.X import Y'
Converts 'from services.X import Y' → 'from ..services.X import Y'
"""

import re
from pathlib import Path

def fix_imports(file_path: Path):
    """Fix imports in a single file."""
    content = file_path.read_text()
    original = content

    # Count path depth from src/
    src_dir = Path(__file__).parent / "src"
    relative_path = file_path.relative_to(src_dir)
    depth = len(relative_path.parts) - 1  # -1 because file itself doesn't count

    # Determine correct relative import prefix
    if depth == 0:
        # File is directly in src/ - shouldn't happen
        return False
    elif depth == 1:
        # File is in src/X/ → use from .models or from .services
        prefix = "."
    else:
        # File is in src/X/Y/... → use from ..models or from ..services
        prefix = ".." * (depth - 1)

    # Fix "from models." imports
    content = re.sub(
        r'^from models\.',
        f'from {prefix}models.',
        content,
        flags=re.MULTILINE
    )

    # Fix "from services." and "from models." imports when in api/routes
    if 'api/routes' in str(file_path):
        # api/routes needs ...models (3 dots to go up to src/)
        content = re.sub(
            r'^from \.\.models\.',
            'from ...models.',
            content,
            flags=re.MULTILINE
        )
        # api/routes needs ...services (3 dots to go up to src/)
        content = re.sub(
            r'^from services\.',
            'from ...services.',
            content,
            flags=re.MULTILINE
        )
    elif 'services' not in str(file_path):
        # Not in services/, use relative to services/
        content = re.sub(
            r'^from services\.',
            f'from {prefix}services.',
            content,
            flags=re.MULTILINE
        )
    else:
        # Within services/, use .sibling
        content = re.sub(
            r'^from services\.',
            'from .',
            content,
            flags=re.MULTILINE
        )

    if content != original:
        file_path.write_text(content)
        print(f"✅ Fixed: {file_path.relative_to(Path(__file__).parent)}")
        return True
    return False

def main():
    src_dir = Path(__file__).parent / "src"

    # Find all Python files with absolute imports
    files_to_fix = []
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        content = py_file.read_text()
        if re.search(r'^from (models|services)\.', content, re.MULTILINE):
            files_to_fix.append(py_file)

    print(f"Found {len(files_to_fix)} files to fix")
    print()

    fixed_count = 0
    for file_path in files_to_fix:
        if fix_imports(file_path):
            fixed_count += 1

    print()
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main()
