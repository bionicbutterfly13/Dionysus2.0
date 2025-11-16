#!/usr/bin/env python3
"""
Import OpenSpec change proposals into Archon MCP as projects with tasks.

Usage:
    python scripts/import_openspec_to_archon.py integrate-openspec-archon-sync
    python scripts/import_openspec_to_archon.py --change-id ingest-specs-to-neo4j
"""

import sys
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional


class OpenSpecToArchonImporter:
    """Import OpenSpec changes to Archon MCP."""

    def __init__(self, archon_url: str = "http://localhost:8051"):
        self.archon_url = archon_url
        self.base_path = Path("openspec/changes")

    def read_proposal(self, change_id: str) -> Dict[str, str]:
        """Read and parse proposal.md for metadata."""
        proposal_path = self.base_path / change_id / "proposal.md"

        if not proposal_path.exists():
            raise FileNotFoundError(f"Change '{change_id}' not found at {proposal_path}")

        content = proposal_path.read_text()

        # Extract title (first H1)
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else change_id

        # Extract description (## What section)
        what_match = re.search(r'## What\s+(.+?)(?=\n##|\Z)', content, re.DOTALL)
        description = what_match.group(1).strip() if what_match else ""

        return {
            "title": title,
            "description": description,
            "change_id": change_id
        }

    def parse_tasks(self, change_id: str) -> List[Dict[str, any]]:
        """Parse tasks from tasks.md."""
        tasks_path = self.base_path / change_id / "tasks.md"

        if not tasks_path.exists():
            raise FileNotFoundError(f"tasks.md not found at {tasks_path}")

        content = tasks_path.read_text()
        tasks = []
        current_phase = None

        for line in content.split('\n'):
            # Detect phase headers
            phase_match = re.match(r'^## (.+)$', line)
            if phase_match:
                current_phase = phase_match.group(1)
                continue

            # Parse checklist items
            task_match = re.match(r'^- \[([ x])\] (.+)$', line)
            if task_match:
                checked = task_match.group(1) == 'x'
                task_title = task_match.group(2).strip()

                tasks.append({
                    "title": task_title,
                    "status": "done" if checked else "todo",
                    "phase": current_phase,
                    "index": len(tasks)
                })

        return tasks

    def create_archon_project(self, metadata: Dict[str, str]) -> str:
        """Create Archon project and return project_id."""
        url = f"{self.archon_url}/mcp"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "manage_project",
                "arguments": {
                    "action": "create",
                    "title": metadata["title"],
                    "description": metadata["description"]
                }
            }
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        if "error" in result:
            raise Exception(f"Archon error: {result['error']}")

        # Parse the result JSON string
        result_data = json.loads(result["result"]["content"][0]["text"])
        project_id = result_data["project"]["id"]

        return project_id

    def create_archon_tasks(self, project_id: str, tasks: List[Dict]) -> int:
        """Create Archon tasks and return count."""
        url = f"{self.archon_url}/mcp"
        created_count = 0

        for idx, task in enumerate(tasks):
            payload = {
                "jsonrpc": "2.0",
                "id": idx + 2,
                "method": "tools/call",
                "params": {
                    "name": "manage_task",
                    "arguments": {
                        "action": "create",
                        "project_id": project_id,
                        "title": task["title"],
                        "description": f"Phase: {task['phase']}" if task['phase'] else "",
                        "status": task["status"],
                        "task_order": 100 - idx  # Higher order = higher priority
                    }
                }
            }

            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                created_count += 1
                print(f"  ✓ Task {idx + 1}/{len(tasks)}: {task['title']}")
            except Exception as e:
                print(f"  ✗ Task {idx + 1}/{len(tasks)} failed: {e}")

        return created_count

    def store_project_id(self, change_id: str, project_id: str):
        """Store Archon project_id in .archon-project-id file."""
        id_file = self.base_path / change_id / ".archon-project-id"
        id_file.write_text(project_id)

    def import_change(self, change_id: str) -> Dict[str, any]:
        """Import an OpenSpec change to Archon."""
        print(f"Importing OpenSpec change: {change_id}")
        print("-" * 60)

        # Step 1: Read proposal
        print("1. Reading proposal.md...")
        metadata = self.read_proposal(change_id)
        print(f"   Title: {metadata['title']}")
        print(f"   Description: {metadata['description'][:80]}...")

        # Step 2: Parse tasks
        print("\n2. Parsing tasks.md...")
        tasks = self.parse_tasks(change_id)
        print(f"   Found {len(tasks)} tasks")

        # Step 3: Create Archon project
        print("\n3. Creating Archon project...")
        try:
            project_id = self.create_archon_project(metadata)
            print(f"   ✓ Project created: {project_id}")
        except Exception as e:
            print(f"   ✗ Failed to create project: {e}")
            return {"success": False, "error": str(e)}

        # Step 4: Create Archon tasks
        print("\n4. Creating Archon tasks...")
        created_count = self.create_archon_tasks(project_id, tasks)

        # Step 5: Store project reference
        print("\n5. Storing project reference...")
        self.store_project_id(change_id, project_id)
        print(f"   ✓ Stored in .archon-project-id")

        # Summary
        print("\n" + "=" * 60)
        print("✅ Import complete!")
        print(f"\nArchon Project: {project_id}")
        print(f"Tasks created: {created_count}/{len(tasks)}")
        print(f"Reference stored: .archon-project-id")
        print("\nNext steps:")
        print(f"  - View tasks: find_tasks(filter_by='project', filter_value='{project_id}')")
        print(f"  - Start working: manage_task('update', task_id='...', status='doing')")
        print(f"  - When done: /openspec:archive {change_id}")

        return {
            "success": True,
            "project_id": project_id,
            "tasks_created": created_count,
            "tasks_total": len(tasks)
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Import OpenSpec changes to Archon MCP"
    )
    parser.add_argument(
        "change_id",
        nargs="?",
        help="OpenSpec change ID (e.g., integrate-openspec-archon-sync)"
    )
    parser.add_argument(
        "--change-id",
        dest="change_id_flag",
        help="Alternative: specify change ID via flag"
    )
    parser.add_argument(
        "--archon-url",
        default="http://localhost:8051",
        help="Archon MCP server URL (default: http://localhost:8051)"
    )

    args = parser.parse_args()

    # Get change_id from positional arg or flag
    change_id = args.change_id or args.change_id_flag

    if not change_id:
        parser.print_help()
        print("\nAvailable changes:")
        changes_dir = Path("openspec/changes")
        if changes_dir.exists():
            for change in changes_dir.iterdir():
                if change.is_dir() and not change.name.startswith('.'):
                    print(f"  - {change.name}")
        sys.exit(1)

    # Import the change
    importer = OpenSpecToArchonImporter(archon_url=args.archon_url)
    try:
        result = importer.import_change(change_id)
        sys.exit(0 if result["success"] else 1)
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
