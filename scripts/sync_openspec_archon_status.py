#!/usr/bin/env python3
"""
Sync Archon task status back to OpenSpec tasks.md checkboxes.

Usage:
    python scripts/sync_openspec_archon_status.py integrate-openspec-archon-sync
    python scripts/sync_openspec_archon_status.py --change-id ingest-specs-to-neo4j
"""

import sys
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from difflib import SequenceMatcher


class ArchonStatusPoller:
    """Poll Archon task status and sync to OpenSpec tasks.md."""

    def __init__(self, archon_url: str = "http://localhost:8051"):
        self.archon_url = archon_url
        self.base_path = Path("openspec/changes")

    def read_project_id(self, change_id: str) -> str:
        """Read Archon project_id from .archon-project-id file."""
        id_file = self.base_path / change_id / ".archon-project-id"

        if not id_file.exists():
            raise FileNotFoundError(
                f"No .archon-project-id found for '{change_id}'. "
                f"Run /openspec:import-to-archon {change_id} first."
            )

        return id_file.read_text().strip()

    def fetch_archon_tasks(self, project_id: str) -> List[Dict]:
        """Fetch all tasks from Archon for a project."""
        url = f"{self.archon_url}/mcp"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "find_tasks",
                "arguments": {
                    "project_id": project_id,
                    "per_page": 100  # Get all tasks
                }
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Archon MCP at {self.archon_url}: {e}")

        result = response.json()
        if "error" in result:
            raise Exception(f"Archon error: {result['error']}")

        # Parse the result JSON string
        result_data = json.loads(result["result"]["content"][0]["text"])

        if not result_data.get("success", False):
            raise Exception(f"Archon query failed: {result_data.get('error', 'Unknown error')}")

        return result_data.get("tasks", [])

    def parse_tasks_md(self, change_id: str) -> List[Dict]:
        """Parse tasks.md and extract tasks with their current checkbox state."""
        tasks_path = self.base_path / change_id / "tasks.md"

        if not tasks_path.exists():
            raise FileNotFoundError(f"tasks.md not found at {tasks_path}")

        content = tasks_path.read_text()
        tasks = []
        current_phase = None
        line_number = 0

        for line in content.split('\n'):
            line_number += 1

            # Detect phase headers
            phase_match = re.match(r'^## (.+)$', line)
            if phase_match:
                current_phase = phase_match.group(1)
                continue

            # Parse checklist items
            task_match = re.match(r'^- \[([ x~-])\] (.+)$', line)
            if task_match:
                checkbox = task_match.group(1)
                task_title = task_match.group(2).strip()

                tasks.append({
                    "title": task_title,
                    "checkbox": checkbox,  # ' ', 'x', '~', '-'
                    "phase": current_phase,
                    "line_number": line_number,
                    "original_line": line
                })

        return tasks

    def similarity(self, a: str, b: str) -> float:
        """Calculate similarity ratio between two strings (0.0 to 1.0)."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def match_tasks(self, archon_tasks: List[Dict], md_tasks: List[Dict]) -> Dict[int, Dict]:
        """
        Match Archon tasks to tasks.md entries by title similarity.

        Returns dict mapping md_task index to archon_task data.
        """
        matches = {}

        for md_idx, md_task in enumerate(md_tasks):
            best_match = None
            best_score = 0.0

            for archon_task in archon_tasks:
                score = self.similarity(md_task["title"], archon_task["title"])
                if score > best_score and score > 0.85:  # 85% similarity threshold
                    best_score = score
                    best_match = archon_task

            if best_match:
                matches[md_idx] = {
                    "archon_task": best_match,
                    "similarity": best_score
                }

        return matches

    def map_status_to_checkbox(self, status: str) -> str:
        """Map Archon task status to checkbox symbol."""
        mapping = {
            "done": "x",
            "review": "~",  # Optional: show review state
            "doing": "-",   # Optional: show in-progress
            "todo": " "
        }
        return mapping.get(status, " ")

    def compute_updates(self, md_tasks: List[Dict], matches: Dict[int, Dict]) -> Dict[int, str]:
        """
        Compute which checkboxes need updating.

        Returns dict mapping md_task index to new checkbox symbol.
        """
        updates = {}

        for md_idx, match_data in matches.items():
            md_task = md_tasks[md_idx]
            archon_task = match_data["archon_task"]

            current_checkbox = md_task["checkbox"]
            archon_status = archon_task["status"]
            new_checkbox = self.map_status_to_checkbox(archon_status)

            # Update if Archon status differs from current checkbox
            # Exception: Never downgrade 'x' to ' ' (manual completion wins)
            if new_checkbox != current_checkbox:
                if current_checkbox == 'x' and new_checkbox == ' ':
                    # Manual completion takes precedence
                    continue
                updates[md_idx] = new_checkbox

        return updates

    def apply_updates(self, change_id: str, md_tasks: List[Dict], updates: Dict[int, str]) -> str:
        """
        Apply checkbox updates to tasks.md.

        Returns updated content.
        """
        tasks_path = self.base_path / change_id / "tasks.md"
        content = tasks_path.read_text()
        lines = content.split('\n')

        for md_idx, new_checkbox in updates.items():
            md_task = md_tasks[md_idx]
            line_idx = md_task["line_number"] - 1  # Convert to 0-indexed

            # Replace checkbox in line
            old_line = lines[line_idx]
            new_line = re.sub(r'^- \[[ x~-]\]', f'- [{new_checkbox}]', old_line)
            lines[line_idx] = new_line

        return '\n'.join(lines)

    def calculate_completion(self, md_tasks: List[Dict]) -> tuple:
        """Calculate completion percentage from checkbox states."""
        total = len(md_tasks)
        completed = sum(1 for task in md_tasks if task["checkbox"] == 'x')
        percentage = int((completed / total) * 100) if total > 0 else 0
        return completed, total, percentage

    def sync_status(self, change_id: str, dry_run: bool = False) -> Dict:
        """
        Sync Archon task status to OpenSpec tasks.md.

        Args:
            change_id: OpenSpec change ID
            dry_run: If True, show changes without writing

        Returns:
            Dict with sync results
        """
        print(f"Syncing Archon status for: {change_id}")
        print("-" * 60)

        # Step 1: Read project ID
        print("1. Reading .archon-project-id...")
        try:
            project_id = self.read_project_id(change_id)
            print(f"   Project ID: {project_id}")
        except FileNotFoundError as e:
            print(f"   ✗ {e}")
            return {"success": False, "error": str(e)}

        # Step 2: Fetch Archon tasks
        print("\n2. Fetching Archon tasks...")
        try:
            archon_tasks = self.fetch_archon_tasks(project_id)
            print(f"   Found {len(archon_tasks)} tasks in Archon")
        except Exception as e:
            print(f"   ✗ {e}")
            return {"success": False, "error": str(e)}

        # Step 3: Parse tasks.md
        print("\n3. Parsing tasks.md...")
        try:
            md_tasks = self.parse_tasks_md(change_id)
            print(f"   Found {len(md_tasks)} tasks in tasks.md")
        except Exception as e:
            print(f"   ✗ {e}")
            return {"success": False, "error": str(e)}

        # Step 4: Match tasks
        print("\n4. Matching tasks...")
        matches = self.match_tasks(archon_tasks, md_tasks)
        print(f"   Matched {len(matches)}/{len(md_tasks)} tasks")

        # Step 5: Compute updates
        print("\n5. Computing updates...")
        updates = self.compute_updates(md_tasks, matches)
        print(f"   {len(updates)} checkboxes need updating")

        if updates:
            print("\n   Updates:")
            for md_idx, new_checkbox in updates.items():
                task = md_tasks[md_idx]
                print(f"   - [{task['checkbox']}] → [{new_checkbox}] {task['title']}")

        # Step 6: Apply updates
        if updates and not dry_run:
            print("\n6. Applying updates to tasks.md...")
            updated_content = self.apply_updates(change_id, md_tasks, updates)

            # Write back
            tasks_path = self.base_path / change_id / "tasks.md"
            tasks_path.write_text(updated_content)
            print(f"   ✓ tasks.md updated")

            # Re-parse to get updated completion stats
            md_tasks = self.parse_tasks_md(change_id)
        elif dry_run:
            print("\n6. Skipping write (dry-run mode)")

        # Step 7: Calculate completion
        completed, total, percentage = self.calculate_completion(md_tasks)

        # Summary
        print("\n" + "=" * 60)
        if updates and not dry_run:
            print("✅ Sync complete!")
        elif dry_run:
            print("✅ Dry-run complete! (no changes written)")
        else:
            print("✅ Already in sync!")

        print(f"\nCompletion: {completed}/{total} tasks ({percentage}%)")
        print(f"Updates applied: {len(updates) if not dry_run else 0}")

        if not dry_run:
            print("\nNext steps:")
            print(f"  - Review changes: git diff openspec/changes/{change_id}/tasks.md")
            print(f"  - Commit: git add . && git commit -m 'chore: sync task status from Archon [{completed}/{total} complete]'")

        return {
            "success": True,
            "project_id": project_id,
            "tasks_matched": len(matches),
            "tasks_total": len(md_tasks),
            "updates_applied": len(updates) if not dry_run else 0,
            "completion": {
                "completed": completed,
                "total": total,
                "percentage": percentage
            }
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync Archon task status to OpenSpec tasks.md"
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing to tasks.md"
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
                    archon_id_file = change / ".archon-project-id"
                    marker = " (synced)" if archon_id_file.exists() else ""
                    print(f"  - {change.name}{marker}")
        sys.exit(1)

    # Sync status
    poller = ArchonStatusPoller(archon_url=args.archon_url)
    try:
        result = poller.sync_status(change_id, dry_run=args.dry_run)
        sys.exit(0 if result["success"] else 1)
    except Exception as e:
        print(f"\n❌ Sync failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
