#!/usr/bin/env python3
"""
OpenSpec Specification Ingestion Script
Dionysus 2.0

Ingests OpenSpec specification documents (spec.md, design.md) into Neo4j
through the Daedalus → LangGraph → DocumentRepository consciousness pipeline.

REQUIREMENTS:
    - Ollama running locally (http://localhost:11434)
    - Required models: qwen2.5:7b (or qwen2.5:14b, llama3.2:3b)
    - Neo4j running (bolt://localhost:7687)
    - Redis running (localhost:6379)
    - Backend API server running (http://localhost:9127)

Setup Ollama:
    1. Install: curl -fsSL https://ollama.com/install.sh | sh
    2. Pull model: ollama pull qwen2.5:7b
    3. Verify: ollama list

Environment:
    - LLM_PROVIDER=ollama (set in .env)
    - OLLAMA_ENDPOINT=http://localhost:11434
    - OLLAMA_MODEL=qwen2.5:7b

Usage:
    python backend/scripts/ingest_openspec_specs.py --all
    python backend/scripts/ingest_openspec_specs.py --capability document-processing
    python backend/scripts/ingest_openspec_specs.py --all --dry-run

Spec: openspec/changes/ingest-specs-to-neo4j/
"""

import os
import sys
import hashlib
import requests
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import signal


class OpenSpecIngester:
    """Ingests OpenSpec specs into Neo4j via Daedalus consciousness pipeline."""

    def __init__(self, api_base_url: str = "http://localhost:9127"):
        self.api_base_url = api_base_url
        # Resolve paths relative to script location
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        self.specs_dir = project_root / "openspec" / "specs"

        if not self.specs_dir.exists():
            raise FileNotFoundError(
                f"OpenSpec specs directory not found: {self.specs_dir}"
            )

    def scan_specs(self, capability: Optional[str] = None) -> List[Dict]:
        """
        Scan openspec/specs/ for spec.md and design.md files.

        Args:
            capability: Optional capability name to filter (e.g., "document-processing")

        Returns:
            List of spec metadata dicts with file_path, capability, spec_type, content, content_hash
        """
        specs = []

        if capability:
            # Scan single capability directory
            cap_dir = self.specs_dir / capability
            if not cap_dir.exists():
                raise FileNotFoundError(
                    f"Capability directory not found: {cap_dir}"
                )
            specs.extend(self._scan_directory(cap_dir, capability))
        else:
            # Scan all capability directories
            for cap_dir in self.specs_dir.iterdir():
                if cap_dir.is_dir() and not cap_dir.name.startswith('.'):
                    capability_name = cap_dir.name
                    specs.extend(self._scan_directory(cap_dir, capability_name))

        return specs

    def _scan_directory(self, cap_dir: Path, capability: str) -> List[Dict]:
        """
        Scan a single capability directory for spec.md and design.md.

        Args:
            cap_dir: Path to capability directory
            capability: Capability name (e.g., "document-processing")

        Returns:
            List of spec metadata dicts
        """
        specs = []

        for file_path in cap_dir.glob("*.md"):
            # Only process spec.md and design.md
            if file_path.stem not in ["spec", "design"]:
                continue

            spec_type = file_path.stem  # "spec" or "design"
            content = file_path.read_text(encoding='utf-8')
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

            specs.append({
                "file_path": str(file_path),
                "capability": capability,
                "spec_type": spec_type,
                "content": content,
                "content_hash": content_hash,
                "filename": file_path.name,
                "size": len(content.encode('utf-8'))
            })

        return specs

    def ingest_spec(self, spec_data: Dict) -> Dict:
        """
        Ingest a single spec via POST /api/documents.

        Args:
            spec_data: Spec metadata dict from scan_specs()

        Returns:
            Result dict with status_code, response, duration
        """
        start_time = time.time()

        # Create multipart/form-data
        # Note: Current API endpoint (documents.py:115-242) only accepts file + tags
        # TODO Phase 2: Add metadata fields (source_type, capability, spec_type, content_hash, version)
        files = {
            "file": (
                f"{spec_data['capability']}-{spec_data['spec_type']}.md",
                spec_data["content"],
                "text/markdown"
            )
        }

        # Encode metadata in tags for now (until Phase 2 enhancement)
        # Format: "openspec:capability_name:spec_type"
        tags = f"openspec,{spec_data['capability']},{spec_data['spec_type']}"
        data = {
            "tags": tags
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/api/v1/documents",
                files=files,
                data=data,
                timeout=120  # 2 minutes for consciousness processing
            )

            duration = time.time() - start_time

            return {
                "file_path": spec_data["file_path"],
                "status_code": response.status_code,
                "response": response.json() if response.ok else {"error": response.text},
                "duration": duration
            }

        except requests.exceptions.ConnectionError:
            return {
                "file_path": spec_data["file_path"],
                "status_code": 0,
                "response": {"error": f"Connection refused. Is the API server running at {self.api_base_url}?"},
                "duration": time.time() - start_time
            }
        except requests.exceptions.Timeout:
            return {
                "file_path": spec_data["file_path"],
                "status_code": 0,
                "response": {"error": "Request timed out after 120 seconds"},
                "duration": time.time() - start_time
            }
        except Exception as e:
            return {
                "file_path": spec_data["file_path"],
                "status_code": 0,
                "response": {"error": str(e)},
                "duration": time.time() - start_time
            }

    def ingest_all(
        self,
        capability: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Ingest all specs or specific capability.

        Args:
            capability: Optional capability name to filter
            dry_run: If True, only scan and preview without ingesting
            verbose: If True, show detailed progress

        Returns:
            List of result dicts
        """
        # Scan specs
        print(f"🔍 Scanning OpenSpec specs in {self.specs_dir}")
        specs = self.scan_specs(capability)

        if not specs:
            print("⚠️  No specs found to ingest")
            return []

        print(f"📋 Found {len(specs)} spec file(s) to ingest")

        if dry_run:
            print("\n[DRY RUN MODE - No ingestion will occur]")
            for i, spec in enumerate(specs, 1):
                print(f"  {i}. {spec['file_path']}")
                print(f"     Capability: {spec['capability']}")
                print(f"     Type: {spec['spec_type']}")
                print(f"     Size: {spec['size']:,} bytes")
                print(f"     Hash: {spec['content_hash'][:16]}...")
            return []

        # Ingest specs with progress reporting (TASK #1: Add progress reporting)
        results = []
        print(f"\n🚀 Starting ingestion ({len(specs)} files)...\n")

        for i, spec in enumerate(specs, 1):
            # Progress indicator: X/Y format
            progress = f"[{i}/{len(specs)}]"

            if verbose:
                print(f"{progress} Processing: {spec['file_path']}")
                print(f"           Capability: {spec['capability']}, Type: {spec['spec_type']}")
            else:
                print(f"{progress} Ingesting {spec['capability']}/{spec['spec_type']}.md... ", end="", flush=True)

            result = self.ingest_spec(spec)
            results.append(result)

            # Status indicator
            if result["status_code"] == 200:
                status = "✓ Success"
                if verbose:
                    print(f"           {status} ({result['duration']:.1f}s)")
                else:
                    print(f"{status} ({result['duration']:.1f}s)")
            elif result["status_code"] == 409:
                status = "⊘ Duplicate (already ingested)"
                print(status if not verbose else f"           {status}")
            elif result["status_code"] == 0:
                status = f"✗ Connection Error"
                error = result['response'].get('error', 'Unknown error')[:60]
                print(status if not verbose else f"           {status}")
                print(f"           Error: {error}")
                # Stop on connection errors (API server likely down)
                print("\n❌ Stopping ingestion due to connection error")
                break
            else:
                status = f"✗ Failed (HTTP {result['status_code']})"
                error = result['response'].get('error', 'Unknown error')[:60]
                print(status if not verbose else f"           {status}")
                if verbose:
                    print(f"           Error: {error}")

            if verbose:
                print()  # Blank line between entries

        return results

    def watch_and_ingest(
        self,
        capability: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Watch openspec/specs/ for changes and auto-ingest.

        Args:
            capability: Optional capability name to filter
            verbose: If True, show detailed progress

        Note: Requires 'watchdog' package. Install with: pip install watchdog
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            print("❌ Error: 'watchdog' package not installed")
            print("   Install with: pip install watchdog")
            return

        class SpecFileHandler(FileSystemEventHandler):
            """Handle file system events for spec files"""

            def __init__(self, ingester, capability, verbose):
                self.ingester = ingester
                self.capability = capability
                self.verbose = verbose
                self.last_processed = {}  # Track last processing time per file
                self.debounce_seconds = 2  # Wait 2 seconds before re-processing

            def on_modified(self, event):
                """Handle file modification events"""
                if event.is_directory:
                    return

                file_path = Path(event.src_path)

                # Only process .md files in specs directory
                if file_path.suffix != '.md':
                    return

                # Check if file is spec.md or design.md
                if file_path.stem not in ['spec', 'design']:
                    return

                # Filter by capability if specified
                if self.capability:
                    if self.capability not in str(file_path):
                        return

                # Debounce: prevent re-processing same file too quickly
                now = time.time()
                last_time = self.last_processed.get(str(file_path), 0)
                if now - last_time < self.debounce_seconds:
                    return

                self.last_processed[str(file_path)] = now

                # Process the changed file
                print(f"\n📝 Change detected: {file_path}")
                print(f"    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Extract capability from path
                parts = file_path.parts
                specs_index = parts.index('specs')
                capability_name = parts[specs_index + 1]
                spec_type = file_path.stem

                # Read and ingest
                try:
                    content = file_path.read_text(encoding='utf-8')
                    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

                    spec_data = {
                        "file_path": str(file_path),
                        "capability": capability_name,
                        "spec_type": spec_type,
                        "content": content,
                        "content_hash": content_hash,
                        "filename": file_path.name,
                        "size": len(content.encode('utf-8'))
                    }

                    print(f"    Ingesting {capability_name}/{spec_type}.md... ", end="", flush=True)
                    result = self.ingester.ingest_spec(spec_data)

                    if result["status_code"] == 200:
                        print(f"✓ Success ({result['duration']:.1f}s)")
                    elif result["status_code"] == 409:
                        print("⊘ Duplicate (no changes)")
                    else:
                        print(f"✗ Failed (HTTP {result['status_code']})")
                        if self.verbose:
                            print(f"    Error: {result['response'].get('error', 'Unknown')}")

                except Exception as e:
                    print(f"✗ Error reading file: {e}")

        # Setup signal handler for clean exit
        def signal_handler(sig, frame):
            print("\n\n⚠️  Watch mode interrupted by user")
            print("    Stopping file watcher...")
            observer.stop()
            observer.join()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Create observer and handler
        event_handler = SpecFileHandler(self, capability, verbose)
        observer = Observer()

        # Watch the specs directory
        watch_path = self.specs_dir / (capability if capability else "")
        if not watch_path.exists():
            print(f"❌ Error: Path does not exist: {watch_path}")
            return

        observer.schedule(event_handler, str(watch_path), recursive=True)
        observer.start()

        # Print watch info
        print("=" * 70)
        print("👀 Watch Mode Active")
        print("=" * 70)
        print(f"Monitoring: {watch_path}")
        if capability:
            print(f"Capability: {capability}")
        print("Watching for: spec.md, design.md changes")
        print("\nPress Ctrl+C to stop\n")

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⚠️  Stopping watch mode...")
            observer.stop()
            observer.join()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest OpenSpec specifications into Neo4j knowledge graph via Daedalus consciousness pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all specs
  python backend/scripts/ingest_openspec_specs.py --all

  # Ingest specific capability
  python backend/scripts/ingest_openspec_specs.py --capability document-processing

  # Preview without ingesting
  python backend/scripts/ingest_openspec_specs.py --all --dry-run

  # Verbose output
  python backend/scripts/ingest_openspec_specs.py --all --verbose

For more details, see: docs/OPENSPEC_INGESTION_EXAMPLES.md
        """
    )

    # Mutually exclusive: --all or --capability
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Ingest all capabilities"
    )
    group.add_argument(
        "--capability",
        type=str,
        help="Ingest specific capability (e.g., document-processing)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without ingesting"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for file changes and auto-ingest (requires: pip install watchdog)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:9127",
        help="API base URL (default: http://localhost:9127)"
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("OpenSpec Specification Ingestion")
    print("Dionysus 2.0 - Consciousness-Enhanced Document Processing")
    print("=" * 70)
    print()

    try:
        # Initialize ingester
        ingester = OpenSpecIngester(api_base_url=args.api_url)

        # Watch mode - different flow
        if args.watch:
            if args.dry_run:
                print("⚠️  Warning: --dry-run is ignored in watch mode")

            ingester.watch_and_ingest(
                capability=args.capability if not args.all else None,
                verbose=args.verbose
            )
            return 0  # Exit after watch mode stops

        # Run ingestion (normal mode)
        start_time = time.time()
        results = ingester.ingest_all(
            capability=args.capability if not args.all else None,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        total_duration = time.time() - start_time

        # Print summary (TASK #1: Progress reporting includes summary)
        if not args.dry_run and results:
            print("\n" + "=" * 70)
            print("📊 Ingestion Summary")
            print("=" * 70)

            success = sum(1 for r in results if r["status_code"] == 200)
            duplicates = sum(1 for r in results if r["status_code"] == 409)
            failed = sum(1 for r in results if r["status_code"] not in [200, 409])

            print(f"  ✓ Success:    {success:2d} files")
            print(f"  ⊘ Duplicates: {duplicates:2d} files")
            print(f"  ✗ Failed:     {failed:2d} files")
            print(f"  ⏱ Duration:   {total_duration:.1f}s")

            if success > 0:
                avg_duration = sum(r['duration'] for r in results if r['status_code'] == 200) / success
                print(f"  ⌀ Avg/file:   {avg_duration:.1f}s")

            print()

            # Show failed files details
            if failed > 0:
                print("Failed files:")
                for r in results:
                    if r["status_code"] not in [200, 409]:
                        print(f"  • {r['file_path']}")
                        print(f"    Error: {r['response'].get('error', 'Unknown')}")

        return 0 if not results or all(r['status_code'] in [200, 409] for r in results) else 1

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
