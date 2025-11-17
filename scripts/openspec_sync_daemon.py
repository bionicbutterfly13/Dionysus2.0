#!/usr/bin/env python3
"""
OpenSpec + Archon Background Sync Daemon

Automatically polls Archon task status and syncs to OpenSpec tasks.md
at configurable intervals.

Usage:
    python scripts/openspec_sync_daemon.py start   # Start daemon in background
    python scripts/openspec_sync_daemon.py stop    # Stop daemon
    python scripts/openspec_sync_daemon.py status  # Check if running
    python scripts/openspec_sync_daemon.py run     # Run in foreground (debug)
"""

import sys
import os
import signal
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Import the existing sync logic
from sync_openspec_archon_status import ArchonStatusPoller


class SyncDaemon:
    """Background daemon for automatic Archon → OpenSpec status sync."""

    def __init__(self, config_path: str = ".openspec.config.json"):
        self.config_path = Path(config_path)
        self.pid_file = Path(".openspec-sync-daemon.pid")
        self.log_file = Path(".openspec-sync-daemon.log")
        self.changes_dir = Path("openspec/changes")

        # Load configuration
        self.config = self._load_config()

        # Setup logging
        self._setup_logging()

        # Initialize poller
        archon_url = os.getenv("ARCHON_MCP_URL", "http://localhost:8051")
        self.poller = ArchonStatusPoller(archon_url=archon_url)

        # Track last sync times to avoid excessive syncing
        self.last_sync_times: Dict[str, float] = {}

        # Graceful shutdown flag
        self.shutdown_requested = False

    def _load_config(self) -> Dict:
        """Load configuration from .openspec.config.json."""
        if not self.config_path.exists():
            # Default config
            return {
                "archon_sync": {
                    "enabled": True,
                    "sync_interval_seconds": 30,
                    "auto_commit": True,
                    "conflict_resolution": "archon_wins",
                    "similarity_threshold": 0.85
                }
            }

        with open(self.config_path) as f:
            return json.load(f)

    def _setup_logging(self):
        """Configure logging to file."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)  # Also log to stdout in foreground mode
            ]
        )
        self.logger = logging.getLogger("openspec-sync-daemon")

    def _discover_changes(self) -> List[str]:
        """
        Discover all OpenSpec changes that have Archon integration.

        Returns list of change IDs (directory names).
        """
        if not self.changes_dir.exists():
            return []

        changes = []
        for change_dir in self.changes_dir.iterdir():
            if not change_dir.is_dir():
                continue

            # Check for .archon-project-id file
            archon_id_file = change_dir / ".archon-project-id"
            if archon_id_file.exists():
                changes.append(change_dir.name)

        return changes

    def _should_sync(self, change_id: str) -> bool:
        """
        Determine if change should be synced now.

        Avoids syncing too frequently (respects sync_interval_seconds).
        """
        sync_interval = self.config["archon_sync"].get("sync_interval_seconds", 30)
        last_sync = self.last_sync_times.get(change_id, 0)
        time_since_last = time.time() - last_sync

        return time_since_last >= sync_interval

    def _sync_change(self, change_id: str, quiet: bool = False) -> Optional[Dict]:
        """
        Sync a single change.

        Args:
            change_id: Change ID to sync
            quiet: Suppress verbose output

        Returns:
            Sync result dict or None if skipped/failed
        """
        try:
            # Redirect print output if quiet mode
            if quiet:
                # Temporarily redirect stdout to suppress verbose output
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

            result = self.poller.sync_status(change_id, dry_run=False)

            if quiet:
                sys.stdout.close()
                sys.stdout = old_stdout

            # Update last sync time
            self.last_sync_times[change_id] = time.time()

            # Log results
            if result["success"]:
                updates = result.get("updates_applied", 0)
                completion = result.get("completion", {})

                if updates > 0:
                    self.logger.info(
                        f"✓ Synced {change_id}: {updates} updates, "
                        f"{completion.get('completed', 0)}/{completion.get('total', 0)} complete"
                    )
                else:
                    self.logger.debug(f"  {change_id}: already in sync")
            else:
                self.logger.warning(f"✗ Failed to sync {change_id}: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            self.logger.error(f"✗ Error syncing {change_id}: {e}", exc_info=True)
            return None

    def _sync_all_changes(self):
        """Sync all discovered changes."""
        changes = self._discover_changes()

        if not changes:
            self.logger.debug("No changes with Archon integration found")
            return

        self.logger.info(f"Polling {len(changes)} changes: {', '.join(changes)}")

        synced_count = 0
        for change_id in changes:
            if not self._should_sync(change_id):
                self.logger.debug(f"  {change_id}: skipping (synced recently)")
                continue

            self._sync_change(change_id, quiet=True)
            synced_count += 1

        if synced_count > 0:
            self.logger.info(f"Poll complete: checked {synced_count}/{len(changes)} changes")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("Shutdown signal received, stopping daemon...")
        self.shutdown_requested = True

    def run(self):
        """Run daemon in foreground (for debugging)."""
        if not self.config["archon_sync"].get("enabled", True):
            self.logger.warning("Archon sync is disabled in config. Exiting.")
            return

        sync_interval = self.config["archon_sync"].get("sync_interval_seconds", 30)

        self.logger.info("=" * 60)
        self.logger.info("OpenSpec + Archon Sync Daemon starting")
        self.logger.info(f"Sync interval: {sync_interval}s")
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Log: {self.log_file}")
        self.logger.info("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Main loop
        iteration = 0
        while not self.shutdown_requested:
            iteration += 1
            self.logger.info(f"\n[Poll #{iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                self._sync_all_changes()
            except Exception as e:
                self.logger.error(f"Error in sync loop: {e}", exc_info=True)

            # Sleep in small increments to allow quick shutdown
            for _ in range(sync_interval):
                if self.shutdown_requested:
                    break
                time.sleep(1)

        self.logger.info("Daemon stopped gracefully")

    def start(self):
        """Start daemon in background."""
        # Check if already running
        if self.is_running():
            print(f"✗ Daemon already running (PID: {self._read_pid()})")
            return 1

        print(f"Starting OpenSpec sync daemon...")
        print(f"  Interval: {self.config['archon_sync'].get('sync_interval_seconds', 30)}s")
        print(f"  Log: {self.log_file}")

        # Fork to background
        try:
            pid = os.fork()
            if pid > 0:
                # Parent process
                print(f"✓ Daemon started (PID: {pid})")
                return 0
        except OSError as e:
            print(f"✗ Fork failed: {e}")
            return 1

        # Child process (daemon)
        # Detach from parent environment
        os.chdir("/")
        os.setsid()
        os.umask(0)

        # Second fork to prevent zombie processes
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            sys.exit(1)

        # Write PID file
        self._write_pid(os.getpid())

        # Redirect stdout/stderr to log file
        sys.stdout.flush()
        sys.stderr.flush()

        # Close stdin
        with open(os.devnull, 'r') as devnull:
            os.dup2(devnull.fileno(), sys.stdin.fileno())

        # Run daemon
        try:
            self.run()
        finally:
            self._remove_pid()

    def stop(self):
        """Stop daemon."""
        if not self.is_running():
            print("✗ Daemon not running")
            return 1

        pid = self._read_pid()
        print(f"Stopping daemon (PID: {pid})...")

        try:
            os.kill(pid, signal.SIGTERM)

            # Wait for process to exit (max 10 seconds)
            for _ in range(10):
                time.sleep(1)
                if not self.is_running():
                    print("✓ Daemon stopped")
                    return 0

            # Force kill if still running
            print("⚠ Daemon did not stop gracefully, forcing...")
            os.kill(pid, signal.SIGKILL)
            self._remove_pid()
            print("✓ Daemon killed")
            return 0

        except ProcessLookupError:
            print("✗ Process not found (cleaning up stale PID file)")
            self._remove_pid()
            return 1
        except Exception as e:
            print(f"✗ Error stopping daemon: {e}")
            return 1

    def status(self):
        """Check daemon status."""
        if self.is_running():
            pid = self._read_pid()
            print(f"✓ Daemon running (PID: {pid})")

            # Show recent log entries
            if self.log_file.exists():
                print("\nRecent activity:")
                with open(self.log_file) as f:
                    lines = f.readlines()
                    for line in lines[-5:]:
                        print(f"  {line.rstrip()}")

            return 0
        else:
            print("✗ Daemon not running")
            return 1

    def is_running(self) -> bool:
        """Check if daemon is running."""
        if not self.pid_file.exists():
            return False

        pid = self._read_pid()
        if pid is None:
            return False

        try:
            # Check if process exists (doesn't actually send signal)
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            # Process doesn't exist, clean up stale PID file
            self._remove_pid()
            return False
        except PermissionError:
            # Process exists but we can't signal it (owned by another user)
            return True

    def _write_pid(self, pid: int):
        """Write PID to file."""
        self.pid_file.write_text(str(pid))

    def _read_pid(self) -> Optional[int]:
        """Read PID from file."""
        if not self.pid_file.exists():
            return None

        try:
            return int(self.pid_file.read_text().strip())
        except (ValueError, IOError):
            return None

    def _remove_pid(self):
        """Remove PID file."""
        if self.pid_file.exists():
            self.pid_file.unlink()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenSpec + Archon background sync daemon"
    )
    parser.add_argument(
        "command",
        choices=["start", "stop", "status", "run"],
        help="Daemon command"
    )
    parser.add_argument(
        "--config",
        default=".openspec.config.json",
        help="Path to config file (default: .openspec.config.json)"
    )

    args = parser.parse_args()

    daemon = SyncDaemon(config_path=args.config)

    if args.command == "start":
        sys.exit(daemon.start())
    elif args.command == "stop":
        sys.exit(daemon.stop())
    elif args.command == "status":
        sys.exit(daemon.status())
    elif args.command == "run":
        daemon.run()
        sys.exit(0)


if __name__ == "__main__":
    main()
