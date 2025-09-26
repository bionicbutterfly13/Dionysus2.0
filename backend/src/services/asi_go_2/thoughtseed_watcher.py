"""
ThoughtSeed Watcher API - Controls watching for specific workspaces
"""
from typing import Dict, Optional
from cognitive_core.state_logger import StateLogger
from cognitive_core.thoughtseed_competition import InnerWorkspace

class ThoughtSeedWatcher:
    """Central controller for ThoughtSeed watching functionality"""

    def __init__(self):
        self.state_logger = StateLogger()
        self.watched_workspaces: Dict[str, InnerWorkspace] = {}

    def enable_watching(self, workspace: InnerWorkspace, workspace_name: str = None) -> str:
        """Enable watching for a specific workspace"""
        workspace_id = workspace_name or f"workspace_{id(workspace)}"

        # Enable watching on the workspace
        workspace.enable_watching(self.state_logger)

        # Track the workspace
        self.watched_workspaces[workspace_id] = workspace

        print(f"✅ Watching enabled for {workspace_id}")
        return workspace_id

    def disable_watching(self, workspace_id: str):
        """Disable watching for a specific workspace"""
        if workspace_id in self.watched_workspaces:
            workspace = self.watched_workspaces[workspace_id]
            workspace.disable_watching()
            del self.watched_workspaces[workspace_id]
            print(f"🔇 Watching disabled for {workspace_id}")
        else:
            print(f"⚠️ Workspace {workspace_id} not found in watched list")

    def get_watched_workspaces(self) -> list:
        """Get list of currently watched workspaces"""
        return list(self.watched_workspaces.keys())

    def get_state_logs(self, workspace_id: Optional[str] = None) -> list:
        """Get state logs for watched workspaces"""
        return self.state_logger.get_watched_states(workspace_id)

    def clear_logs(self, workspace_id: Optional[str] = None):
        """Clear state logs"""
        self.state_logger.clear_watched_states(workspace_id)
        print(f"🗑️ Cleared logs for {workspace_id or 'all workspaces'}")

# Global watcher instance
watcher = ThoughtSeedWatcher()