"""
StateLogger for ThoughtSeed watching - Redis backend with TTL
"""
import json
import redis
from typing import Optional, Dict, Any
from datetime import datetime

class StateLogger:
    """Logs ThoughtSeed competition state to Redis with 10-minute TTL"""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.ttl_seconds = 600  # 10 minutes as specified

    def log_competition_cycle(self, state: Dict[str, Any], phase: str):
        """Log a competition cycle state"""
        log_entry = {
            "phase": phase,
            "state": state,
            "logged_at": self._get_timestamp()
        }

        # Create unique key for this log entry
        workspace_id = state.get("workspace_id", "unknown")
        timestamp = state.get("timestamp", "unknown")
        key = f"thoughtseed:watch:{workspace_id}:{timestamp}:{phase}"

        # Store in Redis with TTL
        try:
            self.redis_client.setex(
                key,
                self.ttl_seconds,
                json.dumps(log_entry, indent=2)
            )
        except Exception as e:
            print(f"⚠️ Failed to log state: {e}")

    def get_watched_states(self, workspace_id: Optional[str] = None) -> list:
        """Get all watched states, optionally filtered by workspace"""
        try:
            pattern = f"thoughtseed:watch:{workspace_id or '*'}:*"
            keys = self.redis_client.keys(pattern)

            states = []
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    states.append(json.loads(data))

            # Sort by timestamp
            states.sort(key=lambda x: x.get("logged_at", ""))
            return states

        except Exception as e:
            print(f"⚠️ Failed to retrieve states: {e}")
            return []

    def clear_watched_states(self, workspace_id: Optional[str] = None):
        """Clear watched states, optionally filtered by workspace"""
        try:
            pattern = f"thoughtseed:watch:{workspace_id or '*'}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            print(f"⚠️ Failed to clear states: {e}")

    def _get_timestamp(self) -> str:
        """Get ISO8601 timestamp with milliseconds"""
        return datetime.now().isoformat(timespec='milliseconds') + 'Z'