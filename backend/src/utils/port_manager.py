"""
Port Management System for Flux
Handles port conflict detection, auto-resolution, and notification system.
"""
import socket
import random
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PortConflictNotification:
    """Notification for port conflicts."""
    timestamp: datetime
    requested_port: int
    allocated_port: int
    service_name: str
    message: str


class PortManager:
    """
    Manages port allocation with automatic conflict resolution.

    Features:
    - Port availability detection
    - Automatic conflict resolution
    - Random port generation for high availability
    - Notification system for conflicts
    - Flux-specific port configuration
    """

    def __init__(self):
        self.notifications: List[PortConflictNotification] = []
        self.flux_port_preferences = {
            'backend_api': 9127,      # Unique odd port less likely to conflict
            'frontend_dev': 9243,     # Another unique odd port
            'websocket_stream': 9129, # Sequential odd from backend
            'health_monitor': 9131    # Sequential odd from websocket
        }

    def is_port_available(self, port: int, host: str = '127.0.0.1') -> bool:
        """
        Check if a port is available for binding.

        Args:
            port: Port number to check
            host: Host address to check (default localhost)

        Returns:
            True if port is available, False if occupied
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                return True  # Successfully bound = port is available
        except PermissionError as e:
            # Some sandboxed environments disallow bind checks; treat as available.
            logger.warning(f"Permission denied checking port {port}: {e}. Assuming available.")
            return True
        except OSError:
            return False  # Failed to bind = port is occupied
        except Exception as e:
            logger.warning(f"Error checking port {port}: {e}")
            return False

    def find_next_available_port(self, start_port: int, max_attempts: int = 100) -> int:
        """
        Find the next available port starting from a given port.

        Args:
            start_port: Starting port number
            max_attempts: Maximum ports to check

        Returns:
            Next available port number

        Raises:
            RuntimeError: If no available port found within max_attempts
        """
        for port in range(start_port, start_port + max_attempts):
            if self.is_port_available(port):
                return port

        raise RuntimeError(f"No available port found starting from {start_port}")

    def get_random_port(self, min_port: int = 49152, max_port: int = 65535) -> int:
        """
        Generate a random available port in the ephemeral range.

        Args:
            min_port: Minimum port number (default ephemeral range start)
            max_port: Maximum port number (default ephemeral range end)

        Returns:
            Random available port number
        """
        max_attempts = 50
        for _ in range(max_attempts):
            port = random.randint(min_port, max_port)
            if self.is_port_available(port):
                return port

        # Fallback to sequential search if random fails
        return self.find_next_available_port(min_port)

    def get_flux_ports(self) -> Dict[str, int]:
        """
        Get all Flux service ports with automatic conflict resolution.

        Returns:
            Dictionary mapping service names to allocated port numbers
        """
        allocated_ports = {}

        for service_name, preferred_port in self.flux_port_preferences.items():
            if self.is_port_available(preferred_port):
                allocated_ports[service_name] = preferred_port
                logger.info(f"Allocated preferred port {preferred_port} for {service_name}")
            else:
                # Port conflict - find alternative
                alternative_port = self.find_next_available_port(preferred_port + 1)
                allocated_ports[service_name] = alternative_port

                # Create notification
                notification = PortConflictNotification(
                    timestamp=datetime.now(),
                    requested_port=preferred_port,
                    allocated_port=alternative_port,
                    service_name=service_name,
                    message=f"Port conflict: {service_name} moved from {preferred_port} to {alternative_port}"
                )
                self.notifications.append(notification)

                logger.warning(f"Port conflict for {service_name}: "
                             f"preferred {preferred_port} occupied, using {alternative_port}")

        return allocated_ports

    def get_notifications(self) -> List[str]:
        """
        Get list of port conflict notification messages.

        Returns:
            List of notification message strings
        """
        return [notification.message for notification in self.notifications]

    def clear_notifications(self) -> None:
        """Clear all port conflict notifications."""
        self.notifications.clear()

    def get_port_status(self) -> Dict[str, any]:
        """
        Get comprehensive port status for all Flux services.

        Returns:
            Dictionary with port allocation status and conflicts
        """
        ports = self.get_flux_ports()
        status = {
            'allocated_ports': ports,
            'conflicts_detected': len(self.notifications),
            'notifications': self.get_notifications(),
            'all_ports_available': len(self.notifications) == 0
        }

        return status


# Global instance for application use
port_manager = PortManager()


def get_flux_backend_port() -> int:
    """Get the allocated backend API port for Flux."""
    ports = port_manager.get_flux_ports()
    return ports['backend_api']


def get_flux_frontend_port() -> int:
    """Get the allocated frontend dev port for Flux."""
    ports = port_manager.get_flux_ports()
    return ports['frontend_dev']


def check_port_conflicts() -> Dict[str, any]:
    """Quick function to check for any port conflicts."""
    return port_manager.get_port_status()
