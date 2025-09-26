"""
Test port conflict detection and auto-resolution system.
This test ensures we can automatically handle port conflicts.
"""
import pytest
import socket
from backend.src.utils.port_manager import PortManager


class TestPortManager:
    """Test suite for port conflict detection and resolution."""

    def test_port_is_available_when_free(self):
        """Test that a free port is detected as available."""
        # Use an unlikely port that should be free
        test_port = 59876
        port_manager = PortManager()

        assert port_manager.is_port_available(test_port) is True

    def test_port_is_unavailable_when_occupied(self):
        """Test that an occupied port is detected as unavailable."""
        port_manager = PortManager()

        # Create a socket to occupy a port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))  # Let OS choose port
        occupied_port = sock.getsockname()[1]

        try:
            assert port_manager.is_port_available(occupied_port) is False
        finally:
            sock.close()

    def test_find_next_available_port(self):
        """Test finding the next available port from a starting point."""
        port_manager = PortManager()

        # Start from Flux's preferred port
        available_port = port_manager.find_next_available_port(9100)

        # Should return a port >= 9100
        assert available_port >= 9100
        assert port_manager.is_port_available(available_port) is True

    def test_get_flux_ports_with_conflicts(self):
        """Test getting Flux ports with automatic conflict resolution."""
        port_manager = PortManager()

        # Get Flux ports (should handle conflicts automatically)
        ports = port_manager.get_flux_ports()

        # Should return all required ports
        expected_ports = ['backend_api', 'frontend_dev', 'websocket_stream', 'health_monitor']
        for port_name in expected_ports:
            assert port_name in ports
            assert isinstance(ports[port_name], int)
            assert ports[port_name] > 1024  # Should be user ports
            assert port_manager.is_port_available(ports[port_name]) is True

    def test_port_conflict_notification(self):
        """Test that port conflicts generate notifications."""
        port_manager = PortManager()

        # First check if port 9127 is available, if not, skip this test gracefully
        if not port_manager.is_port_available(9127):
            # Port already occupied - request ports and verify we get alternatives
            ports = port_manager.get_flux_ports()
            notifications = port_manager.get_notifications()

            # Should have generated notifications for occupied ports
            assert len(notifications) > 0
            assert any('conflict' in notification.lower() for notification in notifications)
            return

        # If 9127 is available, occupy it temporarily for testing
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', 9127))  # Occupy Flux's preferred port

            # Request Flux ports - should detect conflict and notify
            ports = port_manager.get_flux_ports()

            # Should have generated a conflict notification
            notifications = port_manager.get_notifications()
            assert len(notifications) > 0
            assert any('conflict' in notification.lower() for notification in notifications)

            # Should have allocated a different port
            assert ports['backend_api'] != 9127

        finally:
            sock.close()

    def test_random_port_generation(self):
        """Test random port generation for high availability."""
        port_manager = PortManager()

        # Generate multiple random ports
        ports = [port_manager.get_random_port() for _ in range(10)]

        # All should be different
        assert len(set(ports)) == len(ports)

        # All should be in valid range
        for port in ports:
            assert 1024 <= port <= 65535
            assert port_manager.is_port_available(port) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])