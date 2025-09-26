#!/usr/bin/env python
"""
Demo script to show the port conflict detection and auto-resolution system.
This shows how the system tries multiple ports before giving up.
"""
import sys
sys.path.append('backend/src')

from utils.port_manager import PortManager
import socket
import time


def demo_port_conflict_resolution():
    """Demonstrate the port conflict resolution system."""
    print("🔍 Testing Flux Port Management System")
    print("=" * 50)

    port_manager = PortManager()

    # Show preferred ports
    print("\n📋 Flux Preferred Ports:")
    for service, port in port_manager.flux_port_preferences.items():
        print(f"  {service}: {port}")

    # Test 1: Normal case (no conflicts)
    print("\n✅ Test 1: Normal allocation (no conflicts)")
    ports1 = port_manager.get_flux_ports()
    notifications1 = port_manager.get_notifications()

    for service, port in ports1.items():
        print(f"  {service}: {port} {'✅ (preferred)' if port == port_manager.flux_port_preferences[service] else '⚠️ (alternative)'}")

    if notifications1:
        print(f"  Conflicts detected: {len(notifications1)}")
        for notification in notifications1:
            print(f"    - {notification}")
    else:
        print("  No conflicts - all preferred ports available!")

    # Test 2: Create conflicts and show resolution
    print("\n⚠️ Test 2: Simulating port conflicts")

    # Occupy preferred ports to simulate conflicts
    sockets = []
    occupied_ports = []

    try:
        for service, preferred_port in port_manager.flux_port_preferences.items():
            if port_manager.is_port_available(preferred_port):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('127.0.0.1', preferred_port))
                sockets.append(sock)
                occupied_ports.append(preferred_port)
                print(f"  🔒 Occupied port {preferred_port} for {service}")

        # Now request ports - should auto-resolve conflicts
        print("\n🔄 Requesting Flux ports with conflicts...")

        port_manager_conflict = PortManager()  # Fresh instance
        ports2 = port_manager_conflict.get_flux_ports()
        notifications2 = port_manager_conflict.get_notifications()

        print(f"\n🎯 Resolution Results:")
        for service, port in ports2.items():
            preferred = port_manager_conflict.flux_port_preferences[service]
            if port == preferred:
                print(f"  {service}: {port} ✅ (got preferred port)")
            else:
                print(f"  {service}: {port} 🔄 (resolved from {preferred})")

        print(f"\n📢 Notifications ({len(notifications2)} conflicts resolved):")
        for notification in notifications2:
            print(f"  - {notification}")

    finally:
        # Clean up test sockets
        for sock in sockets:
            sock.close()
        print(f"\n🧹 Cleaned up {len(sockets)} test sockets")

    # Test 3: Show retry mechanism
    print("\n🔄 Test 3: Port retry mechanism")
    test_port = 9127
    print(f"Finding next available port starting from {test_port}...")

    available_port = port_manager.find_next_available_port(test_port)
    print(f"Next available port: {available_port}")

    # Test random port generation
    print("\n🎲 Test 4: Random port generation (high availability)")
    random_ports = [port_manager.get_random_port() for _ in range(3)]
    print(f"Generated random ports: {random_ports}")

    print("\n" + "=" * 50)
    print("✅ Port Management System Demo Complete!")
    print("The system:")
    print("  - Detects port conflicts automatically")
    print("  - Tries multiple alternative ports (up to 100 attempts)")
    print("  - Generates notifications for all conflicts")
    print("  - Uses unique odd ports to minimize conflicts")
    print("  - Provides random ports for high availability")


if __name__ == "__main__":
    demo_port_conflict_resolution()