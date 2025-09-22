#!/usr/bin/env python3
"""
🔍 ASI-Arch System Monitor
==========================

Continuous monitoring system for ASI-Arch with Context Engineering integration.
Monitors:
- Python environment and dependencies
- Docker services (MongoDB, OpenSearch)
- API endpoints (Database API, RAG API)
- Pipeline components
- Integration health

Usage:
    python monitor_system.py --continuous
    python monitor_system.py --check-once
"""

import asyncio
import subprocess
import sys
import time
import requests
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "✅ HEALTHY"
    WARNING = "⚠️  WARNING"
    ERROR = "❌ ERROR"
    UNKNOWN = "❓ UNKNOWN"

@dataclass
class ServiceStatus:
    name: str
    status: HealthStatus
    details: str
    last_checked: datetime
    response_time: Optional[float] = None

class ASIArchMonitor:
    """Comprehensive system monitor for ASI-Arch"""
    
    def __init__(self):
        self.services = {}
        self.check_interval = 30  # seconds
        self.running = False
        
    def check_python_environment(self) -> ServiceStatus:
        """Check Python environment and key packages"""
        try:
            # Check if we're in virtual environment
            venv_active = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )
            
            if not venv_active:
                return ServiceStatus(
                    name="Python Environment",
                    status=HealthStatus.WARNING,
                    details="Virtual environment not active",
                    last_checked=datetime.now()
                )
            
            # Check key packages
            required_packages = [
                'torch', 'tensorflow', 'transformers', 'numpy', 
                'pymongo', 'fastapi', 'opensearch-py'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                return ServiceStatus(
                    name="Python Environment",
                    status=HealthStatus.WARNING,
                    details=f"Missing packages: {', '.join(missing_packages)}",
                    last_checked=datetime.now()
                )
            
            return ServiceStatus(
                name="Python Environment",
                status=HealthStatus.HEALTHY,
                details=f"Virtual env active, all packages available",
                last_checked=datetime.now()
            )
            
        except Exception as e:
            return ServiceStatus(
                name="Python Environment",
                status=HealthStatus.ERROR,
                details=f"Error checking environment: {e}",
                last_checked=datetime.now()
            )
    
    def check_docker_services(self) -> List[ServiceStatus]:
        """Check Docker containers status"""
        services = []
        
        try:
            # Check if Docker is running
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            if result.returncode != 0:
                services.append(ServiceStatus(
                    name="Docker",
                    status=HealthStatus.ERROR,
                    details="Docker not running or not accessible",
                    last_checked=datetime.now()
                ))
                return services
            
            # Check specific containers
            containers_to_check = ['mongodb', 'opensearch-rag']
            
            for container in containers_to_check:
                container_result = subprocess.run(
                    ['docker', 'ps', '--filter', f'name={container}', '--format', '{{.Status}}'],
                    capture_output=True, text=True
                )
                
                if container_result.stdout.strip():
                    if 'Up' in container_result.stdout:
                        services.append(ServiceStatus(
                            name=f"Docker: {container}",
                            status=HealthStatus.HEALTHY,
                            details=container_result.stdout.strip(),
                            last_checked=datetime.now()
                        ))
                    else:
                        services.append(ServiceStatus(
                            name=f"Docker: {container}",
                            status=HealthStatus.WARNING,
                            details=container_result.stdout.strip(),
                            last_checked=datetime.now()
                        ))
                else:
                    services.append(ServiceStatus(
                        name=f"Docker: {container}",
                        status=HealthStatus.ERROR,
                        details="Container not found",
                        last_checked=datetime.now()
                    ))
                    
        except Exception as e:
            services.append(ServiceStatus(
                name="Docker",
                status=HealthStatus.ERROR,
                details=f"Error checking Docker: {e}",
                last_checked=datetime.now()
            ))
            
        return services
    
    def check_api_endpoints(self) -> List[ServiceStatus]:
        """Check API endpoint health"""
        endpoints = [
            ("Database API", "http://localhost:8001/health"),
            ("RAG API", "http://localhost:5000/health"),
        ]
        
        services = []
        
        for name, url in endpoints:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=5)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    services.append(ServiceStatus(
                        name=name,
                        status=HealthStatus.HEALTHY,
                        details=f"Responding (HTTP {response.status_code})",
                        last_checked=datetime.now(),
                        response_time=response_time
                    ))
                else:
                    services.append(ServiceStatus(
                        name=name,
                        status=HealthStatus.WARNING,
                        details=f"HTTP {response.status_code}",
                        last_checked=datetime.now(),
                        response_time=response_time
                    ))
                    
            except requests.exceptions.ConnectionError:
                services.append(ServiceStatus(
                    name=name,
                    status=HealthStatus.ERROR,
                    details="Connection refused",
                    last_checked=datetime.now()
                ))
            except requests.exceptions.Timeout:
                services.append(ServiceStatus(
                    name=name,
                    status=HealthStatus.WARNING,
                    details="Request timeout",
                    last_checked=datetime.now()
                ))
            except Exception as e:
                services.append(ServiceStatus(
                    name=name,
                    status=HealthStatus.ERROR,
                    details=f"Error: {e}",
                    last_checked=datetime.now()
                ))
                
        return services
    
    def check_system_resources(self) -> ServiceStatus:
        """Check system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY
            details = []
            
            if cpu_percent > 80:
                status = HealthStatus.WARNING
                details.append(f"High CPU: {cpu_percent:.1f}%")
            else:
                details.append(f"CPU: {cpu_percent:.1f}%")
                
            if memory.percent > 85:
                status = HealthStatus.WARNING
                details.append(f"High Memory: {memory.percent:.1f}%")
            else:
                details.append(f"Memory: {memory.percent:.1f}%")
                
            if disk.percent > 90:
                status = HealthStatus.WARNING
                details.append(f"High Disk: {disk.percent:.1f}%")
            else:
                details.append(f"Disk: {disk.percent:.1f}%")
            
            return ServiceStatus(
                name="System Resources",
                status=status,
                details=" | ".join(details),
                last_checked=datetime.now()
            )
            
        except Exception as e:
            return ServiceStatus(
                name="System Resources",
                status=HealthStatus.ERROR,
                details=f"Error checking resources: {e}",
                last_checked=datetime.now()
            )
    
    def run_health_check(self) -> Dict[str, ServiceStatus]:
        """Run comprehensive health check"""
        logger.info("🔍 Running health check...")
        
        all_services = {}
        
        # Check Python environment
        env_status = self.check_python_environment()
        all_services[env_status.name] = env_status
        
        # Check Docker services
        docker_services = self.check_docker_services()
        for service in docker_services:
            all_services[service.name] = service
            
        # Check API endpoints
        api_services = self.check_api_endpoints()
        for service in api_services:
            all_services[service.name] = service
            
        # Check system resources
        resource_status = self.check_system_resources()
        all_services[resource_status.name] = resource_status
        
        return all_services
    
    def print_status_report(self, services: Dict[str, ServiceStatus]):
        """Print formatted status report"""
        print("\n" + "="*80)
        print(f"🚀 ASI-Arch System Status Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        healthy_count = 0
        warning_count = 0
        error_count = 0
        
        for service_name, service in services.items():
            print(f"{service.status.value:12} {service_name:25} {service.details}")
            
            if service.response_time:
                print(f"             {'':25} Response: {service.response_time:.3f}s")
                
            if service.status == HealthStatus.HEALTHY:
                healthy_count += 1
            elif service.status == HealthStatus.WARNING:
                warning_count += 1
            elif service.status == HealthStatus.ERROR:
                error_count += 1
        
        print("-"*80)
        print(f"📊 Summary: {healthy_count} healthy, {warning_count} warnings, {error_count} errors")
        print("="*80)
        
        # Save status to file for external monitoring
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "healthy": healthy_count,
                "warnings": warning_count,
                "errors": error_count
            },
            "services": {
                name: {
                    "status": service.status.name,
                    "details": service.details,
                    "response_time": service.response_time,
                    "last_checked": service.last_checked.isoformat()
                }
                for name, service in services.items()
            }
        }
        
        with open('system_status.json', 'w') as f:
            json.dump(status_data, f, indent=2)
    
    async def continuous_monitor(self):
        """Run continuous monitoring"""
        print("🔄 Starting continuous monitoring...")
        print(f"📊 Checking every {self.check_interval} seconds")
        print("🛑 Press Ctrl+C to stop")
        
        self.running = True
        
        try:
            while self.running:
                services = self.run_health_check()
                self.print_status_report(services)
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            self.running = False
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {e}")
            self.running = False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ASI-Arch System Monitor')
    parser.add_argument('--continuous', action='store_true', 
                       help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=30,
                       help='Check interval in seconds (default: 30)')
    parser.add_argument('--check-once', action='store_true',
                       help='Run single health check')
    
    args = parser.parse_args()
    
    monitor = ASIArchMonitor()
    
    if args.interval:
        monitor.check_interval = args.interval
    
    if args.continuous:
        asyncio.run(monitor.continuous_monitor())
    else:
        # Single check
        services = monitor.run_health_check()
        monitor.print_status_report(services)

if __name__ == "__main__":
    main()
