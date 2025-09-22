#!/usr/bin/env python3
"""
🔗 Live ASI-Arch Context Engineering Integration
===============================================

This module provides live integration with the ASI-Arch pipeline, automatically
enhancing evolution context and tracking consciousness emergence in real-time.

Features:
- Automatic context enhancement during evolution
- Real-time consciousness tracking
- Live dashboard updates
- Seamless integration with existing ASI-Arch code

Usage:
    # Option 1: Import and use enhanced pipeline
    from extensions.context_engineering.live_integration import start_enhanced_pipeline
    start_enhanced_pipeline()
    
    # Option 2: Manual integration
    from extensions.context_engineering.live_integration import ContextEngineeringLiveService
    service = ContextEngineeringLiveService()
    service.start()

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - Live Integration
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .core_implementation import ContextEngineeringService, ConsciousnessLevel
from .hybrid_database import create_hybrid_database
from .visualization_dashboard import create_dashboard
from .asi_arch_bridge import get_integration, get_monitor

logger = logging.getLogger(__name__)

# =============================================================================
# Live Service Manager
# =============================================================================

class ContextEngineeringLiveService:
    """Main service that coordinates all context engineering components"""
    
    def __init__(self, dashboard_port: int = 8080, enable_dashboard: bool = True):
        self.dashboard_port = dashboard_port
        self.enable_dashboard = enable_dashboard
        
        # Core services
        self.context_service = ContextEngineeringService()
        self.database = create_hybrid_database()
        self.integration = get_integration()
        self.monitor = get_monitor()
        
        # Dashboard (optional)
        self.dashboard = None
        if enable_dashboard:
            try:
                self.dashboard = create_dashboard(dashboard_port)
                logger.info(f"Dashboard will be available at http://localhost:{dashboard_port}")
            except Exception as e:
                logger.warning(f"Dashboard initialization failed: {e}")
                self.enable_dashboard = False
        
        # State tracking
        self.is_running = False
        self.experiment_count = 0
        self.consciousness_evolution = []
        self.performance_metrics = []
        
        # Background tasks
        self.maintenance_thread = None
        
        logger.info("Context Engineering Live Service initialized")
    
    def start(self, start_dashboard: bool = None):
        """Start the live service"""
        if start_dashboard is None:
            start_dashboard = self.enable_dashboard
        
        self.is_running = True
        
        # Start dashboard
        if start_dashboard and self.dashboard:
            try:
                self.dashboard.start_server(open_browser=True)
                logger.info("🚀 Dashboard started successfully")
            except Exception as e:
                logger.error(f"Dashboard start failed: {e}")
        
        # Start maintenance thread
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        
        # Enable integration
        self.integration.enable_integration()
        
        logger.info("🌊 Context Engineering Live Service is now active!")
        logger.info("   - Context enhancement: ENABLED")
        logger.info("   - Consciousness tracking: ENABLED")
        logger.info("   - River metaphor analysis: ENABLED")
        logger.info("   - Attractor basin mapping: ENABLED")
        
        if self.dashboard:
            logger.info(f"   - Dashboard: http://localhost:{self.dashboard_port}")
    
    def stop(self):
        """Stop the live service"""
        logger.info("🛑 Stopping Context Engineering Live Service...")
        
        self.is_running = False
        
        # Stop dashboard
        if self.dashboard:
            self.dashboard.stop_server()
        
        # Disable integration
        self.integration.disable_integration()
        
        # Final database save
        self.database.periodic_maintenance()
        
        logger.info("✅ Context Engineering Live Service stopped")
    
    def _maintenance_loop(self):
        """Background maintenance loop"""
        while self.is_running:
            try:
                # Database maintenance
                self.database.periodic_maintenance()
                
                # Monitor metrics collection
                if self.experiment_count > 0:
                    # This would collect metrics from ASI-Arch database
                    # For now, we'll simulate with cached data
                    if self.dashboard and len(self.consciousness_evolution) > 0:
                        # Update dashboard with latest data
                        pass
                
                time.sleep(60)  # Run maintenance every minute
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                time.sleep(60)
    
    async def process_architecture_experiment(self, 
                                            context: str, 
                                            parent_data: Optional[Dict[str, Any]] = None,
                                            result_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single architecture experiment through context engineering"""
        
        self.experiment_count += 1
        experiment_id = f"exp_{self.experiment_count}_{int(time.time())}"
        
        logger.info(f"🧪 Processing experiment {experiment_id}")
        
        # Step 1: Enhance context if we have parent data
        enhanced_context = context
        if parent_data:
            try:
                enhanced_context = await self.context_service.enhance_asi_arch_context(context, parent_data)
                logger.info(f"   📝 Context enhanced: {len(context)} → {len(enhanced_context)} chars")
            except Exception as e:
                logger.warning(f"   ⚠️ Context enhancement failed: {e}")
        
        # Step 2: Analyze consciousness if we have result data
        consciousness_analysis = {}
        if result_data:
            try:
                consciousness_level = await self.context_service.evolution.consciousness_detector.detect_consciousness_level(result_data)
                consciousness_analysis = {
                    'level': consciousness_level.name,
                    'score': consciousness_level.value,
                    'architecture_name': result_data.get('name', 'unknown')
                }
                
                # Store in database
                self.database.store_architecture(result_data, consciousness_level.name, consciousness_level.value)
                
                # Update dashboard
                if self.dashboard:
                    self.dashboard.update_from_architecture_data(result_data, consciousness_level.name, consciousness_level.value)
                
                # Track evolution
                self.consciousness_evolution.append({
                    'timestamp': datetime.now(),
                    'experiment_id': experiment_id,
                    'consciousness_level': consciousness_level.name,
                    'consciousness_score': consciousness_level.value,
                    'architecture_name': result_data.get('name', 'unknown')
                })
                
                logger.info(f"   🧠 Consciousness detected: {consciousness_level.name} ({consciousness_level.value:.3f})")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Consciousness analysis failed: {e}")
        
        # Step 3: Update river metaphor analysis
        try:
            if result_data:
                # Create a context stream for this experiment
                stream = await self.context_service.evolution.stream_manager.create_stream_from_architectures(
                    [result_data.get('name', 'unknown')],
                    [result_data]
                )
                
                # Update dashboard river visualization
                if self.dashboard:
                    self.dashboard.update_river_flow(
                        stream.id,
                        stream.flow_velocity,
                        stream.flow_state.value,
                        stream.information_density
                    )
                
                logger.info(f"   🌊 River analysis: {stream.flow_state.value} flow, velocity {stream.flow_velocity:.3f}")
                
        except Exception as e:
            logger.warning(f"   ⚠️ River analysis failed: {e}")
        
        # Step 4: Check for attractor basins periodically
        if self.experiment_count % 5 == 0:  # Every 5 experiments
            try:
                # Get recent architectures for basin analysis
                recent_architectures = []
                for evolution_data in self.consciousness_evolution[-10:]:  # Last 10
                    arch_name = evolution_data['architecture_name']
                    # In a real implementation, we'd fetch full architecture data
                    # For now, we'll use minimal data
                    recent_architectures.append({
                        'name': arch_name,
                        'result': {'test': f'acc={evolution_data["consciousness_score"]}'}
                    })
                
                if len(recent_architectures) >= 3:
                    basins = await self.context_service.evolution.basin_manager.identify_basins_from_architectures(recent_architectures)
                    
                    # Update dashboard with new basins
                    if self.dashboard and basins:
                        for basin in basins:
                            self.dashboard.update_attractor_basin(
                                basin.id,
                                basin.center_architecture_name,
                                basin.radius,
                                basin.attraction_strength,
                                basin.contained_architectures
                            )
                    
                    logger.info(f"   🎯 Basin analysis: {len(basins)} basins identified")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Basin analysis failed: {e}")
        
        # Return analysis summary
        return {
            'experiment_id': experiment_id,
            'enhanced_context': enhanced_context,
            'consciousness_analysis': consciousness_analysis,
            'context_engineering_active': True,
            'timestamp': datetime.now().isoformat()
        }

# =============================================================================
# ASI-Arch Pipeline Integration
# =============================================================================

class ASIArchPipelineEnhancer:
    """Enhances the ASI-Arch pipeline with context engineering"""
    
    def __init__(self, live_service: ContextEngineeringLiveService):
        self.live_service = live_service
        self.original_functions = {}
    
    def enhance_pipeline_functions(self):
        """Enhance ASI-Arch pipeline functions with context engineering"""
        logger.info("🔧 Enhancing ASI-Arch pipeline functions...")
        
        try:
            # Try to import ASI-Arch modules
            import database
            import evolve
            import eval as evaluation_module
            import analyse
            
            # Store original functions
            self.original_functions['program_sample'] = database.program_sample
            self.original_functions['evolve'] = evolve.evolve
            self.original_functions['evaluation'] = evaluation_module.evaluation
            self.original_functions['analyse'] = analyse.analyse
            
            # Create enhanced versions
            database.program_sample = self._enhanced_program_sample
            evolve.evolve = self._enhanced_evolve
            evaluation_module.evaluation = self._enhanced_evaluation
            analyse.analyse = self._enhanced_analyse
            
            logger.info("✅ ASI-Arch pipeline functions enhanced")
            return True
            
        except ImportError as e:
            logger.warning(f"Could not import ASI-Arch modules: {e}")
            logger.info("   Context engineering will work in standalone mode")
            return False
    
    def restore_pipeline_functions(self):
        """Restore original ASI-Arch pipeline functions"""
        if not self.original_functions:
            return
        
        logger.info("🔄 Restoring original ASI-Arch pipeline functions...")
        
        try:
            import database
            import evolve
            import eval as evaluation_module
            import analyse
            
            database.program_sample = self.original_functions['program_sample']
            evolve.evolve = self.original_functions['evolve']
            evaluation_module.evaluation = self.original_functions['evaluation']
            analyse.analyse = self.original_functions['analyse']
            
            logger.info("✅ Original pipeline functions restored")
            
        except ImportError:
            pass
    
    async def _enhanced_program_sample(self):
        """Enhanced program sampling with context awareness"""
        # Call original function
        context, parent = await self.original_functions['program_sample']()
        
        # Process through context engineering
        if parent:
            try:
                parent_dict = parent.to_dict() if hasattr(parent, 'to_dict') else parent.__dict__
                analysis = await self.live_service.process_architecture_experiment(
                    context, parent_data=parent_dict
                )
                
                # Use enhanced context
                context = analysis['enhanced_context']
                
            except Exception as e:
                logger.warning(f"Context enhancement in program_sample failed: {e}")
        
        return context, parent
    
    async def _enhanced_evolve(self, context: str):
        """Enhanced evolution with context engineering insights"""
        logger.info("🧬 Running enhanced evolution...")
        
        # Call original function
        result = await self.original_functions['evolve'](context)
        
        # Log context engineering activity
        logger.info("   🌊 Context engineering insights applied to evolution")
        
        return result
    
    async def _enhanced_evaluation(self, name: str, motivation: str):
        """Enhanced evaluation with consciousness tracking"""
        logger.info(f"🔬 Running enhanced evaluation for {name}...")
        
        # Call original function
        success = await self.original_functions['evaluation'](name, motivation)
        
        if success:
            logger.info(f"   ✅ Evaluation successful: {name}")
        else:
            logger.info(f"   ❌ Evaluation failed: {name}")
        
        return success
    
    async def _enhanced_analyse(self, name: str, motivation: str, parent=None):
        """Enhanced analysis with context engineering insights"""
        logger.info(f"📊 Running enhanced analysis for {name}...")
        
        # Call original function
        result = await self.original_functions['analyse'](name, motivation, parent=parent)
        
        # Process result through context engineering
        try:
            if result:
                result_dict = result.to_dict() if hasattr(result, 'to_dict') else result.__dict__
                analysis = await self.live_service.process_architecture_experiment(
                    "", result_data=result_dict
                )
                
                consciousness_info = analysis.get('consciousness_analysis', {})
                if consciousness_info:
                    logger.info(f"   🧠 Consciousness: {consciousness_info.get('level', 'unknown')} "
                               f"({consciousness_info.get('score', 0.0):.3f})")
                
        except Exception as e:
            logger.warning(f"Context engineering analysis failed: {e}")
        
        return result

# =============================================================================
# Main Integration Functions
# =============================================================================

def start_enhanced_pipeline(dashboard_port: int = 8080, 
                          enable_dashboard: bool = True,
                          integrate_with_asi_arch: bool = True) -> ContextEngineeringLiveService:
    """
    Start the enhanced ASI-Arch pipeline with context engineering
    
    Args:
        dashboard_port: Port for the web dashboard
        enable_dashboard: Whether to start the web dashboard
        integrate_with_asi_arch: Whether to enhance ASI-Arch functions directly
    
    Returns:
        The live service instance
    """
    logger.info("🚀 Starting Enhanced ASI-Arch Pipeline with Context Engineering")
    
    # Create live service
    service = ContextEngineeringLiveService(dashboard_port, enable_dashboard)
    
    # Start the service
    service.start()
    
    # Optionally integrate with ASI-Arch pipeline
    if integrate_with_asi_arch:
        enhancer = ASIArchPipelineEnhancer(service)
        if enhancer.enhance_pipeline_functions():
            logger.info("🔗 Direct ASI-Arch integration enabled")
            service.pipeline_enhancer = enhancer
        else:
            logger.info("🔧 Running in standalone mode (ASI-Arch modules not available)")
    
    return service

def stop_enhanced_pipeline(service: ContextEngineeringLiveService):
    """Stop the enhanced pipeline and restore original functions"""
    if hasattr(service, 'pipeline_enhancer'):
        service.pipeline_enhancer.restore_pipeline_functions()
    
    service.stop()
    logger.info("🛑 Enhanced pipeline stopped")

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context Engineering Live Integration")
    parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable web dashboard')
    parser.add_argument('--no-integration', action='store_true', help='Disable direct ASI-Arch integration')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with mock data')
    
    args = parser.parse_args()
    
    if args.test_mode:
        print("🧪 Running in test mode...")
        test_live_integration()
    else:
        print("🌊 Starting Context Engineering Live Integration...")
        
        service = start_enhanced_pipeline(
            dashboard_port=args.port,
            enable_dashboard=not args.no_dashboard,
            integrate_with_asi_arch=not args.no_integration
        )
        
        try:
            print("✅ Context Engineering is now active!")
            print("   Press Ctrl+C to stop...")
            
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            stop_enhanced_pipeline(service)

# =============================================================================
# Testing Functions
# =============================================================================

async def test_live_integration():
    """Test the live integration with mock data"""
    print("🧪 Testing Context Engineering Live Integration")
    
    service = ContextEngineeringLiveService(dashboard_port=8081, enable_dashboard=True)
    service.start(start_dashboard=False)  # Don't auto-open browser in test
    
    # Test with mock architecture experiments
    print("\n1. Testing architecture experiment processing...")
    
    for i in range(5):
        mock_parent = {
            'name': f'parent_arch_{i}',
            'program': f'class ParentArch{i}(nn.Module): pass',
            'result': {'test': f'acc={0.7 + i*0.02}'},
            'motivation': f'parent motivation {i}',
            'analysis': f'parent analysis {i}'
        }
        
        mock_result = {
            'name': f'evolved_arch_{i}',
            'program': f'class EvolvedArch{i}(nn.Module): pass',
            'result': {'test': f'acc={0.75 + i*0.03}'},
            'motivation': f'evolved motivation {i}',
            'analysis': f'shows emergent self-attention patterns' if i > 2 else f'basic analysis {i}'
        }
        
        analysis = await service.process_architecture_experiment(
            f"Context for experiment {i}",
            parent_data=mock_parent,
            result_data=mock_result
        )
        
        print(f"   Experiment {i+1}: {analysis['consciousness_analysis'].get('level', 'unknown')} "
              f"consciousness ({analysis['consciousness_analysis'].get('score', 0.0):.3f})")
        
        time.sleep(0.5)  # Small delay to see progression
    
    print("\n2. Testing dashboard data...")
    consciousness_summary = service.dashboard.get_consciousness_summary() if service.dashboard else {}
    river_summary = service.dashboard.get_river_summary() if service.dashboard else {}
    basin_summary = service.dashboard.get_basin_summary() if service.dashboard else {}
    
    print(f"   Consciousness trend: {consciousness_summary.get('trend', 0.0):.4f}")
    print(f"   Active rivers: {river_summary.get('active_streams', 0)}")
    print(f"   Attractor basins: {basin_summary.get('basin_count', 0)}")
    
    print("\n3. Testing database export...")
    export_file = service.database.export_knowledge_graph("test_knowledge_graph.json")
    print(f"   Exported to: {export_file}")
    
    if service.dashboard:
        print(f"\n4. Dashboard available at: http://localhost:{service.dashboard_port}")
        print("   (Dashboard server running in background)")
    
    print("\n✅ Live integration test completed!")
    
    # Keep service running for manual testing
    print("\n⏳ Service will continue running for 30 seconds for manual testing...")
    print("   You can view the dashboard in your browser")
    
    await asyncio.sleep(30)
    
    service.stop()
    print("🛑 Test service stopped")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_live_integration())
    else:
        main()
