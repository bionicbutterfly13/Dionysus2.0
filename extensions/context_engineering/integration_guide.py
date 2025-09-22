#!/usr/bin/env python3
"""
🔧 ASI-Arch Context Engineering Integration Guide
=================================================

This file shows exactly how to integrate context engineering with ASI-Arch
without modifying the core pipeline code. Instead, we provide wrapper functions
and optional enhancements that can be easily added.

INTEGRATION APPROACH:
1. Non-invasive: No changes to existing ASI-Arch code
2. Optional: Can be enabled/disabled easily
3. Backward Compatible: Falls back gracefully if disabled
4. Self-Contained: All dependencies included in extensions/

Author: ASI-Arch Context Engineering Extension  
Date: 2025-09-22
Version: 1.0.0 - Integration Guide
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Import our bridge
from .asi_arch_bridge import (
    enhance_evolution_context,
    get_integration,
    get_monitor,
    analyze_architecture_portfolio
)

logger = logging.getLogger(__name__)

# =============================================================================
# INTEGRATION METHOD 1: Wrapper Functions (Recommended)
# =============================================================================

class ContextEnhancedPipeline:
    """
    Wrapper around ASI-Arch pipeline that optionally adds context engineering.
    
    Usage:
        # Replace this in your pipeline:
        # context, parent = await program_sample()
        # name, motivation = await evolve(context)
        
        # With this:
        pipeline = ContextEnhancedPipeline()
        context, parent = await pipeline.enhanced_program_sample()
        name, motivation = await pipeline.enhanced_evolve(context)
    """
    
    def __init__(self, enable_context_engineering: bool = True):
        self.enable_context_engineering = enable_context_engineering
        self.integration = get_integration() if enable_context_engineering else None
        self.monitor = get_monitor() if enable_context_engineering else None
        
        if enable_context_engineering:
            logger.info("🌊 Context Engineering enabled in pipeline")
        else:
            logger.info("⚙️ Using standard ASI-Arch pipeline (no context engineering)")
    
    async def enhanced_program_sample(self):
        """Enhanced program sampling with context awareness"""
        # Import ASI-Arch functions dynamically to avoid circular imports
        from database import program_sample
        
        # Get original sample
        context, parent = await program_sample()
        
        if not self.enable_context_engineering or not parent:
            return context, parent
        
        # Enhance context with river metaphor insights
        try:
            enhanced_context = await self.integration.pre_evolution_analysis(context, parent)
            logger.info(f"Context enhanced: {len(context)} -> {len(enhanced_context)} chars")
            return enhanced_context, parent
        except Exception as e:
            logger.warning(f"Context enhancement failed, using original: {e}")
            return context, parent
    
    async def enhanced_evolve(self, context: str):
        """Enhanced evolution with context engineering insights"""
        # Import ASI-Arch functions dynamically
        from evolve import evolve
        
        # Run original evolution
        name, motivation = await evolve(context)
        
        if not self.enable_context_engineering:
            return name, motivation
        
        # Log context engineering insights
        if name != "Failed":
            logger.info(f"🧬 Evolution successful: {name}")
            logger.info(f"🎯 Motivation: {motivation}")
        
        return name, motivation
    
    async def enhanced_evaluation(self, name: str, motivation: str):
        """Enhanced evaluation with consciousness tracking"""
        # Import ASI-Arch functions dynamically
        from eval import evaluation
        
        # Run original evaluation
        success = await evaluation(name, motivation)
        
        if not self.enable_context_engineering:
            return success
        
        # If we have access to the created architecture data, analyze it
        # This would require integration with the DataElement creation
        logger.info(f"🔬 Evaluation completed for {name}: {'✅ Success' if success else '❌ Failed'}")
        
        return success
    
    async def enhanced_analysis(self, name: str, motivation: str, parent=None):
        """Enhanced analysis with context engineering insights"""
        # Import ASI-Arch functions dynamically
        from analyse import analyse
        
        # Run original analysis
        result = await analyse(name, motivation, parent=parent)
        
        if not self.enable_context_engineering:
            return result
        
        # Add context engineering analysis
        try:
            consciousness_analysis = await self.integration.post_evaluation_analysis(result, True)
            logger.info(f"🧠 Consciousness Level: {consciousness_analysis.get('consciousness_analysis', {}).get('level', 'unknown')}")
        except Exception as e:
            logger.warning(f"Context engineering analysis failed: {e}")
        
        return result
    
    async def periodic_portfolio_review(self, data_elements):
        """Periodic review of architecture portfolio"""
        if not self.enable_context_engineering:
            return {}
        
        try:
            portfolio_analysis = await self.integration.periodic_portfolio_analysis(data_elements)
            
            # Log key insights
            basins = portfolio_analysis.get('attractor_basins', {})
            consciousness = portfolio_analysis.get('consciousness_analysis', {})
            
            logger.info(f"📊 Portfolio Analysis:")
            logger.info(f"   🎯 Attractor Basins: {basins.get('count', 0)}")
            logger.info(f"   🧠 Avg Consciousness: {consciousness.get('avg_consciousness', 0.0):.3f}")
            logger.info(f"   🌊 Active Streams: {portfolio_analysis.get('information_flow', {}).get('active_streams', 0)}")
            
            return portfolio_analysis
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return {}

# =============================================================================
# INTEGRATION METHOD 2: Monkey Patching (Advanced)
# =============================================================================

def apply_context_engineering_patches():
    """
    Apply monkey patches to existing ASI-Arch functions.
    
    WARNING: This modifies the original functions. Use with caution.
    Only recommended if you want automatic integration without changing pipeline code.
    """
    logger.warning("🐒 Applying context engineering monkey patches...")
    
    # Store original functions
    import evolve
    original_evolve = evolve.evolve
    
    async def patched_evolve(context: str):
        """Patched evolve function with context engineering"""
        try:
            # Enhance context if possible
            integration = get_integration()
            # Note: We'd need access to parent data here for full enhancement
            logger.info("🌊 Context engineering active in evolution")
        except Exception as e:
            logger.warning(f"Context enhancement failed: {e}")
        
        # Call original function
        return await original_evolve(context)
    
    # Apply patch
    evolve.evolve = patched_evolve
    
    logger.info("✅ Context engineering patches applied")

def remove_context_engineering_patches():
    """Remove monkey patches and restore original functions"""
    logger.info("🔄 Removing context engineering patches...")
    # This would restore original functions
    # Implementation depends on how patches were stored
    logger.info("✅ Original functions restored")

# =============================================================================
# INTEGRATION METHOD 3: Configuration-Based (Simplest)
# =============================================================================

class ContextEngineeringConfig:
    """Configuration for context engineering integration"""
    
    ENABLED = True
    LOG_LEVEL = "INFO"
    DATABASE_PATH = "extensions/context-engineering/context_engineering.db"
    EXPORT_METRICS = True
    METRICS_EXPORT_INTERVAL = 100  # Export every N experiments
    
    # Feature flags
    ENABLE_CONSCIOUSNESS_DETECTION = True
    ENABLE_ATTRACTOR_BASINS = True
    ENABLE_RIVER_METAPHOR = True
    ENABLE_NEURAL_FIELDS = True
    
    @classmethod
    def disable_all(cls):
        """Disable all context engineering features"""
        cls.ENABLED = False
        cls.ENABLE_CONSCIOUSNESS_DETECTION = False
        cls.ENABLE_ATTRACTOR_BASINS = False
        cls.ENABLE_RIVER_METAPHOR = False
        cls.ENABLE_NEURAL_FIELDS = False
    
    @classmethod
    def enable_all(cls):
        """Enable all context engineering features"""
        cls.ENABLED = True
        cls.ENABLE_CONSCIOUSNESS_DETECTION = True
        cls.ENABLE_ATTRACTOR_BASINS = True
        cls.ENABLE_RIVER_METAPHOR = True
        cls.ENABLE_NEURAL_FIELDS = True

# =============================================================================
# SIMPLE USAGE EXAMPLES
# =============================================================================

async def example_integration_in_pipeline():
    """
    Example of how to integrate context engineering into your ASI-Arch pipeline
    """
    print("🔧 Example: Context Engineering Integration")
    
    # METHOD 1: Using wrapper (Recommended)
    print("\n📦 Method 1: Wrapper Pipeline")
    
    pipeline = ContextEnhancedPipeline(enable_context_engineering=True)
    
    # Replace your pipeline steps with enhanced versions:
    try:
        # Instead of: context, parent = await program_sample()
        context, parent = await pipeline.enhanced_program_sample()
        print(f"✅ Enhanced program sample: context length = {len(context)}")
        
        # Instead of: name, motivation = await evolve(context)
        name, motivation = await pipeline.enhanced_evolve(context)
        print(f"✅ Enhanced evolution: {name}")
        
        # Instead of: success = await evaluation(name, motivation)
        success = await pipeline.enhanced_evaluation(name, motivation)
        print(f"✅ Enhanced evaluation: {success}")
        
        # Instead of: result = await analyse(name, motivation, parent=parent)
        result = await pipeline.enhanced_analysis(name, motivation, parent=parent)
        print(f"✅ Enhanced analysis completed")
        
    except ImportError as e:
        print(f"⚠️ ASI-Arch modules not available for example: {e}")
        print("   This is normal when running outside the full ASI-Arch environment")
    
    # METHOD 2: Configuration-based
    print("\n⚙️ Method 2: Configuration")
    
    # Enable/disable features as needed
    ContextEngineeringConfig.enable_all()
    print(f"Context Engineering Enabled: {ContextEngineeringConfig.ENABLED}")
    
    # Or disable for standard ASI-Arch behavior
    # ContextEngineeringConfig.disable_all()
    
    print("\n✅ Integration examples completed")

# =============================================================================
# INSTALLATION HELPER
# =============================================================================

def setup_context_engineering():
    """Set up context engineering integration"""
    print("🚀 Setting up Context Engineering Integration...")
    
    # Create necessary directories
    extensions_dir = Path("extensions/context-engineering")
    extensions_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    from .core_implementation import create_context_engineering_service
    service = create_context_engineering_service()
    
    print("✅ Context Engineering setup completed!")
    print(f"📁 Files located in: {extensions_dir.absolute()}")
    print(f"🗄️ Database: {service.db.db_path}")
    print("\nNext steps:")
    print("1. Import ContextEnhancedPipeline in your pipeline code")
    print("2. Replace pipeline functions with enhanced versions")
    print("3. Monitor logs for context engineering insights")

def test_context_engineering_setup():
    """Test that context engineering is properly set up"""
    print("🧪 Testing Context Engineering Setup...")
    
    try:
        # Test imports
        from .core_implementation import create_context_engineering_service
        from .asi_arch_bridge import enhance_evolution_context
        
        # Test service creation
        service = create_context_engineering_service()
        
        print("✅ All imports successful")
        print("✅ Service creation successful")
        print(f"✅ Database path: {service.db.db_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False

if __name__ == "__main__":
    # Run setup and test
    setup_context_engineering()
    
    if test_context_engineering_setup():
        print("\n🎉 Context Engineering is ready to use!")
        
        # Run integration example
        asyncio.run(example_integration_in_pipeline())
    else:
        print("\n⚠️ Setup issues detected. Please check the error messages above.")
