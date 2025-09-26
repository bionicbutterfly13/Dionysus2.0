#!/usr/bin/env python3
"""
🌱🧠 Complete ThoughtSeed Implementation Test
============================================

Comprehensive test of the ThoughtSeed ASI-Arch implementation including:
- Core ThoughtSeed functionality
- Active inference integration
- Dionysus bridge (fallback mode)
- Enhanced pipeline components
- Database connections

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-23
Version: 1.0.0 - Implementation Validation
"""

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.append('extensions/context_engineering')

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing Critical Imports...")

    results = {}

    # Test ThoughtSeed core
    try:
        from thoughtseed_active_inference import (
            ASIArchThoughtseedIntegration,
            Thoughtseed,
            NeuronalPacket,
            ThoughtseedType
        )
        results['thoughtseed_core'] = "✅ Available"
    except Exception as e:
        results['thoughtseed_core'] = f"❌ Failed: {e}"

    # Test ASI-Arch bridge
    try:
        from asi_arch_thoughtseed_bridge import ASIArchThoughtseedBridge
        results['asi_arch_bridge'] = "✅ Available"
    except Exception as e:
        results['asi_arch_bridge'] = f"❌ Failed: {e}"

    # Test Dionysus integration
    try:
        from dionysus_thoughtseed_integration import DionysusThoughtseedBridge
        results['dionysus_integration'] = "✅ Available (with fallback)"
    except Exception as e:
        results['dionysus_integration'] = f"❌ Failed: {e}"

    # Test enhanced pipeline
    try:
        from thoughtseed_enhanced_pipeline import ThoughtseedEnhancedPipeline
        results['enhanced_pipeline'] = "✅ Available"
    except Exception as e:
        results['enhanced_pipeline'] = f"❌ Failed: {e}"

    # Test unified database
    try:
        from unified_database import Neo4jKnowledgeGraph
        results['unified_database'] = "✅ Available"
    except Exception as e:
        results['unified_database'] = f"❌ Failed: {e}"

    return results

async def test_thoughtseed_functionality():
    """Test core ThoughtSeed functionality"""
    print("\n🌱 Testing ThoughtSeed Core Functionality...")

    try:
        from thoughtseed_active_inference import ASIArchThoughtseedIntegration

        # Initialize integration
        integration = ASIArchThoughtseedIntegration()
        print("   ✅ ThoughtSeed integration initialized")

        # Test context enhancement
        test_context = "Design an attention mechanism with emergent consciousness"
        test_arch_data = {
            'name': 'TestArchitecture',
            'performance': 0.85,
            'innovations': ['attention', 'consciousness']
        }

        result = await integration.enhance_evolution_context(test_context, test_arch_data)

        print(f"   ✅ Context enhancement working")
        print(f"   📊 Consciousness level: {result['consciousness_level']:.2f}")
        print(f"   📊 Enhanced context length: {len(result['enhanced_context'])} chars")

        return True

    except Exception as e:
        print(f"   ❌ ThoughtSeed functionality failed: {e}")
        return False

async def test_dionysus_integration():
    """Test Dionysus integration (fallback mode)"""
    print("\n🧠 Testing Dionysus Integration...")

    try:
        from dionysus_thoughtseed_integration import DionysusThoughtseedBridge

        # Initialize bridge
        bridge = DionysusThoughtseedBridge()
        print("   ✅ Dionysus bridge initialized")

        # Test architecture analysis
        test_arch_data = {
            'name': 'DionysusTestArch',
            'motivation': 'Test active inference with prediction error minimization',
            'program': 'class TestNet: def forward(self): return self.predict()',
            'performance': 0.75
        }

        profile = await bridge.analyze_architecture_with_dionysus(test_arch_data)

        print(f"   ✅ Architecture analysis working")
        print(f"   📊 Free energy: {profile.free_energy:.3f}")
        print(f"   📊 Consciousness: {profile.architecture_consciousness:.3f}")
        print(f"   📊 Meta-awareness: {profile.meta_awareness_level:.3f}")

        return True

    except Exception as e:
        print(f"   ❌ Dionysus integration failed: {e}")
        return False

async def test_enhanced_pipeline():
    """Test enhanced pipeline components"""
    print("\n🔄 Testing Enhanced Pipeline Components...")

    try:
        from thoughtseed_enhanced_pipeline import ThoughtseedEnhancedPipeline

        # Initialize pipeline
        pipeline = ThoughtseedEnhancedPipeline()
        print("   ✅ Enhanced pipeline initialized")

        # Test enhanced evolution (without ASI-Arch backend)
        test_context = "Create consciousness-aware transformer architecture"

        try:
            evolution_result = await pipeline.enhanced_evolve(test_context)
            print(f"   ✅ Enhanced evolution functional")
            print(f"   📊 Result: {evolution_result[0]}")
        except Exception as e:
            print(f"   ⚠️  Enhanced evolution failed (expected without ASI-Arch): {e}")

        # Test enhanced evaluation
        test_eval_result = await pipeline.enhanced_eval("TestArch", "Test motivation")
        print(f"   ✅ Enhanced evaluation functional")
        print(f"   📊 Consciousness: {test_eval_result['consciousness_level']:.2f}")

        return True

    except Exception as e:
        print(f"   ❌ Enhanced pipeline failed: {e}")
        return False

def test_database_connections():
    """Test database connection capabilities"""
    print("\n🗄️ Testing Database Connections...")

    results = {}

    # Test Redis connection
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        results['redis'] = "✅ Connected"
    except Exception as e:
        results['redis'] = f"❌ Failed: {e}"

    # Test Neo4j connection (may fail if not running)
    try:
        from unified_database import Neo4jKnowledgeGraph
        neo4j = Neo4jKnowledgeGraph()
        # Don't actually connect, just test initialization
        results['neo4j'] = "✅ Initialize ready (may need startup)"
    except Exception as e:
        results['neo4j'] = f"❌ Failed: {e}"

    return results

async def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🌱🧠 ThoughtSeed ASI-Arch Implementation Comprehensive Test")
    print("=" * 70)

    # Test imports
    import_results = test_imports()
    print("\n📦 Import Test Results:")
    for component, status in import_results.items():
        print(f"   {component}: {status}")

    # Test core functionality
    thoughtseed_success = await test_thoughtseed_functionality()

    # Test Dionysus integration
    dionysus_success = await test_dionysus_integration()

    # Test enhanced pipeline
    pipeline_success = await test_enhanced_pipeline()

    # Test database connections
    db_results = test_database_connections()
    print("\n🗄️ Database Connection Results:")
    for db, status in db_results.items():
        print(f"   {db}: {status}")

    # Overall assessment
    print("\n🎯 COMPREHENSIVE TEST SUMMARY")
    print("=" * 50)

    successes = [
        thoughtseed_success,
        dionysus_success,
        pipeline_success
    ]

    success_rate = sum(successes) / len(successes)

    print(f"Overall Success Rate: {success_rate * 100:.0f}%")
    print(f"ThoughtSeed Core: {'✅ PASS' if thoughtseed_success else '❌ FAIL'}")
    print(f"Dionysus Integration: {'✅ PASS' if dionysus_success else '❌ FAIL'}")
    print(f"Enhanced Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")

    print(f"\n🌟 IMPLEMENTATION STATUS:")
    if success_rate >= 0.8:
        print("   ✅ EXCELLENT - Ready for production use")
    elif success_rate >= 0.6:
        print("   ⚠️  GOOD - Minor issues to address")
    else:
        print("   ❌ NEEDS WORK - Major issues to resolve")

    print(f"\n🚀 NEXT STEPS:")
    if not thoughtseed_success:
        print("   - Debug ThoughtSeed core functionality")
    if not dionysus_success:
        print("   - Fix Dionysus import issues")
    if not pipeline_success:
        print("   - Address pipeline integration problems")
    if all(successes):
        print("   - Deploy and begin architecture discovery!")
        print("   - Set up production Neo4j instance")
        print("   - Configure ASI-Arch backend integration")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())