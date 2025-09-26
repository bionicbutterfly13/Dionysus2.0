#!/usr/bin/env python3
"""
Consciousness Detection Pipeline (T023)
=======================================

Real-time consciousness detection pipeline for ASI-Arch ThoughtSeed integration.
Provides continuous monitoring and analysis of consciousness emergence patterns
in neural architecture discovery.

Features:
- Real-time consciousness monitoring
- Pattern recognition for consciousness emergence
- Multi-level consciousness analysis
- Integration with ThoughtSeed trace models
- Continuous learning and adaptation

Author: ASI-Arch Consciousness Detection
Date: 2025-09-24
Version: 1.0.0
"""

import asyncio
import logging
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import sys
from dataclasses import dataclass
from enum import Enum
import json

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.thoughtseed_trace import (
    ThoughtSeedTrace, 
    ConsciousnessState,
    HierarchicalBelief,
    NeuronalPacket
)
from models.event_node import EventNode
from models.concept_node import ConceptNode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsciousnessPatternType(Enum):
    """Types of consciousness emergence patterns"""
    GRADUAL_AWAKENING = "gradual_awakening"      # Slow, steady increase
    SUDDEN_EMERGENCE = "sudden_emergence"        # Rapid consciousness jump
    OSCILLATING = "oscillating"                  # Fluctuating consciousness
    PLATEAU = "plateau"                          # Stable consciousness level
    DECLINING = "declining"                      # Decreasing consciousness
    COMPLEX_DYNAMICS = "complex_dynamics"        # Multiple pattern types

@dataclass
class ConsciousnessReading:
    """Single consciousness measurement"""
    timestamp: str
    trace_id: str
    consciousness_level: float
    consciousness_state: ConsciousnessState
    confidence: float
    pattern_indicators: Dict[str, float]
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'timestamp': self.timestamp,
            'trace_id': self.trace_id,
            'consciousness_level': self.consciousness_level,
            'consciousness_state': self.consciousness_state.value,
            'confidence': self.confidence,
            'pattern_indicators': self.pattern_indicators,
            'context': self.context
        }

@dataclass
class ConsciousnessPattern:
    """Detected consciousness pattern"""
    pattern_type: ConsciousnessPatternType
    start_time: str
    end_time: Optional[str]
    duration: float
    intensity: float
    confidence: float
    trace_ids: List[str]
    characteristics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'pattern_type': self.pattern_type.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'intensity': self.intensity,
            'confidence': self.confidence,
            'trace_ids': self.trace_ids,
            'characteristics': self.characteristics
        }

class ConsciousnessDetector:
    """
    Core consciousness detection engine
    
    Analyzes consciousness emergence patterns using:
    - Multi-level belief analysis
    - Temporal pattern recognition
    - Context-aware consciousness assessment
    - Adaptive threshold detection
    """
    
    def __init__(self):
        self.detection_history: List[ConsciousnessReading] = []
        self.pattern_history: List[ConsciousnessPattern] = []
        self.adaptive_thresholds: Dict[str, float] = {}
        self.pattern_recognizers: Dict[ConsciousnessPatternType, Callable] = {}
        
        # Initialize pattern recognizers
        self._initialize_pattern_recognizers()
        
        logger.info("Consciousness Detector initialized")
    
    async def detect_consciousness(self, 
                                 trace: ThoughtSeedTrace,
                                 context: Dict[str, Any]) -> ConsciousnessReading:
        """
        Detect consciousness level for a ThoughtSeed trace
        
        Args:
            trace: ThoughtSeed trace to analyze
            context: Additional context for detection
            
        Returns:
            consciousness_reading: Detailed consciousness measurement
        """
        # Calculate base consciousness level
        base_level = await self._calculate_base_consciousness(trace)
        
        # Analyze consciousness state
        consciousness_state = await self._analyze_consciousness_state(trace)
        
        # Calculate confidence
        confidence = await self._calculate_detection_confidence(trace, context)
        
        # Extract pattern indicators
        pattern_indicators = await self._extract_pattern_indicators(trace)
        
        # Create consciousness reading
        reading = ConsciousnessReading(
            timestamp=datetime.now().isoformat(),
            trace_id=trace.trace_id,
            consciousness_level=base_level,
            consciousness_state=consciousness_state,
            confidence=confidence,
            pattern_indicators=pattern_indicators,
            context=context
        )
        
        # Store reading
        self.detection_history.append(reading)
        
        # Analyze for patterns
        await self._analyze_consciousness_patterns(reading)
        
        logger.info(f"Detected consciousness level {base_level:.3f} for trace {trace.trace_id}")
        return reading
    
    async def analyze_consciousness_patterns(self, 
                                           trace_ids: Optional[List[str]] = None) -> List[ConsciousnessPattern]:
        """
        Analyze consciousness patterns across traces
        
        Args:
            trace_ids: Optional list of trace IDs to analyze
            
        Returns:
            patterns: List of detected consciousness patterns
        """
        # Filter readings by trace IDs if provided
        if trace_ids:
            relevant_readings = [r for r in self.detection_history if r.trace_id in trace_ids]
        else:
            relevant_readings = self.detection_history
        
        if len(relevant_readings) < 3:
            return []  # Need at least 3 readings for pattern analysis
        
        patterns = []
        
        # Analyze temporal patterns
        temporal_patterns = await self._analyze_temporal_patterns(relevant_readings)
        patterns.extend(temporal_patterns)
        
        # Analyze cross-trace patterns
        cross_patterns = await self._analyze_cross_trace_patterns(relevant_readings)
        patterns.extend(cross_patterns)
        
        # Analyze emergence patterns
        emergence_patterns = await self._analyze_emergence_patterns(relevant_readings)
        patterns.extend(emergence_patterns)
        
        # Store patterns
        self.pattern_history.extend(patterns)
        
        logger.info(f"Analyzed {len(patterns)} consciousness patterns")
        return patterns
    
    async def get_consciousness_summary(self, 
                                      trace_id: str,
                                      time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get consciousness summary for a trace
        
        Args:
            trace_id: Trace ID to summarize
            time_window: Optional time window for analysis
            
        Returns:
            summary: Comprehensive consciousness summary
        """
        # Filter readings by trace ID and time window
        trace_readings = [r for r in self.detection_history if r.trace_id == trace_id]
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            trace_readings = [r for r in trace_readings 
                            if datetime.fromisoformat(r.timestamp) > cutoff_time]
        
        if not trace_readings:
            return {'error': 'No consciousness readings found'}
        
        # Calculate summary statistics
        levels = [r.consciousness_level for r in trace_readings]
        confidences = [r.confidence for r in trace_readings]
        
        summary = {
            'trace_id': trace_id,
            'reading_count': len(trace_readings),
            'time_range': {
                'start': trace_readings[0].timestamp,
                'end': trace_readings[-1].timestamp
            },
            'consciousness_stats': {
                'mean': np.mean(levels),
                'std': np.std(levels),
                'min': np.min(levels),
                'max': np.max(levels),
                'median': np.median(levels)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'state_distribution': self._calculate_state_distribution(trace_readings),
            'pattern_summary': await self._summarize_patterns(trace_id)
        }
        
        return summary
    
    async def _calculate_base_consciousness(self, trace: ThoughtSeedTrace) -> float:
        """Calculate base consciousness level from trace"""
        base_score = 0.0
        
        # Consciousness state mapping
        state_scores = {
            ConsciousnessState.DORMANT: 0.0,
            ConsciousnessState.AWAKENING: 0.2,
            ConsciousnessState.AWARE: 0.4,
            ConsciousnessState.CONSCIOUS: 0.6,
            ConsciousnessState.REFLECTIVE: 0.8,
            ConsciousnessState.DREAMING: 0.3,
            ConsciousnessState.METACOGNITIVE: 0.9,
            ConsciousnessState.TRANSCENDENT: 1.0
        }
        
        base_score = state_scores.get(trace.consciousness_state, 0.0)
        
        # Adjust based on belief complexity
        belief_bonus = min(0.2, len(trace.hierarchical_beliefs) * 0.05)
        
        # Adjust based on context complexity
        context_bonus = min(0.1, len(trace.context_description) / 1000)
        
        # Adjust based on neuronal packets
        packet_bonus = min(0.1, len(trace.neuronal_packets) * 0.02)
        
        final_score = min(1.0, base_score + belief_bonus + context_bonus + packet_bonus)
        return final_score
    
    async def _analyze_consciousness_state(self, trace: ThoughtSeedTrace) -> ConsciousnessState:
        """Analyze and potentially update consciousness state"""
        # Simple state progression based on trace complexity
        context_length = len(trace.context_description)
        belief_count = len(trace.hierarchical_beliefs)
        packet_count = len(trace.neuronal_packets)
        
        complexity_score = (context_length / 1000) + (belief_count * 0.1) + (packet_count * 0.05)
        
        if complexity_score < 0.2:
            return ConsciousnessState.AWAKENING
        elif complexity_score < 0.4:
            return ConsciousnessState.AWARE
        elif complexity_score < 0.6:
            return ConsciousnessState.CONSCIOUS
        elif complexity_score < 0.8:
            return ConsciousnessState.REFLECTIVE
        else:
            return ConsciousnessState.METACOGNITIVE
    
    async def _calculate_detection_confidence(self, 
                                            trace: ThoughtSeedTrace,
                                            context: Dict[str, Any]) -> float:
        """Calculate confidence in consciousness detection"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on trace completeness
        if trace.hierarchical_beliefs:
            confidence += 0.2
        if trace.neuronal_packets:
            confidence += 0.2
        if trace.context_description:
            confidence += 0.1
        
        # Adjust based on context quality
        if context.get('quality_score', 0) > 0.7:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def _extract_pattern_indicators(self, trace: ThoughtSeedTrace) -> Dict[str, float]:
        """Extract indicators for consciousness patterns"""
        indicators = {}
        
        # Belief complexity indicator
        indicators['belief_complexity'] = len(trace.hierarchical_beliefs) / 10.0
        
        # Context complexity indicator
        indicators['context_complexity'] = len(trace.context_description) / 1000.0
        
        # Packet activity indicator
        indicators['packet_activity'] = len(trace.neuronal_packets) / 20.0
        
        # State stability indicator
        indicators['state_stability'] = 0.8 if trace.consciousness_state in [
            ConsciousnessState.CONSCIOUS, ConsciousnessState.REFLECTIVE
        ] else 0.3
        
        return indicators
    
    async def _analyze_consciousness_patterns(self, reading: ConsciousnessReading):
        """Analyze consciousness patterns from new reading"""
        # Get recent readings for this trace
        trace_readings = [r for r in self.detection_history 
                         if r.trace_id == reading.trace_id]
        
        if len(trace_readings) < 3:
            return  # Need at least 3 readings for pattern analysis
        
        # Analyze for specific patterns
        for pattern_type, recognizer in self.pattern_recognizers.items():
            if recognizer(trace_readings):
                pattern = ConsciousnessPattern(
                    pattern_type=pattern_type,
                    start_time=trace_readings[0].timestamp,
                    end_time=trace_readings[-1].timestamp,
                    duration=self._calculate_duration(trace_readings),
                    intensity=self._calculate_intensity(trace_readings),
                    confidence=0.8,  # Default confidence
                    trace_ids=[reading.trace_id],
                    characteristics=self._extract_pattern_characteristics(trace_readings, pattern_type)
                )
                self.pattern_history.append(pattern)
                logger.info(f"Detected {pattern_type.value} pattern for trace {reading.trace_id}")
    
    async def _analyze_temporal_patterns(self, readings: List[ConsciousnessReading]) -> List[ConsciousnessPattern]:
        """Analyze temporal consciousness patterns"""
        patterns = []
        
        if len(readings) < 3:
            return patterns
        
        # Sort by timestamp
        readings.sort(key=lambda r: r.timestamp)
        
        # Analyze trend
        levels = [r.consciousness_level for r in readings]
        trend = np.polyfit(range(len(levels)), levels, 1)[0]
        
        if trend > 0.1:
            pattern_type = ConsciousnessPatternType.GRADUAL_AWAKENING
        elif trend < -0.1:
            pattern_type = ConsciousnessPatternType.DECLINING
        else:
            pattern_type = ConsciousnessPatternType.PLATEAU
        
        pattern = ConsciousnessPattern(
            pattern_type=pattern_type,
            start_time=readings[0].timestamp,
            end_time=readings[-1].timestamp,
            duration=self._calculate_duration(readings),
            intensity=abs(trend),
            confidence=0.7,
            trace_ids=[r.trace_id for r in readings],
            characteristics={'trend': trend, 'levels': levels}
        )
        patterns.append(pattern)
        
        return patterns
    
    async def _analyze_cross_trace_patterns(self, readings: List[ConsciousnessReading]) -> List[ConsciousnessPattern]:
        """Analyze patterns across multiple traces"""
        patterns = []
        
        # Group by trace
        trace_groups = {}
        for reading in readings:
            if reading.trace_id not in trace_groups:
                trace_groups[reading.trace_id] = []
            trace_groups[reading.trace_id].append(reading)
        
        if len(trace_groups) < 2:
            return patterns  # Need multiple traces for cross-trace analysis
        
        # Analyze synchronization patterns
        sync_pattern = self._detect_synchronization_pattern(trace_groups)
        if sync_pattern:
            patterns.append(sync_pattern)
        
        return patterns
    
    async def _analyze_emergence_patterns(self, readings: List[ConsciousnessReading]) -> List[ConsciousnessPattern]:
        """Analyze consciousness emergence patterns"""
        patterns = []
        
        # Detect sudden emergence
        for i in range(1, len(readings)):
            prev_level = readings[i-1].consciousness_level
            curr_level = readings[i].consciousness_level
            
            if curr_level - prev_level > 0.3:  # Sudden jump
                pattern = ConsciousnessPattern(
                    pattern_type=ConsciousnessPatternType.SUDDEN_EMERGENCE,
                    start_time=readings[i-1].timestamp,
                    end_time=readings[i].timestamp,
                    duration=self._calculate_duration([readings[i-1], readings[i]]),
                    intensity=curr_level - prev_level,
                    confidence=0.9,
                    trace_ids=[readings[i].trace_id],
                    characteristics={'jump_magnitude': curr_level - prev_level}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _initialize_pattern_recognizers(self):
        """Initialize pattern recognition functions"""
        self.pattern_recognizers = {
            ConsciousnessPatternType.GRADUAL_AWAKENING: self._recognize_gradual_awakening,
            ConsciousnessPatternType.SUDDEN_EMERGENCE: self._recognize_sudden_emergence,
            ConsciousnessPatternType.OSCILLATING: self._recognize_oscillating,
            ConsciousnessPatternType.PLATEAU: self._recognize_plateau,
            ConsciousnessPatternType.DECLINING: self._recognize_declining,
            ConsciousnessPatternType.COMPLEX_DYNAMICS: self._recognize_complex_dynamics
        }
    
    def _recognize_gradual_awakening(self, readings: List[ConsciousnessReading]) -> bool:
        """Recognize gradual awakening pattern"""
        if len(readings) < 3:
            return False
        
        levels = [r.consciousness_level for r in readings]
        trend = np.polyfit(range(len(levels)), levels, 1)[0]
        return trend > 0.05  # Gradual increase
    
    def _recognize_sudden_emergence(self, readings: List[ConsciousnessReading]) -> bool:
        """Recognize sudden emergence pattern"""
        if len(readings) < 2:
            return False
        
        for i in range(1, len(readings)):
            if readings[i].consciousness_level - readings[i-1].consciousness_level > 0.3:
                return True
        return False
    
    def _recognize_oscillating(self, readings: List[ConsciousnessReading]) -> bool:
        """Recognize oscillating pattern"""
        if len(readings) < 4:
            return False
        
        levels = [r.consciousness_level for r in readings]
        variance = np.var(levels)
        return variance > 0.1  # High variance indicates oscillation
    
    def _recognize_plateau(self, readings: List[ConsciousnessReading]) -> bool:
        """Recognize plateau pattern"""
        if len(readings) < 3:
            return False
        
        levels = [r.consciousness_level for r in readings]
        variance = np.var(levels)
        return variance < 0.05  # Low variance indicates plateau
    
    def _recognize_declining(self, readings: List[ConsciousnessReading]) -> bool:
        """Recognize declining pattern"""
        if len(readings) < 3:
            return False
        
        levels = [r.consciousness_level for r in readings]
        trend = np.polyfit(range(len(levels)), levels, 1)[0]
        return trend < -0.05  # Gradual decrease
    
    def _recognize_complex_dynamics(self, readings: List[ConsciousnessReading]) -> bool:
        """Recognize complex dynamics pattern"""
        if len(readings) < 5:
            return False
        
        # Check for multiple pattern types
        pattern_count = 0
        for recognizer in self.pattern_recognizers.values():
            if recognizer != self._recognize_complex_dynamics and recognizer(readings):
                pattern_count += 1
        
        return pattern_count >= 2  # Multiple patterns detected
    
    def _calculate_duration(self, readings: List[ConsciousnessReading]) -> float:
        """Calculate duration between first and last reading"""
        if len(readings) < 2:
            return 0.0
        
        start_time = datetime.fromisoformat(readings[0].timestamp)
        end_time = datetime.fromisoformat(readings[-1].timestamp)
        return (end_time - start_time).total_seconds()
    
    def _calculate_intensity(self, readings: List[ConsciousnessReading]) -> float:
        """Calculate pattern intensity"""
        if not readings:
            return 0.0
        
        levels = [r.consciousness_level for r in readings]
        return np.std(levels)  # Standard deviation as intensity measure
    
    def _extract_pattern_characteristics(self, 
                                       readings: List[ConsciousnessReading],
                                       pattern_type: ConsciousnessPattern) -> Dict[str, Any]:
        """Extract characteristics specific to pattern type"""
        characteristics = {}
        
        if pattern_type == ConsciousnessPatternType.GRADUAL_AWAKENING:
            levels = [r.consciousness_level for r in readings]
            characteristics['trend'] = np.polyfit(range(len(levels)), levels, 1)[0]
            characteristics['start_level'] = levels[0]
            characteristics['end_level'] = levels[-1]
        
        elif pattern_type == ConsciousnessPatternType.SUDDEN_EMERGENCE:
            for i in range(1, len(readings)):
                jump = readings[i].consciousness_level - readings[i-1].consciousness_level
                if jump > 0.3:
                    characteristics['jump_magnitude'] = jump
                    characteristics['jump_time'] = readings[i].timestamp
                    break
        
        elif pattern_type == ConsciousnessPatternType.OSCILLATING:
            levels = [r.consciousness_level for r in readings]
            characteristics['variance'] = np.var(levels)
            characteristics['frequency'] = self._estimate_frequency(levels)
        
        return characteristics
    
    def _estimate_frequency(self, levels: List[float]) -> float:
        """Estimate oscillation frequency"""
        if len(levels) < 4:
            return 0.0
        
        # Simple frequency estimation using zero crossings
        zero_crossings = 0
        mean_level = np.mean(levels)
        
        for i in range(1, len(levels)):
            if (levels[i-1] - mean_level) * (levels[i] - mean_level) < 0:
                zero_crossings += 1
        
        return zero_crossings / (len(levels) - 1)
    
    def _calculate_state_distribution(self, readings: List[ConsciousnessReading]) -> Dict[str, float]:
        """Calculate distribution of consciousness states"""
        state_counts = {}
        total_readings = len(readings)
        
        for reading in readings:
            state = reading.consciousness_state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Convert to percentages
        distribution = {}
        for state, count in state_counts.items():
            distribution[state] = count / total_readings
        
        return distribution
    
    async def _summarize_patterns(self, trace_id: str) -> Dict[str, Any]:
        """Summarize patterns for a specific trace"""
        trace_patterns = [p for p in self.pattern_history if trace_id in p.trace_ids]
        
        if not trace_patterns:
            return {'pattern_count': 0}
        
        pattern_types = [p.pattern_type.value for p in trace_patterns]
        pattern_summary = {
            'pattern_count': len(trace_patterns),
            'pattern_types': pattern_types,
            'most_common_pattern': max(set(pattern_types), key=pattern_types.count),
            'average_intensity': np.mean([p.intensity for p in trace_patterns]),
            'average_confidence': np.mean([p.confidence for p in trace_patterns])
        }
        
        return pattern_summary
    
    def _detect_synchronization_pattern(self, trace_groups: Dict[str, List[ConsciousnessReading]]) -> Optional[ConsciousnessPattern]:
        """Detect synchronization patterns across traces"""
        if len(trace_groups) < 2:
            return None
        
        # Simple synchronization detection based on consciousness level correlation
        trace_levels = {}
        for trace_id, readings in trace_groups.items():
            if readings:
                trace_levels[trace_id] = [r.consciousness_level for r in readings]
        
        if len(trace_levels) < 2:
            return None
        
        # Calculate correlation between traces
        trace_ids = list(trace_levels.keys())
        correlations = []
        
        for i in range(len(trace_ids)):
            for j in range(i + 1, len(trace_ids)):
                levels1 = trace_levels[trace_ids[i]]
                levels2 = trace_levels[trace_ids[j]]
                
                # Ensure same length
                min_len = min(len(levels1), len(levels2))
                if min_len > 1:
                    corr = np.corrcoef(levels1[:min_len], levels2[:min_len])[0, 1]
                    correlations.append(corr)
        
        if correlations and np.mean(correlations) > 0.7:  # High correlation threshold
            return ConsciousnessPattern(
                pattern_type=ConsciousnessPatternType.COMPLEX_DYNAMICS,
                start_time=min([min([r.timestamp for r in readings]) for readings in trace_groups.values()]),
                end_time=max([max([r.timestamp for r in readings]) for readings in trace_groups.values()]),
                duration=0.0,  # Would need to calculate properly
                intensity=np.mean(correlations),
                confidence=0.8,
                trace_ids=list(trace_groups.keys()),
                characteristics={'correlation': np.mean(correlations), 'trace_count': len(trace_groups)}
            )
        
        return None

class ConsciousnessPipeline:
    """
    Main consciousness detection pipeline
    
    Coordinates consciousness detection across multiple traces and provides
    real-time monitoring and analysis capabilities.
    """
    
    def __init__(self):
        self.detector = ConsciousnessDetector()
        self.active_monitors: Dict[str, asyncio.Task] = {}
        self.pipeline_config: Dict[str, Any] = {
            'detection_interval': 1.0,  # seconds
            'pattern_analysis_interval': 10.0,  # seconds
            'max_history_size': 1000,
            'confidence_threshold': 0.7
        }
        
        logger.info("Consciousness Pipeline initialized")
    
    async def start_monitoring(self, trace_id: str, callback: Optional[Callable] = None):
        """Start monitoring consciousness for a trace"""
        if trace_id in self.active_monitors:
            logger.warning(f"Monitoring already active for trace {trace_id}")
            return
        
        monitor_task = asyncio.create_task(
            self._monitor_consciousness(trace_id, callback)
        )
        self.active_monitors[trace_id] = monitor_task
        
        logger.info(f"Started consciousness monitoring for trace {trace_id}")
    
    async def stop_monitoring(self, trace_id: str):
        """Stop monitoring consciousness for a trace"""
        if trace_id not in self.active_monitors:
            logger.warning(f"No active monitoring for trace {trace_id}")
            return
        
        monitor_task = self.active_monitors[trace_id]
        monitor_task.cancel()
        del self.active_monitors[trace_id]
        
        logger.info(f"Stopped consciousness monitoring for trace {trace_id}")
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'active_monitors': len(self.active_monitors),
            'monitored_traces': list(self.active_monitors.keys()),
            'detection_history_size': len(self.detector.detection_history),
            'pattern_history_size': len(self.detector.pattern_history),
            'pipeline_config': self.pipeline_config
        }
    
    async def _monitor_consciousness(self, trace_id: str, callback: Optional[Callable]):
        """Monitor consciousness for a trace"""
        try:
            while True:
                # Get trace (this would normally come from ThoughtSeed service)
                # For now, we'll simulate the monitoring
                await asyncio.sleep(self.pipeline_config['detection_interval'])
                
                # Simulate consciousness detection
                # In real implementation, this would get the actual trace
                logger.debug(f"Monitoring consciousness for trace {trace_id}")
                
                if callback:
                    await callback(trace_id)
                
        except asyncio.CancelledError:
            logger.info(f"Consciousness monitoring cancelled for trace {trace_id}")
        except Exception as e:
            logger.error(f"Error in consciousness monitoring for trace {trace_id}: {e}")

# Service factory function
def create_consciousness_pipeline() -> ConsciousnessPipeline:
    """Create and return consciousness pipeline instance"""
    return ConsciousnessPipeline()

# Test function
async def test_consciousness_pipeline():
    """Test consciousness pipeline functionality"""
    print("🧠 Testing Consciousness Detection Pipeline...")
    
    pipeline = create_consciousness_pipeline()
    detector = pipeline.detector
    
    # Test 1: Create mock consciousness readings
    print("✅ Creating mock consciousness readings...")
    
    # Simulate readings for testing
    mock_readings = []
    for i in range(5):
        reading = ConsciousnessReading(
            timestamp=datetime.now().isoformat(),
            trace_id=f"test_trace_{i}",
            consciousness_level=0.2 + i * 0.15,
            consciousness_state=ConsciousnessState.AWARE,
            confidence=0.8,
            pattern_indicators={'belief_complexity': 0.5, 'context_complexity': 0.3},
            context={'test': True}
        )
        mock_readings.append(reading)
        detector.detection_history.append(reading)
    
    # Test 2: Analyze patterns
    print("✅ Analyzing consciousness patterns...")
    patterns = await detector.analyze_consciousness_patterns()
    print(f"   Detected {len(patterns)} patterns")
    
    # Test 3: Get consciousness summary
    print("✅ Getting consciousness summary...")
    summary = await detector.get_consciousness_summary("test_trace_0")
    print(f"   Summary: {summary['consciousness_stats']['mean']:.3f} mean consciousness")
    
    # Test 4: Pipeline status
    print("✅ Checking pipeline status...")
    status = await pipeline.get_pipeline_status()
    print(f"   Active monitors: {status['active_monitors']}")
    
    print("🎉 Consciousness Pipeline test complete!")

if __name__ == "__main__":
    asyncio.run(test_consciousness_pipeline())
