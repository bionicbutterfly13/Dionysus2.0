"""
Component Discovery Service

Analyzes legacy Dionysus consciousness codebase to identify components
and their consciousness functionality characteristics.
"""

import ast
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..config import get_migration_config
from ..logging_config import get_migration_logger
from ..models.legacy_component import (
    AnalysisStatus,
    ConsciousnessFunctionality,
    LegacyComponent,
    StrategicValue
)


class ComponentDiscoveryService:
    """
    Service for discovering and analyzing legacy consciousness components
    """

    def __init__(self):
        self.config = get_migration_config()
        self.logger = get_migration_logger()
        self.consciousness_patterns = self._load_consciousness_patterns()
        self.strategic_indicators = self._load_strategic_indicators()

    def discover_components(self, codebase_path: str) -> List[LegacyComponent]:
        """
        Discover all components in the legacy codebase

        Args:
            codebase_path: Path to legacy Dionysus consciousness codebase

        Returns:
            List of discovered legacy components
        """
        self.logger.info(
            "Starting component discovery",
            codebase_path=codebase_path
        )

        codebase = Path(codebase_path)
        if not codebase.exists():
            raise ValueError(f"Codebase path does not exist: {codebase_path}")

        discovered_components = []
        python_files = list(codebase.rglob("*.py"))

        self.logger.info(
            "Found Python files for analysis",
            file_count=len(python_files),
            codebase_path=codebase_path
        )

        for file_path in python_files:
            try:
                components = self._analyze_file(file_path)
                discovered_components.extend(components)
            except Exception as e:
                self.logger.error(
                    "Failed to analyze file",
                    file_path=str(file_path),
                    error=str(e)
                )

        self.logger.info(
            "Component discovery completed",
            total_components=len(discovered_components),
            codebase_path=codebase_path
        )

        return discovered_components

    def _analyze_file(self, file_path: Path) -> List[LegacyComponent]:
        """
        Analyze a single Python file for consciousness components

        Args:
            file_path: Path to the Python file

        Returns:
            List of components found in the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            self.logger.warning(
                "Could not read file",
                file_path=str(file_path),
                error=str(e)
            )
            return []

        # Parse AST to extract components
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            self.logger.warning(
                "Syntax error in file",
                file_path=str(file_path),
                error=str(e)
            )
            return []

        components = []

        # Analyze classes and functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                component = self._analyze_component_node(
                    node, file_path, source_code
                )
                if component and self._is_consciousness_relevant(component):
                    components.append(component)

        return components

    def _analyze_component_node(
        self,
        node: ast.AST,
        file_path: Path,
        source_code: str
    ) -> Optional[LegacyComponent]:
        """
        Analyze a specific AST node (class or function) for consciousness patterns

        Args:
            node: AST node to analyze
            file_path: Path to the source file
            source_code: Complete source code

        Returns:
            LegacyComponent if consciousness patterns detected, None otherwise
        """
        if not hasattr(node, 'name'):
            return None

        component_name = node.name

        # Extract component source code
        try:
            component_source = ast.get_source_segment(source_code, node)
            if not component_source:
                # Fallback for older Python versions
                component_source = self._extract_component_source(
                    source_code, node
                )
        except Exception:
            component_source = ""

        # Generate component ID
        component_id = LegacyComponent.generate_component_id(
            str(file_path), component_source
        )

        # Analyze consciousness functionality
        consciousness_functionality = self._analyze_consciousness_patterns(
            component_source, component_name
        )

        # Analyze strategic value
        strategic_value = self._analyze_strategic_value(
            component_source, component_name, file_path
        )

        # Calculate quality score
        quality_score = LegacyComponent.calculate_quality_score(
            consciousness_functionality,
            strategic_value,
            consciousness_weight=self.config.consciousness_weight,
            strategic_weight=self.config.strategic_weight
        )

        # Create component
        component = LegacyComponent(
            component_id=component_id,
            name=component_name,
            file_path=str(file_path.absolute()),
            consciousness_functionality=consciousness_functionality,
            strategic_value=strategic_value,
            quality_score=quality_score,
            analysis_status=AnalysisStatus.ANALYZED,
            source_code_hash=hashlib.sha256(component_source.encode()).hexdigest(),
            file_size_bytes=len(component_source.encode()),
            consciousness_patterns=self._detect_consciousness_patterns(component_source)
        )

        return component

    def _analyze_consciousness_patterns(
        self,
        source_code: str,
        component_name: str
    ) -> ConsciousnessFunctionality:
        """
        Analyze source code for consciousness functionality patterns

        Args:
            source_code: Component source code
            component_name: Name of the component

        Returns:
            ConsciousnessFunctionality metrics
        """
        awareness_score = self._calculate_awareness_score(source_code, component_name)
        inference_score = self._calculate_inference_score(source_code, component_name)
        memory_score = self._calculate_memory_score(source_code, component_name)

        return ConsciousnessFunctionality(
            awareness_score=awareness_score,
            inference_score=inference_score,
            memory_score=memory_score
        )

    def _analyze_strategic_value(
        self,
        source_code: str,
        component_name: str,
        file_path: Path
    ) -> StrategicValue:
        """
        Analyze strategic value for migration prioritization

        Args:
            source_code: Component source code
            component_name: Name of the component
            file_path: Path to the source file

        Returns:
            StrategicValue metrics
        """
        uniqueness_score = self._calculate_uniqueness_score(source_code, component_name)
        reusability_score = self._calculate_reusability_score(source_code, component_name)
        framework_alignment_score = self._calculate_framework_alignment(source_code)

        return StrategicValue(
            uniqueness_score=uniqueness_score,
            reusability_score=reusability_score,
            framework_alignment_score=framework_alignment_score
        )

    def _calculate_awareness_score(self, source_code: str, component_name: str) -> float:
        """Calculate awareness processing capability score"""
        awareness_patterns = [
            r'awareness|conscious|perceive|observe|monitor',
            r'state.*track|context.*aware|situation.*aware',
            r'attention|focus|alert|vigilant',
            r'self.*monitor|meta.*aware|introspect'
        ]

        score = 0.0
        for pattern in awareness_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                score += 0.25

        # Boost for awareness-related naming
        if re.search(r'awareness|conscious|perceive', component_name, re.IGNORECASE):
            score += 0.2

        return min(1.0, score)

    def _calculate_inference_score(self, source_code: str, component_name: str) -> float:
        """Calculate inference and reasoning capability score"""
        inference_patterns = [
            r'infer|deduce|conclude|reason',
            r'predict|forecast|anticipate',
            r'analyze|evaluate|assess|judge',
            r'decision|choice|select|choose',
            r'logic|reasoning|inference'
        ]

        score = 0.0
        for pattern in inference_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                score += 0.2

        # Check for complex reasoning structures
        if re.search(r'if.*elif.*else|switch|case', source_code):
            score += 0.1

        # Boost for inference-related naming
        if re.search(r'infer|reason|logic|decision', component_name, re.IGNORECASE):
            score += 0.2

        return min(1.0, score)

    def _calculate_memory_score(self, source_code: str, component_name: str) -> float:
        """Calculate memory integration capability score"""
        memory_patterns = [
            r'memory|remember|recall|retrieve',
            r'store|save|persist|cache',
            r'episodic|semantic|procedural',
            r'knowledge|experience|learning',
            r'history|past|temporal'
        ]

        score = 0.0
        for pattern in memory_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                score += 0.2

        # Check for data structures that suggest memory
        if re.search(r'dict|list|array|database|storage', source_code):
            score += 0.1

        # Boost for memory-related naming
        if re.search(r'memory|store|cache|knowledge', component_name, re.IGNORECASE):
            score += 0.2

        return min(1.0, score)

    def _calculate_uniqueness_score(self, source_code: str, component_name: str) -> float:
        """Calculate architectural uniqueness score"""
        # Look for unique patterns, custom algorithms, novel approaches
        uniqueness_indicators = [
            r'custom|novel|unique|proprietary',
            r'algorithm|heuristic|optimization',
            r'experimental|innovative|advanced',
            r'patent|research|academic'
        ]

        score = 0.5  # Base score

        for pattern in uniqueness_indicators:
            if re.search(pattern, source_code, re.IGNORECASE):
                score += 0.1

        # Check code complexity as uniqueness indicator
        if len(source_code) > 1000:  # Large components likely more unique
            score += 0.1

        return min(1.0, score)

    def _calculate_reusability_score(self, source_code: str, component_name: str) -> float:
        """Calculate reusability potential score"""
        reusability_indicators = [
            r'class|def|function',
            r'abstract|interface|base',
            r'util|helper|tool|service',
            r'generic|template|pattern'
        ]

        score = 0.3  # Base score

        for pattern in reusability_indicators:
            matches = len(re.findall(pattern, source_code, re.IGNORECASE))
            score += min(0.2, matches * 0.05)

        # Check for good documentation (indicates reusability intent)
        docstring_count = len(re.findall(r'""".*?"""', source_code, re.DOTALL))
        score += min(0.2, docstring_count * 0.1)

        return min(1.0, score)

    def _calculate_framework_alignment(self, source_code: str) -> float:
        """Calculate alignment with Dionysus 2.0 framework"""
        framework_patterns = [
            r'async|await',  # Modern async patterns
            r'pydantic|fastapi',  # Modern Python frameworks
            r'typing|type.*hint',  # Type safety
            r'dataclass|attrs',  # Modern data structures
            r'pytest|unittest'  # Testing frameworks
        ]

        score = 0.2  # Base score

        for pattern in framework_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                score += 0.15

        return min(1.0, score)

    def _detect_consciousness_patterns(self, source_code: str) -> List[str]:
        """Detect specific consciousness patterns in code"""
        patterns = []

        consciousness_indicators = {
            'awareness_processing': r'awareness|conscious|perceive',
            'inference_engine': r'infer|deduce|reason|logic',
            'memory_system': r'memory|remember|recall|store',
            'attention_mechanism': r'attention|focus|select',
            'meta_cognition': r'meta|self.*aware|introspect',
            'state_management': r'state|context|situation',
            'learning_system': r'learn|adapt|improve|evolve'
        }

        for pattern_name, regex in consciousness_indicators.items():
            if re.search(regex, source_code, re.IGNORECASE):
                patterns.append(pattern_name)

        return patterns

    def _is_consciousness_relevant(self, component: LegacyComponent) -> bool:
        """Check if component has sufficient consciousness relevance for migration"""
        # Component is relevant if it has consciousness patterns or reasonable scores
        has_consciousness_patterns = len(component.consciousness_patterns) > 0
        has_sufficient_consciousness = component.consciousness_functionality.composite_score >= 0.3
        has_strategic_value = component.strategic_value.composite_score >= 0.4

        return has_consciousness_patterns or has_sufficient_consciousness or has_strategic_value

    def _extract_component_source(self, source_code: str, node: ast.AST) -> str:
        """Fallback method to extract component source code"""
        lines = source_code.split('\n')
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            start_line = max(0, node.lineno - 1)
            end_line = min(len(lines), node.end_lineno or node.lineno)
            return '\n'.join(lines[start_line:end_line])
        return ""

    def _load_consciousness_patterns(self) -> Dict[str, List[str]]:
        """Load consciousness pattern definitions"""
        return {
            'awareness': [
                'consciousness', 'awareness', 'perception', 'observation',
                'monitoring', 'detection', 'recognition', 'identification'
            ],
            'inference': [
                'reasoning', 'inference', 'deduction', 'analysis',
                'evaluation', 'judgment', 'decision', 'prediction'
            ],
            'memory': [
                'memory', 'storage', 'recall', 'retrieval',
                'episodic', 'semantic', 'procedural', 'knowledge'
            ]
        }

    def _load_strategic_indicators(self) -> Dict[str, List[str]]:
        """Load strategic value indicators"""
        return {
            'uniqueness': [
                'novel', 'unique', 'custom', 'proprietary',
                'innovative', 'experimental', 'research', 'patent'
            ],
            'reusability': [
                'utility', 'helper', 'service', 'library',
                'framework', 'generic', 'abstract', 'interface'
            ],
            'alignment': [
                'async', 'modern', 'standard', 'best-practice',
                'scalable', 'maintainable', 'testable', 'documented'
            ]
        }