"""
Document Cognition Base - Strategy repository for document processing
Adapted from ASI-GO-2 cognition_base.py with consciousness enhancements
"""
from typing import Dict, List, Any
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("dionysus.document_cognition")


class DocumentCognitionBase:
    """Stores and retrieves document processing strategies and patterns"""

    def __init__(self, knowledge_file: str = "document_cognition_knowledge.json"):
        self.knowledge_file = Path(knowledge_file)
        self.knowledge = self._load_knowledge()
        self.session_insights = []

    def _load_knowledge(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load existing knowledge or initialize with base strategies"""
        if self.knowledge_file.exists():
            try:
                with open(self.knowledge_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load knowledge file: {e}")

        return self._initialize_base_knowledge()

    def _initialize_base_knowledge(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize with document processing strategies from external sources"""
        return {
            "extraction_strategies": [
                {
                    "name": "Content Hash Deduplication",
                    "source": "SurfSense",
                    "description": "Generate SHA-256 hash to detect duplicate documents",
                    "applicable_to": ["pdf", "text", "markdown"],
                    "success_rate": 1.0,
                    "use_case": "Prevent re-processing identical content"
                },
                {
                    "name": "Markdown Conversion",
                    "source": "SurfSense",
                    "description": "Convert documents to markdown for structured parsing",
                    "applicable_to": ["pdf", "html", "docx"],
                    "success_rate": 0.95,
                    "use_case": "Preserve document hierarchy and structure"
                },
                {
                    "name": "Semantic Chunking",
                    "source": "SurfSense + Perplexica",
                    "description": "Split text at semantic boundaries with overlap",
                    "applicable_to": ["all_text"],
                    "success_rate": 0.9,
                    "use_case": "Maintain context across chunk boundaries"
                }
            ],
            "concept_extraction_strategies": [
                {
                    "name": "NLP Keyword Extraction",
                    "source": "Dionysus Original",
                    "description": "Extract key concepts using NLP techniques",
                    "applicable_to": ["all_text"],
                    "success_rate": 0.85,
                    "use_case": "Identify primary document concepts"
                },
                {
                    "name": "Entity Recognition",
                    "source": "OpenNotebook",
                    "description": "Extract named entities and technical terms",
                    "applicable_to": ["academic", "technical"],
                    "success_rate": 0.8,
                    "use_case": "Identify specific entities and terminology"
                }
            ],
            "consciousness_strategies": [
                {
                    "name": "Attractor Basin Integration",
                    "source": "Dionysus Consciousness System",
                    "description": "Create basins for concept relationships",
                    "applicable_to": ["concept_processing"],
                    "success_rate": 0.9,
                    "use_case": "Enable pattern learning and emergence"
                },
                {
                    "name": "Active Inference Belief Update",
                    "source": "Dionysus Active Inference",
                    "description": "Update hierarchical beliefs based on prediction errors",
                    "applicable_to": ["learning"],
                    "success_rate": 0.95,
                    "use_case": "Minimize free energy and trigger curiosity"
                },
                {
                    "name": "ThoughtSeed Generation",
                    "source": "Dionysus ThoughtSeed System",
                    "description": "Generate thoughtseeds from concepts for propagation",
                    "applicable_to": ["concept_processing"],
                    "success_rate": 0.88,
                    "use_case": "Enable concept evolution and cross-pollination"
                }
            ],
            "curiosity_strategies": [
                {
                    "name": "Prediction Error Analysis",
                    "source": "Active Inference + R-Zero",
                    "description": "Identify high prediction error concepts for exploration",
                    "applicable_to": ["learning"],
                    "success_rate": 0.92,
                    "use_case": "Trigger curiosity-driven information seeking"
                },
                {
                    "name": "Knowledge Gap Detection",
                    "source": "Dionysus Curiosity Learning",
                    "description": "Identify gaps in knowledge tree",
                    "applicable_to": ["knowledge_graph"],
                    "success_rate": 0.87,
                    "use_case": "Plan exploration paths for missing knowledge"
                }
            ],
            "pattern_learning_strategies": [
                {
                    "name": "Co-Evolution Learning",
                    "source": "R-Zero",
                    "description": "Challenger-Solver iterative improvement",
                    "applicable_to": ["understanding"],
                    "success_rate": 0.85,
                    "use_case": "Progressively deepen document comprehension"
                },
                {
                    "name": "Iterative Refinement",
                    "source": "ASI-GO-2",
                    "description": "Refine understanding through multiple passes",
                    "applicable_to": ["complex_documents"],
                    "success_rate": 0.9,
                    "use_case": "Handle difficult or ambiguous content"
                }
            ],
            "common_issues": [
                {
                    "type": "PDF Extraction Errors",
                    "description": "Corrupted or image-based PDFs fail text extraction",
                    "prevention": "Use OCR fallback for image-based PDFs",
                    "recovery": "Log error and mark for manual review"
                },
                {
                    "type": "Duplicate Concepts",
                    "description": "Same concept extracted multiple times with variations",
                    "prevention": "Normalize concepts using embeddings similarity",
                    "recovery": "Merge similar concepts above similarity threshold"
                }
            ]
        }

    def get_relevant_strategies(self,
                               context: str,
                               category: str = None) -> List[Dict[str, Any]]:
        """Retrieve strategies relevant to the processing context"""
        relevant_strategies = []
        keywords = context.lower().split()

        # Search in specified category or all categories
        categories = [category] if category else self.knowledge.keys()

        for cat in categories:
            if cat not in self.knowledge or cat == "common_issues":
                continue

            for strategy in self.knowledge[cat]:
                # Check if strategy is applicable
                applicable_to = strategy.get("applicable_to", [])
                use_case = strategy.get("use_case", "").lower()
                description = strategy.get("description", "").lower()

                # Keyword matching
                if any(kw in use_case or kw in description for kw in keywords):
                    relevant_strategies.append(strategy)
                    continue

                # Applicability matching
                if any(app in context.lower() for app in applicable_to):
                    relevant_strategies.append(strategy)

        # Sort by success rate
        relevant_strategies.sort(key=lambda x: x.get("success_rate", 0), reverse=True)

        return relevant_strategies

    def add_insight(self, insight: Dict[str, Any]):
        """Add new insight from processing session"""
        insight["timestamp"] = datetime.now().isoformat()
        self.session_insights.append(insight)

        # Add to permanent knowledge if significant
        if insight.get("significance", 0) > 0.7:
            category = insight.get("category", "learned_patterns")
            if category not in self.knowledge:
                self.knowledge[category] = []

            self.knowledge[category].append({
                "name": insight.get("name", "Unnamed Insight"),
                "source": "Session Learning",
                "description": insight.get("description", ""),
                "applicable_to": insight.get("applicable_to", []),
                "success_rate": insight.get("success_rate", 0.5),
                "use_case": insight.get("use_case", "")
            })

            self.save_knowledge()
            logger.info(f"Significant insight added to knowledge base: {insight.get('name')}")

    def update_strategy_success_rate(self, strategy_name: str, success: bool):
        """Update strategy success rate based on actual performance"""
        for category in self.knowledge.values():
            if isinstance(category, list):
                for strategy in category:
                    if strategy.get("name") == strategy_name:
                        # Exponential moving average
                        current_rate = strategy.get("success_rate", 0.5)
                        alpha = 0.1  # Learning rate
                        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
                        strategy["success_rate"] = new_rate
                        self.save_knowledge()
                        logger.info(f"Updated {strategy_name} success rate: {new_rate:.2f}")
                        return

    def save_knowledge(self):
        """Persist knowledge to file"""
        try:
            self.knowledge_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.knowledge_file, 'w') as f:
                json.dump(self.knowledge, f, indent=2)
            logger.info("Document cognition knowledge saved")
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session insights"""
        return {
            "total_insights": len(self.session_insights),
            "insights": self.session_insights,
            "strategies_learned": list(set(
                i.get("name") for i in self.session_insights if i.get("name")
            )),
            "avg_significance": sum(i.get("significance", 0) for i in self.session_insights) /
                              len(self.session_insights) if self.session_insights else 0
        }

    def get_strategy_by_name(self, name: str) -> Dict[str, Any]:
        """Retrieve specific strategy by name"""
        for category in self.knowledge.values():
            if isinstance(category, list):
                for strategy in category:
                    if strategy.get("name") == name:
                        return strategy
        return {}
