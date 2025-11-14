"""
Skills Manager - Initialize and manage skills database via Dionysus consciousness system

This service:
1. Scans the skills library (/Volumes/Asylum/skills-library/)
2. Processes each skill as a document through Daedalus
3. Stores skills in Neo4j via consciousness-enhanced processing
4. Creates ThoughtSeeds and AttractorBasins for skill relationships
5. Builds Claude Code skill index

Integration: Uses Daedalus → DocumentProcessingGraph → AutoSchemaKG → Neo4j
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import io

logger = logging.getLogger(__name__)


class SkillsManager:
    """
    Manage skills database through Dionysus consciousness system.

    All skills are processed as documents through Daedalus to ensure:
    - Consciousness-enhanced concept extraction
    - Attractor basin formation
    - ThoughtSeed generation for cross-skill relationships
    - Proper knowledge graph integration
    """

    def __init__(self,
                 skills_library_path: str = "/Volumes/Asylum/skills-library",
                 daedalus_instance=None):
        """
        Initialize Skills Manager.

        Args:
            skills_library_path: Path to skills library
            daedalus_instance: Optional Daedalus instance (creates new if None)
        """
        self.skills_library = Path(skills_library_path)
        self.claude_skills = Path.home() / ".claude" / "skills"

        # Import Daedalus lazily to avoid circular imports
        if daedalus_instance:
            self.daedalus = daedalus_instance
        else:
            from .daedalus import Daedalus
            self.daedalus = Daedalus()

        logger.info(f"SkillsManager initialized with library: {self.skills_library}")

    def initialize_skills_database(self) -> Dict[str, Any]:
        """
        Initialize complete skills database.

        Process:
        1. Scan all skill categories
        2. Process each skill through Daedalus consciousness pipeline
        3. Store in Neo4j with full graph relationships
        4. Create Claude Code skill index

        Returns:
            Status dict with initialization results
        """
        logger.info("🔧 Initializing skills database through consciousness system...")

        # Ensure Claude skills directory exists
        self.claude_skills.mkdir(parents=True, exist_ok=True)

        skills_discovered = []
        processing_results = []

        # Scan all skill categories
        categories = ["official", "community", "personal/superpowers", "project-specific"]
        for category in categories:
            category_path = self.skills_library / category
            if category_path.exists():
                logger.info(f"📂 Scanning category: {category}")
                category_skills = self._scan_category(category_path, category)
                skills_discovered.extend(category_skills)

                # Process each skill through Daedalus
                for skill in category_skills:
                    result = self._process_skill_through_consciousness(skill)
                    processing_results.append(result)

        # Create Claude Code index
        index_path = self._create_claude_index(skills_discovered, processing_results)

        result = {
            "status": "success",
            "skills_discovered": len(skills_discovered),
            "skills_processed": len(processing_results),
            "categories_scanned": len(categories),
            "index_path": str(index_path),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"✅ Skills database initialized: {result}")
        return result

    def _scan_category(self, path: Path, category: str) -> List[Dict[str, Any]]:
        """
        Scan a skill category directory.

        Args:
            path: Category directory path
            category: Category name

        Returns:
            List of skill metadata dicts
        """
        skills = []

        # Look for markdown files (skill definitions)
        for skill_file in path.rglob("*.md"):
            if skill_file.is_file() and skill_file.name != "README.md":
                try:
                    skill_meta = self._extract_skill_metadata(skill_file, category)
                    if skill_meta:
                        skills.append(skill_meta)
                        logger.debug(f"  ✓ Found skill: {skill_meta['name']}")
                except Exception as e:
                    logger.warning(f"  ⚠ Could not process {skill_file}: {e}")

        return skills

    def _extract_skill_metadata(self, skill_file: Path, category: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from skill file.

        Args:
            skill_file: Path to skill markdown file
            category: Skill category

        Returns:
            Skill metadata dict or None
        """
        try:
            content = skill_file.read_text(encoding='utf-8')

            # Extract title from first heading or filename
            title_line = [line for line in content.split('\n') if line.startswith('#')]
            if title_line:
                title = title_line[0].lstrip('#').strip()
            else:
                title = skill_file.stem.replace('_', ' ').replace('-', ' ').title()

            return {
                "name": skill_file.stem,
                "title": title,
                "category": category,
                "file_path": str(skill_file),
                "content": content,
                "size": len(content),
                "modified": datetime.fromtimestamp(skill_file.stat().st_mtime).isoformat()
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {skill_file}: {e}")
            return None

    def _process_skill_through_consciousness(self, skill: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process skill through Daedalus consciousness system.

        This creates:
        - Document node with skill content
        - Extracted concepts (5-level hierarchy)
        - Attractor basins for skill domains
        - ThoughtSeeds for cross-skill relationships

        Args:
            skill: Skill metadata dict

        Returns:
            Processing result from Daedalus
        """
        logger.info(f"🧠 Processing skill through consciousness: {skill['title']}")

        # Prepare skill as document
        skill_document = f"""# Skill: {skill['title']}

Category: {skill['category']}
Source: {skill['file_path']}
Last Modified: {skill['modified']}

---

{skill['content']}

---

Tags: #skill #{skill['category'].replace('/', '-')} #knowledge #capability
""".encode('utf-8')

        # Process through Daedalus
        try:
            result = self.daedalus.receive_perceptual_information(
                data=io.BytesIO(skill_document),
                tags=["skill", skill['category'], "capability", "knowledge"],
                max_iterations=2,  # Skills don't need as much iteration
                quality_threshold=0.65
            )

            logger.info(f"  ✓ Processed: {skill['title']} (Quality: {result.get('quality', {}).get('scores', {}).get('overall', 'N/A')})")

            return {
                "skill_name": skill['name'],
                "status": result.get('status'),
                "document_id": result.get('document', {}).get('document_id'),
                "concepts_extracted": len(result.get('consciousness', {}).get('concepts', [])),
                "basins_created": len(result.get('consciousness', {}).get('basins', [])),
                "thoughtseeds_generated": len(result.get('consciousness', {}).get('thoughtseeds', [])),
                "quality_score": result.get('quality', {}).get('scores', {}).get('overall', 0.0)
            }
        except Exception as e:
            logger.error(f"  ✗ Error processing {skill['title']}: {e}")
            return {
                "skill_name": skill['name'],
                "status": "error",
                "error": str(e)
            }

    def _create_claude_index(self, skills: List[Dict], processing_results: List[Dict]) -> Path:
        """
        Create skill index for Claude Code.

        Args:
            skills: List of skill metadata
            processing_results: List of Daedalus processing results

        Returns:
            Path to created index file
        """
        # Combine skill metadata with processing results
        indexed_skills = []
        for skill, result in zip(skills, processing_results):
            indexed_skills.append({
                "name": skill['name'],
                "title": skill['title'],
                "category": skill['category'],
                "file_path": skill['file_path'],
                "document_id": result.get('document_id'),
                "concepts_extracted": result.get('concepts_extracted', 0),
                "quality_score": result.get('quality_score', 0.0),
                "status": result.get('status', 'unknown')
            })

        # Create index
        index = {
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "total_skills": len(indexed_skills),
            "processed_via": "Dionysus Consciousness System (Daedalus → DocumentProcessingGraph → AutoSchemaKG → Neo4j)",
            "skills": indexed_skills,
            "categories": list(set(s['category'] for s in skills)),
            "stats": {
                "successful": sum(1 for r in processing_results if r.get('status') == 'received'),
                "failed": sum(1 for r in processing_results if r.get('status') == 'error'),
                "total_concepts": sum(r.get('concepts_extracted', 0) for r in processing_results),
                "average_quality": sum(r.get('quality_score', 0.0) for r in processing_results) / len(processing_results) if processing_results else 0.0
            }
        }

        # Write index
        index_path = self.claude_skills / "index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

        logger.info(f"📝 Created skill index: {index_path}")
        return index_path

    def get_skill_status(self) -> Dict[str, Any]:
        """
        Get current skills database status.

        Returns:
            Status dict with skill counts and index info
        """
        index_path = self.claude_skills / "index.json"

        if not index_path.exists():
            return {
                "status": "not_initialized",
                "skills_count": 0,
                "message": "Skills database not initialized. Run initialize_skills_database()"
            }

        with open(index_path, 'r') as f:
            index = json.load(f)

        return {
            "status": "initialized",
            "skills_count": index['total_skills'],
            "categories": index['categories'],
            "stats": index['stats'],
            "last_updated": index['created'],
            "index_path": str(index_path)
        }
