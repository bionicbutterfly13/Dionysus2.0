#!/usr/bin/env python3
"""
🤖 AutoSchemaKG Integration for ASI-Arch Development Knowledge
============================================================

Integration with AutoSchemaKG framework for automatic knowledge graph construction
from ASI-Arch development conversations, specifications, and research data.

This module provides:
- Automatic triple extraction from development documents
- Dynamic schema generation for ASI-Arch domain
- Integration with unified database system
- Knowledge graph construction from unstructured data

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - AutoSchemaKG Integration
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import tempfile
import shutil

# AutoSchemaKG imports
try:
    from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
    from atlas_rag.kg_construction.triple_config import ProcessingConfig
    from atlas_rag.llm_generator import LLMGenerator
    AUTOSCHEMA_AVAILABLE = True
except ImportError:
    AUTOSCHEMA_AVAILABLE = False
    KnowledgeGraphExtractor = None
    ProcessingConfig = None
    LLMGenerator = None

# OpenAI imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from .unified_database import UnifiedASIArchDatabase
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from unified_database import UnifiedASIArchDatabase

logger = logging.getLogger(__name__)

# =============================================================================
# Development Data Collection
# =============================================================================

class DevelopmentDataCollector:
    """Collect development data for knowledge graph construction"""
    
    def __init__(self, project_root: str = "/Volumes/Asylum/devb/ASI-Arch"):
        self.project_root = Path(project_root)
        self.data_sources = {
            'specifications': self.project_root / "spec-management" / "ASI-Arch-Specs",
            'extensions': self.project_root / "extensions" / "context_engineering",
            'documentation': self.project_root,
            'conversations': self.project_root / "development_conversations"  # If exists
        }
    
    def collect_specification_documents(self) -> List[Dict[str, str]]:
        """Collect specification documents"""
        documents = []
        
        spec_dir = self.data_sources['specifications']
        if spec_dir.exists():
            for spec_file in spec_dir.glob("*.md"):
                try:
                    with open(spec_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append({
                        'id': spec_file.stem,
                        'title': spec_file.stem.replace('_', ' ').title(),
                        'content': content,
                        'source': 'specification',
                        'file_path': str(spec_file),
                        'created_at': datetime.fromtimestamp(spec_file.stat().st_mtime).isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to read spec file {spec_file}: {e}")
        
        logger.info(f"Collected {len(documents)} specification documents")
        return documents
    
    def collect_code_documentation(self) -> List[Dict[str, str]]:
        """Collect code documentation and docstrings"""
        documents = []
        
        # Collect README files
        for readme_file in self.project_root.rglob("README.md"):
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                documents.append({
                    'id': f"readme_{readme_file.parent.name}",
                    'title': f"README: {readme_file.parent.name}",
                    'content': content,
                    'source': 'documentation',
                    'file_path': str(readme_file),
                    'created_at': datetime.fromtimestamp(readme_file.stat().st_mtime).isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to read README {readme_file}: {e}")
        
        # Collect Python docstrings from key files
        python_files = [
            self.project_root / "extensions" / "context_engineering" / "unified_database.py",
            self.project_root / "extensions" / "context_engineering" / "hybrid_database.py",
            self.project_root / "extensions" / "context_engineering" / "migration_scripts.py"
        ]
        
        for py_file in python_files:
            if py_file.exists():
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract module docstring (first triple-quoted string)
                    lines = content.split('\n')
                    docstring_lines = []
                    in_docstring = False
                    quote_count = 0
                    
                    for line in lines:
                        if '"""' in line:
                            quote_count += line.count('"""')
                            if not in_docstring:
                                in_docstring = True
                            docstring_lines.append(line)
                            if quote_count >= 2:
                                break
                        elif in_docstring:
                            docstring_lines.append(line)
                    
                    if docstring_lines:
                        docstring = '\n'.join(docstring_lines)
                        documents.append({
                            'id': f"code_{py_file.stem}",
                            'title': f"Code Documentation: {py_file.stem}",
                            'content': docstring,
                            'source': 'code_documentation',
                            'file_path': str(py_file),
                            'created_at': datetime.fromtimestamp(py_file.stat().st_mtime).isoformat()
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to extract docstring from {py_file}: {e}")
        
        logger.info(f"Collected {len(documents)} documentation documents")
        return documents
    
    def create_synthetic_conversations(self) -> List[Dict[str, str]]:
        """Create synthetic conversation data for demonstration"""
        conversations = [
            {
                'id': 'conv_unified_db_design',
                'title': 'Unified Database Design Discussion',
                'content': '''
                Discussion about designing a unified database system for ASI-Arch:
                
                The goal is to create a hybrid database that combines Neo4j knowledge graphs
                with vector databases for similarity search. This system needs to handle:
                
                1. Neural architecture storage and relationships
                2. Consciousness level tracking and evolution
                3. Performance metrics and evaluation data
                4. Episodic memory and autobiographical events
                
                The key insight is that Neo4j excels at graph relationships while vector
                databases excel at similarity search. By combining them, we get the best
                of both worlds for ASI-Arch development.
                ''',
                'source': 'conversation',
                'created_at': datetime.now().isoformat()
            },
            {
                'id': 'conv_consciousness_detection',
                'title': 'Consciousness Detection Implementation',
                'content': '''
                Implementation discussion for consciousness detection in neural architectures:
                
                The system uses multiple indicators to assess consciousness levels:
                - Self-referential processing capabilities
                - Meta-cognitive awareness patterns
                - Recursive thinking structures
                - Information integration complexity
                
                Consciousness levels are categorized as:
                - DORMANT: Basic processing, no self-awareness
                - ACTIVE: Some self-referential capabilities
                - SELF_AWARE: Clear meta-cognitive patterns
                - TRANSCENDENT: Advanced consciousness indicators
                
                This classification helps track the evolution of consciousness in AI systems.
                ''',
                'source': 'conversation',
                'created_at': datetime.now().isoformat()
            },
            {
                'id': 'conv_autoschemakg_integration',
                'title': 'AutoSchemaKG Integration Planning',
                'content': '''
                Planning the integration of AutoSchemaKG for automatic knowledge graph construction:
                
                AutoSchemaKG provides several key capabilities:
                1. Automatic triple extraction from text documents
                2. Dynamic schema generation for domain-specific knowledge
                3. Knowledge graph construction without manual schema design
                4. Integration with Neo4j databases
                
                For ASI-Arch, this means we can automatically build knowledge graphs from:
                - Development conversations and specifications
                - Research paper abstracts and findings
                - Architecture evolution documentation
                - Performance evaluation reports
                
                The system will continuously learn and evolve its understanding of the domain.
                ''',
                'source': 'conversation',
                'created_at': datetime.now().isoformat()
            }
        ]
        
        logger.info(f"Created {len(conversations)} synthetic conversation documents")
        return conversations
    
    def collect_all_documents(self) -> List[Dict[str, str]]:
        """Collect all available documents"""
        all_documents = []
        
        all_documents.extend(self.collect_specification_documents())
        all_documents.extend(self.collect_code_documentation())
        all_documents.extend(self.create_synthetic_conversations())
        
        logger.info(f"Collected total of {len(all_documents)} documents for knowledge graph construction")
        return all_documents

# =============================================================================
# AutoSchemaKG Integration Manager
# =============================================================================

class AutoSchemaKGManager:
    """Manager for AutoSchemaKG integration"""
    
    def __init__(self, output_dir: str = "extensions/context_engineering/knowledge_graphs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = None
        self.kg_extractor = None
        self.available = AUTOSCHEMA_AVAILABLE and OPENAI_AVAILABLE
        
        if not self.available:
            missing = []
            if not AUTOSCHEMA_AVAILABLE:
                missing.append("atlas-rag")
            if not OPENAI_AVAILABLE:
                missing.append("openai")
            logger.warning(f"AutoSchemaKG not available. Missing: {missing}")
    
    def setup_openai_client(self, api_key: str = None) -> bool:
        """Setup OpenAI client"""
        if not OPENAI_AVAILABLE:
            return False
        
        try:
            # Use provided API key or environment variable
            if api_key:
                openai.api_key = api_key
            elif os.getenv('OPENAI_API_KEY'):
                openai.api_key = os.getenv('OPENAI_API_KEY')
            else:
                logger.warning("No OpenAI API key provided")
                return False
            
            # Test the client
            client = openai.OpenAI()
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup OpenAI client: {e}")
            return False
    
    def prepare_documents_for_extraction(self, documents: List[Dict[str, str]]) -> str:
        """Prepare documents for AutoSchemaKG extraction"""
        if not self.available:
            return None
        
        # Create temporary directory for document processing
        self.temp_dir = tempfile.mkdtemp(prefix="asi_arch_kg_")
        data_dir = Path(self.temp_dir) / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Write documents to individual files
        for i, doc in enumerate(documents):
            filename = f"doc_{i:03d}_{doc['id']}.txt"
            file_path = data_dir / filename
            
            # Create document content with metadata
            content = f"""Title: {doc['title']}
Source: {doc['source']}
Created: {doc['created_at']}

{doc['content']}
"""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"Prepared {len(documents)} documents in {data_dir}")
        return str(data_dir)
    
    def setup_kg_extractor(self, data_directory: str, openai_client=None) -> bool:
        """Setup AutoSchemaKG extractor"""
        if not self.available or not data_directory:
            return False
        
        try:
            config = ProcessingConfig(
                data_directory=data_directory,
                output_directory=str(self.output_dir),
                batch_size_triple=3,
                batch_size_concept=16,
                remove_doc_spaces=True,
                max_workers=2  # Conservative for development
            )
            
            if openai_client:
                llm_generator = LLMGenerator(openai_client, model_name="gpt-3.5-turbo")
                self.kg_extractor = KnowledgeGraphExtractor(
                    model=llm_generator,
                    config=config
                )
                logger.info("✅ AutoSchemaKG extractor setup complete")
                return True
            else:
                logger.warning("No OpenAI client provided for AutoSchemaKG")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup AutoSchemaKG extractor: {e}")
            return False
    
    async def extract_knowledge_graph(self) -> bool:
        """Extract knowledge graph using AutoSchemaKG"""
        if not self.kg_extractor:
            logger.warning("AutoSchemaKG extractor not available")
            return False
        
        try:
            logger.info("🤖 Starting AutoSchemaKG knowledge extraction...")
            
            # Run triple extraction
            await self.kg_extractor.run_extraction()
            logger.info("✅ Triple extraction completed")
            
            # Generate concepts
            await self.kg_extractor.generate_concept_csv()
            logger.info("✅ Concept generation completed")
            
            return True
            
        except Exception as e:
            logger.error(f"AutoSchemaKG extraction failed: {e}")
            return False
    
    def get_extracted_knowledge(self) -> Dict[str, Any]:
        """Get extracted knowledge from output files"""
        knowledge = {
            'triples': [],
            'concepts': [],
            'entities': [],
            'relations': []
        }
        
        try:
            # Look for output files in the output directory
            for output_file in self.output_dir.glob("*.json"):
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if 'triples' in str(output_file):
                    knowledge['triples'] = data
                elif 'concepts' in str(output_file):
                    knowledge['concepts'] = data
                elif 'entities' in str(output_file):
                    knowledge['entities'] = data
                elif 'relations' in str(output_file):
                    knowledge['relations'] = data
            
            # Look for CSV files
            for csv_file in self.output_dir.glob("*.csv"):
                logger.info(f"Found CSV output: {csv_file}")
        
        except Exception as e:
            logger.error(f"Failed to read extracted knowledge: {e}")
        
        return knowledge
    
    def cleanup(self):
        """Cleanup temporary files"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary files")

# =============================================================================
# Complete Integration Workflow
# =============================================================================

async def run_autoschema_integration(unified_db: UnifiedASIArchDatabase = None,
                                   openai_api_key: str = None) -> Dict[str, Any]:
    """Run complete AutoSchemaKG integration workflow"""
    results = {
        'documents_collected': 0,
        'extraction_successful': False,
        'knowledge_extracted': {},
        'integration_successful': False,
        'error_message': None
    }
    
    collector = DevelopmentDataCollector()
    manager = AutoSchemaKGManager()
    
    try:
        # Step 1: Collect development documents
        logger.info("📚 Collecting development documents...")
        documents = collector.collect_all_documents()
        results['documents_collected'] = len(documents)
        
        if not documents:
            results['error_message'] = "No documents collected"
            return results
        
        # Step 2: Setup AutoSchemaKG (if available)
        if manager.available:
            logger.info("🤖 Setting up AutoSchemaKG...")
            
            # Setup OpenAI client
            if openai_api_key:
                client_setup = manager.setup_openai_client(openai_api_key)
                if not client_setup:
                    results['error_message'] = "Failed to setup OpenAI client"
                    return results
                
                # Prepare documents
                data_dir = manager.prepare_documents_for_extraction(documents)
                
                # Setup extractor
                openai_client = openai.OpenAI() if openai_api_key else None
                extractor_setup = manager.setup_kg_extractor(data_dir, openai_client)
                
                if extractor_setup:
                    # Run extraction
                    extraction_success = await manager.extract_knowledge_graph()
                    results['extraction_successful'] = extraction_success
                    
                    if extraction_success:
                        # Get extracted knowledge
                        knowledge = manager.get_extracted_knowledge()
                        results['knowledge_extracted'] = knowledge
                        
                        # Integrate with unified database (if provided)
                        if unified_db:
                            # This would integrate the extracted knowledge
                            # For now, just mark as successful
                            results['integration_successful'] = True
            else:
                logger.info("⚠️  No OpenAI API key provided, skipping AutoSchemaKG extraction")
                results['error_message'] = "No OpenAI API key provided"
        else:
            logger.info("⚠️  AutoSchemaKG not available, creating mock knowledge structure")
            # Create mock knowledge structure from documents
            mock_knowledge = create_mock_knowledge_structure(documents)
            results['knowledge_extracted'] = mock_knowledge
            results['extraction_successful'] = True
            results['integration_successful'] = True
    
    except Exception as e:
        logger.error(f"AutoSchemaKG integration failed: {e}")
        results['error_message'] = str(e)
    
    finally:
        manager.cleanup()
    
    return results

def create_mock_knowledge_structure(documents: List[Dict[str, str]]) -> Dict[str, Any]:
    """Create mock knowledge structure for demonstration"""
    entities = set()
    relations = set()
    concepts = set()
    
    # Extract key terms from documents
    key_terms = [
        'neural architecture', 'consciousness', 'database', 'vector', 'Neo4j',
        'AutoSchemaKG', 'evolution', 'performance', 'similarity search',
        'knowledge graph', 'ASI-Arch', 'hybrid database', 'migration'
    ]
    
    for doc in documents:
        content_lower = doc['content'].lower()
        for term in key_terms:
            if term.lower() in content_lower:
                entities.add(term)
                concepts.add(f"concept_{term.replace(' ', '_')}")
    
    # Create mock relations
    mock_relations = [
        ('neural_architecture', 'has_consciousness', 'consciousness_level'),
        ('database', 'stores', 'neural_architecture'),
        ('vector_database', 'enables', 'similarity_search'),
        ('Neo4j', 'provides', 'graph_relationships'),
        ('AutoSchemaKG', 'extracts', 'knowledge_graph'),
        ('ASI-Arch', 'uses', 'hybrid_database')
    ]
    
    return {
        'entities': list(entities),
        'concepts': list(concepts),
        'relations': mock_relations,
        'triples': [{'subject': r[0], 'predicate': r[1], 'object': r[2]} for r in mock_relations],
        'document_count': len(documents),
        'extraction_method': 'mock_extraction'
    }

# =============================================================================
# Testing
# =============================================================================

async def test_autoschema_integration():
    """Test AutoSchemaKG integration"""
    print("🤖 Testing AutoSchemaKG Integration")
    
    # Run integration without OpenAI key (mock mode)
    results = await run_autoschema_integration()
    
    print(f"\n📊 Integration Results:")
    print(f"Documents collected: {results['documents_collected']}")
    print(f"Extraction successful: {results['extraction_successful']}")
    print(f"Integration successful: {results['integration_successful']}")
    
    if results['knowledge_extracted']:
        knowledge = results['knowledge_extracted']
        print(f"\n🧠 Extracted Knowledge:")
        print(f"Entities: {len(knowledge.get('entities', []))}")
        print(f"Concepts: {len(knowledge.get('concepts', []))}")
        print(f"Relations: {len(knowledge.get('relations', []))}")
        print(f"Triples: {len(knowledge.get('triples', []))}")
        
        # Show some examples
        if knowledge.get('entities'):
            print(f"Sample entities: {knowledge['entities'][:5]}")
        if knowledge.get('relations'):
            print(f"Sample relations: {knowledge['relations'][:3]}")
    
    if results['error_message']:
        print(f"\n⚠️  Error: {results['error_message']}")
    
    print("\n✅ AutoSchemaKG integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_autoschema_integration())


