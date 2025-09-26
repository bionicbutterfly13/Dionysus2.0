#!/usr/bin/env python3
"""
🔄 Database Migration Scripts for Unified ASI-Arch System
=========================================================

Migration scripts to move existing data from:
- MongoDB (ASI-Arch original) → Unified System
- SQLite/JSON (Context Engineering) → Enhanced Unified System
- Legacy FAISS indices → Enhanced Vector Database

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - Unified Database Migration
"""

import asyncio
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import traceback

try:
    from .unified_database import UnifiedASIArchDatabase, create_unified_database
    from .hybrid_database import HybridContextDatabase
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from unified_database import UnifiedASIArchDatabase, create_unified_database
    from hybrid_database import HybridContextDatabase

# MongoDB imports (optional)
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoClient = None

logger = logging.getLogger(__name__)

# =============================================================================
# Migration Status Tracking
# =============================================================================

class MigrationTracker:
    """Track migration progress and status"""
    
    def __init__(self, db_path: str = "extensions/context_engineering/data/migration_status.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tracking_db()
    
    def _init_tracking_db(self):
        """Initialize migration tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migration_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_type TEXT NOT NULL,
                    source_system TEXT NOT NULL,
                    target_system TEXT NOT NULL,
                    records_processed INTEGER DEFAULT 0,
                    records_successful INTEGER DEFAULT 0,
                    records_failed INTEGER DEFAULT 0,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT DEFAULT 'running',
                    error_details TEXT,
                    notes TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migration_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_log_id INTEGER,
                    record_id TEXT,
                    error_message TEXT,
                    error_traceback TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (migration_log_id) REFERENCES migration_log (id)
                )
            """)
            conn.commit()
    
    def start_migration(self, migration_type: str, source: str, target: str) -> int:
        """Start tracking a migration"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO migration_log (migration_type, source_system, target_system, started_at)
                VALUES (?, ?, ?, ?)
            """, (migration_type, source, target, datetime.now().isoformat()))
            conn.commit()
            return cursor.lastrowid
    
    def update_progress(self, migration_id: int, processed: int, successful: int, failed: int):
        """Update migration progress"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE migration_log 
                SET records_processed = ?, records_successful = ?, records_failed = ?
                WHERE id = ?
            """, (processed, successful, failed, migration_id))
            conn.commit()
    
    def complete_migration(self, migration_id: int, status: str = 'completed', notes: str = None):
        """Mark migration as completed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE migration_log 
                SET completed_at = ?, status = ?, notes = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), status, notes, migration_id))
            conn.commit()
    
    def log_error(self, migration_id: int, record_id: str, error: Exception):
        """Log migration error"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO migration_errors (migration_log_id, record_id, error_message, error_traceback, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (migration_id, record_id, str(error), traceback.format_exc(), datetime.now().isoformat()))
            conn.commit()
    
    def get_migration_status(self) -> List[Dict[str, Any]]:
        """Get all migration statuses"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM migration_log ORDER BY started_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

# =============================================================================
# MongoDB Migration
# =============================================================================

class MongoDBMigration:
    """Migrate data from MongoDB to unified system"""
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27018", 
                 database_name: str = "myapp", collection_name: str = "data_elements"):
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.collection = None
    
    def connect(self) -> bool:
        """Connect to MongoDB"""
        if not MONGODB_AVAILABLE:
            logger.warning("MongoDB not available (pymongo not installed)")
            return False
        
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.collection = self.client[self.database_name][self.collection_name]
            logger.info(f"✅ Connected to MongoDB: {self.mongo_uri}")
            return True
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}")
            return False
    
    def count_architectures(self) -> int:
        """Count architectures in MongoDB"""
        if not self.collection:
            return 0
        try:
            return self.collection.count_documents({})
        except Exception as e:
            logger.error(f"Failed to count MongoDB documents: {e}")
            return 0
    
    def get_all_architectures(self) -> List[Dict[str, Any]]:
        """Get all architectures from MongoDB"""
        if not self.collection:
            return []
        
        try:
            architectures = []
            for doc in self.collection.find({}):
                # Convert MongoDB document to architecture format
                arch_data = {
                    'id': str(doc.get('_id', doc.get('id', ''))),
                    'name': doc.get('name', ''),
                    'program': doc.get('program', ''),
                    'result': doc.get('result', {}),
                    'motivation': doc.get('motivation', ''),
                    'analysis': doc.get('analysis', ''),
                    'created_at': doc.get('created_at', datetime.now().isoformat())
                }
                architectures.append(arch_data)
            
            logger.info(f"Retrieved {len(architectures)} architectures from MongoDB")
            return architectures
            
        except Exception as e:
            logger.error(f"Failed to retrieve architectures from MongoDB: {e}")
            return []
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()

# =============================================================================
# Context Engineering Data Migration
# =============================================================================

class ContextEngineeringMigration:
    """Migrate existing context engineering data to unified system"""
    
    def __init__(self, source_path: str = "extensions/context_engineering/data"):
        self.source_path = Path(source_path)
        self.hybrid_db = None
    
    def load_existing_data(self) -> Dict[str, Any]:
        """Load existing context engineering data"""
        data = {
            'architectures': [],
            'context_streams': [],
            'attractor_basins': [],
            'graph_nodes': [],
            'graph_edges': []
        }
        
        try:
            # Load from existing hybrid database
            if (self.source_path / "context_engineering.db").exists():
                self.hybrid_db = HybridContextDatabase(str(self.source_path))
                
                # Get architectures
                with self.hybrid_db.get_sqlite_connection() as conn:
                    cursor = conn.execute("SELECT * FROM architectures")
                    for row in cursor.fetchall():
                        data['architectures'].append(dict(row))
                
                # Get graph data
                data['graph_nodes'] = [node.to_dict() for node in self.hybrid_db.graph_db.nodes.values()]
                data['graph_edges'] = [edge.to_dict() for edge in self.hybrid_db.graph_db.edges]
                
                logger.info(f"Loaded {len(data['architectures'])} architectures and {len(data['graph_nodes'])} graph nodes")
        
        except Exception as e:
            logger.error(f"Failed to load existing context engineering data: {e}")
        
        return data

# =============================================================================
# Main Migration Controller
# =============================================================================

class UnifiedDatabaseMigration:
    """Main controller for unified database migration"""
    
    def __init__(self, target_path: str = "extensions/context_engineering/data"):
        self.target_path = target_path
        self.tracker = MigrationTracker()
        self.unified_db = None
    
    async def initialize_target_database(self):
        """Initialize the target unified database"""
        self.unified_db = await create_unified_database(base_path=self.target_path)
        logger.info("✅ Target unified database initialized")
    
    async def migrate_mongodb_data(self, mongo_uri: str = "mongodb://localhost:27018") -> Dict[str, Any]:
        """Migrate data from MongoDB"""
        migration_id = self.tracker.start_migration("mongodb_architectures", "MongoDB", "UnifiedDB")
        
        mongo_migration = MongoDBMigration(mongo_uri)
        
        results = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'connected': False
        }
        
        try:
            # Connect to MongoDB
            if not mongo_migration.connect():
                self.tracker.complete_migration(migration_id, 'skipped', 'MongoDB not available')
                return results
            
            results['connected'] = True
            
            # Get all architectures
            architectures = mongo_migration.get_all_architectures()
            results['total'] = len(architectures)
            
            # Migrate each architecture
            for i, arch_data in enumerate(architectures):
                try:
                    # Enhance with default consciousness data
                    success = await self.unified_db.store_architecture(
                        arch_data, 
                        consciousness_level='MIGRATED',
                        consciousness_score=0.5
                    )
                    
                    if success:
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        self.tracker.log_error(migration_id, arch_data.get('id', f'arch_{i}'), 
                                             Exception("Failed to store architecture"))
                
                except Exception as e:
                    results['failed'] += 1
                    self.tracker.log_error(migration_id, arch_data.get('id', f'arch_{i}'), e)
                
                # Update progress every 10 items
                if (i + 1) % 10 == 0:
                    self.tracker.update_progress(migration_id, i + 1, results['successful'], results['failed'])
            
            # Final progress update
            self.tracker.update_progress(migration_id, results['total'], results['successful'], results['failed'])
            self.tracker.complete_migration(migration_id, 'completed', 
                                          f"Migrated {results['successful']}/{results['total']} architectures")
            
        except Exception as e:
            logger.error(f"MongoDB migration failed: {e}")
            self.tracker.complete_migration(migration_id, 'failed', str(e))
        
        finally:
            mongo_migration.close()
        
        return results
    
    async def migrate_context_engineering_data(self) -> Dict[str, Any]:
        """Migrate existing context engineering data"""
        migration_id = self.tracker.start_migration("context_engineering", "HybridDB", "UnifiedDB")
        
        results = {
            'architectures': {'total': 0, 'successful': 0, 'failed': 0},
            'relationships': {'total': 0, 'successful': 0, 'failed': 0}
        }
        
        try:
            # Load existing data
            ce_migration = ContextEngineeringMigration()
            existing_data = ce_migration.load_existing_data()
            
            # Migrate architectures
            architectures = existing_data['architectures']
            results['architectures']['total'] = len(architectures)
            
            for i, arch_data in enumerate(architectures):
                try:
                    success = await self.unified_db.store_architecture(arch_data)
                    if success:
                        results['architectures']['successful'] += 1
                    else:
                        results['architectures']['failed'] += 1
                
                except Exception as e:
                    results['architectures']['failed'] += 1
                    self.tracker.log_error(migration_id, arch_data.get('id', f'ce_arch_{i}'), e)
            
            # Migrate graph relationships
            edges = existing_data['graph_edges']
            results['relationships']['total'] = len(edges)
            
            for i, edge_data in enumerate(edges):
                try:
                    success = await self.unified_db.create_evolution_relationship(
                        edge_data['source_id'],
                        edge_data['target_id'],
                        {'strategy': edge_data.get('type', 'migrated'), 'strength': edge_data.get('weight', 1.0)}
                    )
                    if success:
                        results['relationships']['successful'] += 1
                    else:
                        results['relationships']['failed'] += 1
                
                except Exception as e:
                    results['relationships']['failed'] += 1
                    self.tracker.log_error(migration_id, f"edge_{i}", e)
            
            total_successful = results['architectures']['successful'] + results['relationships']['successful']
            total_items = results['architectures']['total'] + results['relationships']['total']
            
            self.tracker.update_progress(migration_id, total_items, total_successful, 
                                       results['architectures']['failed'] + results['relationships']['failed'])
            self.tracker.complete_migration(migration_id, 'completed', 
                                          f"Migrated {total_successful}/{total_items} items")
        
        except Exception as e:
            logger.error(f"Context engineering migration failed: {e}")
            self.tracker.complete_migration(migration_id, 'failed', str(e))
        
        return results
    
    async def run_full_migration(self) -> Dict[str, Any]:
        """Run complete migration process"""
        logger.info("🚀 Starting unified database migration")
        
        # Initialize target database
        await self.initialize_target_database()
        
        # Run migrations
        results = {
            'mongodb_migration': await self.migrate_mongodb_data(),
            'context_engineering_migration': await self.migrate_context_engineering_data(),
            'migration_status': self.tracker.get_migration_status()
        }
        
        # Generate summary
        total_processed = (results['mongodb_migration']['total'] + 
                          results['context_engineering_migration']['architectures']['total'] +
                          results['context_engineering_migration']['relationships']['total'])
        
        total_successful = (results['mongodb_migration']['successful'] + 
                           results['context_engineering_migration']['architectures']['successful'] +
                           results['context_engineering_migration']['relationships']['successful'])
        
        logger.info(f"✅ Migration completed: {total_successful}/{total_processed} items migrated successfully")
        
        # Get final system statistics
        results['final_system_stats'] = await self.unified_db.get_system_statistics()
        
        return results
    
    async def close(self):
        """Close database connections"""
        if self.unified_db:
            await self.unified_db.close()

# =============================================================================
# Migration Testing and Utilities
# =============================================================================

async def test_migration():
    """Test the migration system"""
    print("🧪 Testing Unified Database Migration")
    
    migration = UnifiedDatabaseMigration()
    
    try:
        # Test migration process
        results = await migration.run_full_migration()
        
        print("\n📊 Migration Results:")
        print(f"MongoDB Migration: {results['mongodb_migration']}")
        print(f"Context Engineering Migration: {results['context_engineering_migration']}")
        print(f"Final System Stats: {results['final_system_stats']}")
        
        print("\n✅ Migration test completed!")
        
    finally:
        await migration.close()

def generate_migration_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive migration report"""
    report = f"""
# Unified Database Migration Report
Generated: {datetime.now().isoformat()}

## Migration Summary

### MongoDB Migration
- Connected: {results['mongodb_migration']['connected']}
- Total Records: {results['mongodb_migration']['total']}
- Successful: {results['mongodb_migration']['successful']}
- Failed: {results['mongodb_migration']['failed']}

### Context Engineering Migration
- Architectures: {results['context_engineering_migration']['architectures']['successful']}/{results['context_engineering_migration']['architectures']['total']}
- Relationships: {results['context_engineering_migration']['relationships']['successful']}/{results['context_engineering_migration']['relationships']['total']}

### Final System Status
- Neo4j Available: {results['final_system_stats']['neo4j_available']}
- Vector Embeddings: {results['final_system_stats']['vector_embeddings_count']}
- AutoSchemaKG Available: {results['final_system_stats']['autoschema_kg_available']}

## Migration Log
"""
    
    for log_entry in results.get('migration_status', []):
        report += f"""
### {log_entry['migration_type']} ({log_entry['source_system']} → {log_entry['target_system']})
- Status: {log_entry['status']}
- Records: {log_entry['records_successful']}/{log_entry['records_processed']}
- Started: {log_entry['started_at']}
- Completed: {log_entry['completed_at']}
"""
    
    return report

if __name__ == "__main__":
    asyncio.run(test_migration())


