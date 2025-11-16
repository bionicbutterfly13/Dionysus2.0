# Design: OpenSpec Specification Ingestion Pipeline

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│               OpenSpec File System (Markdown Specs)                  │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ openspec/specs/                                                  │ │
│ │   ├── document-processing/                                       │ │
│ │   │   ├── spec.md           (Requirements)                       │ │
│ │   │   └── design.md         (Implementation patterns)            │ │
│ │   ├── clause-multi-agent/                                        │ │
│ │   │   ├── spec.md                                                │ │
│ │   │   └── design.md                                              │ │
│ │   └── knowledge-graph/                                           │ │
│ │       ├── spec.md                                                │ │
│ │       └── design.md                                              │ │
│ └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │ File Scan
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Ingestion Script (New Component)                    │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ backend/scripts/ingest_openspec_specs.py                         │ │
│ │                                                                  │ │
│ │ 1. Scan openspec/specs/**/spec.md, design.md                    │ │
│ │ 2. Extract metadata:                                            │ │
│ │    - capability = parent directory name                         │ │
│ │    - spec_type = "spec" or "design"                             │ │
│ │    - content_hash = SHA-256(file_content)                       │ │
│ │ 3. Read file content                                            │ │
│ │ 4. POST to /api/documents with multipart/form-data              │ │
│ └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP POST
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Existing Document Processing Pipeline                   │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ POST /api/documents (backend/src/api/routes/documents.py)       │ │
│ │   ↓                                                              │ │
│ │ Daedalus Gateway (receive_perceptual_information)                │ │
│ │   ↓                                                              │ │
│ │ DocumentProcessingGraph (6 nodes)                                │ │
│ │   ├── Extract & Process                                          │ │
│ │   ├── Research (ASI-GO-2)                                        │ │
│ │   ├── Consciousness (5-level concepts)                           │ │
│ │   ├── Analyze Results                                            │ │
│ │   ├── Refine Processing                                          │ │
│ │   └── Finalize Output                                            │ │
│ │   ↓                                                              │ │
│ │ DocumentRepository (persist_document)                            │ │
│ └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │ Cypher Write
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Neo4j Knowledge Graph                         │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ Nodes:                                                           │ │
│ │   (d:Document:Specification {                                   │ │
│ │     id: uuid,                                                    │ │
│ │     title: "Document Processing",                               │ │
│ │     content_hash: "abc123...",                                  │ │
│ │     source_type: "openspec",                                    │ │
│ │     capability: "document-processing",                          │ │
│ │     spec_type: "spec",                                          │ │
│ │     version: "1.0"                                              │ │
│ │   })                                                            │ │
│ │   (c:Concept:Requirement {title: "Import OpenSpec..."})         │ │
│ │   (s:Concept:Scenario {given: "...", when: "...", then: "..."})│ │
│ │   (t:ThoughtSeed {content: "Authentication patterns..."})       │ │
│ │   (b:AttractorBasin {name: "Specification Architecture"})       │ │
│ │                                                                  │ │
│ │ Relationships:                                                   │ │
│ │   (d)-[:HAS_REQUIREMENT]->(c)                                   │ │
│ │   (c)-[:HAS_SCENARIO]->(s)                                      │ │
│ │   (d)-[:HAS_THOUGHTSEED]->(t)                                   │ │
│ │   (d)-[:BELONGS_TO_BASIN]->(b)                                  │ │
│ │   (d1:Specification)-[:SIMILAR_TO]->(d2:Specification)          │ │
│ └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Ingestion Script

**Location**: `backend/scripts/ingest_openspec_specs.py`

**Implementation**:
```python
import os
import hashlib
import requests
from pathlib import Path
from typing import List, Dict

class OpenSpecIngester:
    def __init__(self, api_base_url: str = "http://localhost:9127"):
        self.api_base_url = api_base_url
        self.specs_dir = Path("openspec/specs")

    def scan_specs(self, capability: str = None) -> List[Dict]:
        """Scan openspec/specs/ for spec.md and design.md files."""
        specs = []

        if capability:
            # Scan single capability
            cap_dir = self.specs_dir / capability
            specs.extend(self._scan_directory(cap_dir, capability))
        else:
            # Scan all capabilities
            for cap_dir in self.specs_dir.iterdir():
                if cap_dir.is_dir():
                    capability_name = cap_dir.name
                    specs.extend(self._scan_directory(cap_dir, capability_name))

        return specs

    def _scan_directory(self, cap_dir: Path, capability: str) -> List[Dict]:
        """Scan a single capability directory."""
        specs = []

        for file_path in cap_dir.glob("*.md"):
            spec_type = file_path.stem  # "spec" or "design"
            content = file_path.read_text()
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            specs.append({
                "file_path": str(file_path),
                "capability": capability,
                "spec_type": spec_type,
                "content": content,
                "content_hash": content_hash
            })

        return specs

    def ingest_spec(self, spec_data: Dict) -> Dict:
        """Ingest a single spec via POST /api/documents."""
        # Create multipart/form-data
        files = {
            "file": (
                f"{spec_data['capability']}-{spec_data['spec_type']}.md",
                spec_data["content"],
                "text/markdown"
            )
        }

        # Add metadata as form fields
        data = {
            "source_type": "openspec",
            "capability": spec_data["capability"],
            "spec_type": spec_data["spec_type"],
            "content_hash": spec_data["content_hash"],
            "version": "1.0"
        }

        response = requests.post(
            f"{self.api_base_url}/api/documents",
            files=files,
            data=data
        )

        return {
            "file_path": spec_data["file_path"],
            "status_code": response.status_code,
            "response": response.json()
        }

    def ingest_all(self, capability: str = None, dry_run: bool = False):
        """Ingest all specs or specific capability."""
        specs = self.scan_specs(capability)

        print(f"Found {len(specs)} spec files to ingest")

        results = []
        for spec in specs:
            if dry_run:
                print(f"[DRY RUN] Would ingest: {spec['file_path']}")
                continue

            print(f"Ingesting: {spec['file_path']}... ", end="")
            result = self.ingest_spec(spec)

            if result["status_code"] == 200:
                print("✓ Success")
            elif result["status_code"] == 409:
                print("⊘ Duplicate (already ingested)")
            else:
                print(f"✗ Failed ({result['status_code']})")

            results.append(result)

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest OpenSpec specifications into Neo4j knowledge graph"
    )
    parser.add_argument(
        "--capability",
        help="Ingest specific capability (e.g., document-processing)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all capabilities"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without ingesting"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:9127",
        help="API base URL (default: http://localhost:9127)"
    )

    args = parser.parse_args()

    ingester = OpenSpecIngester(api_base_url=args.api_url)

    if args.all:
        results = ingester.ingest_all(dry_run=args.dry_run)
    elif args.capability:
        results = ingester.ingest_all(
            capability=args.capability,
            dry_run=args.dry_run
        )
    else:
        print("Error: Specify --all or --capability <name>")
        return 1

    # Summary
    if not args.dry_run:
        success = sum(1 for r in results if r["status_code"] == 200)
        duplicates = sum(1 for r in results if r["status_code"] == 409)
        failed = sum(1 for r in results if r["status_code"] not in [200, 409])

        print(f"\nSummary:")
        print(f"  ✓ Success: {success}")
        print(f"  ⊘ Duplicates: {duplicates}")
        print(f"  ✗ Failed: {failed}")

    return 0


if __name__ == "__main__":
    exit(main())
```

### 2. API Endpoint Enhancement

**Existing**: `POST /api/documents` (backend/src/api/routes/documents.py)

**Enhancement**: Accept additional metadata fields
```python
class DocumentUploadRequest(BaseModel):
    # Existing fields
    file: UploadFile
    url: Optional[str] = None

    # NEW: OpenSpec-specific metadata
    source_type: Optional[str] = None  # "openspec"
    capability: Optional[str] = None  # "document-processing"
    spec_type: Optional[str] = None  # "spec" or "design"
    content_hash: Optional[str] = None  # SHA-256 hash
    version: Optional[str] = "1.0"
```

**Storage**: Pass metadata to DocumentRepository
```python
async def upload_document(file: UploadFile, metadata: Dict):
    # Extract content
    content = await file.read()

    # Call Daedalus
    result = await daedalus.receive_perceptual_information(
        content=content.decode(),
        metadata={
            **metadata,  # Include source_type, capability, etc.
            "filename": file.filename
        }
    )

    return result
```

### 3. Neo4j Schema Enhancements

**Node Labels**:
```cypher
CREATE CONSTRAINT spec_id IF NOT EXISTS
FOR (s:Specification) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT design_id IF NOT EXISTS
FOR (d:DesignDocument) REQUIRE d.id IS UNIQUE;

CREATE INDEX spec_capability IF NOT EXISTS
FOR (s:Specification) ON (s.capability);

CREATE INDEX spec_content_hash IF NOT EXISTS
FOR (s:Specification) ON (s.content_hash);
```

**Metadata Properties**:
```cypher
(:Document:Specification {
  id: "uuid",
  title: "Document Processing",
  content_hash: "abc123...",
  source_type: "openspec",
  capability: "document-processing",
  spec_type: "spec",
  version: "1.0",
  extracted_text: "...",
  summary: "...",
  created_at: datetime(),
  updated_at: datetime()
})
```

### 4. Requirement & Scenario Extraction

**Parser Logic** (in DocumentProcessingGraph):
```python
def extract_requirements(spec_content: str) -> List[Dict]:
    """Extract requirements from ### Requirement: headers."""
    requirements = []
    lines = spec_content.split('\n')

    current_req = None
    for line in lines:
        if line.startswith("### Requirement:"):
            if current_req:
                requirements.append(current_req)

            title = line.replace("### Requirement:", "").strip()
            current_req = {
                "title": title,
                "scenarios": []
            }

        elif line.startswith("#### Scenario:") and current_req:
            scenario_title = line.replace("#### Scenario:", "").strip()
            current_req["scenarios"].append({
                "title": scenario_title,
                "given": "",
                "when": "",
                "then": ""
            })

    if current_req:
        requirements.append(current_req)

    return requirements
```

**Neo4j Storage**:
```cypher
// Create requirement nodes
MATCH (s:Specification {id: $spec_id})
UNWIND $requirements AS req
CREATE (r:Concept:Requirement {
  id: randomUUID(),
  title: req.title
})
CREATE (s)-[:HAS_REQUIREMENT]->(r)

// Create scenario nodes
FOREACH (scenario IN req.scenarios |
  CREATE (sc:Concept:Scenario {
    id: randomUUID(),
    title: scenario.title,
    given: scenario.given,
    when: scenario.when,
    then: scenario.then
  })
  CREATE (r)-[:HAS_SCENARIO]->(sc)
)
```

## Data Flow

### Ingestion Flow
1. **Scan**: Script finds all spec.md/design.md files
2. **Hash**: Calculate SHA-256 content hash
3. **POST**: Send to /api/documents with metadata
4. **Process**: Daedalus → LangGraph → DocumentRepository
5. **Store**: Document node created in Neo4j
6. **Extract**: Requirements/scenarios parsed and linked
7. **Connect**: ThoughtSeeds discover cross-spec relationships

### Deduplication Flow
1. **Check Hash**: Before ingestion, query Neo4j for existing content_hash
2. **If Exists**: Return 409 Conflict, skip processing
3. **If New**: Proceed with normal ingestion

### Version Update Flow
1. **Detect Change**: content_hash differs from existing spec
2. **Increment Version**: 1.0 → 1.1
3. **Create New Node**: Don't overwrite old version
4. **Link Versions**: (v1.0)-[:NEXT_VERSION]->(v1.1)

## Search Integration

### Semantic Search Query
```python
# Query: "Find specs about authentication"
POST /api/query
{
  "query": "authentication patterns",
  "filters": {
    "source_type": "openspec"
  }
}

# Returns:
[
  {
    "document_id": "uuid-1",
    "title": "Document Processing",
    "capability": "document-processing",
    "relevance_score": 0.87,
    "matched_concepts": ["Authentication", "User credentials"]
  },
  {
    "document_id": "uuid-2",
    "title": "CLAUSE Multi-Agent System",
    "capability": "clause-multi-agent",
    "relevance_score": 0.72,
    "matched_concepts": ["Agent authentication", "Trust verification"]
  }
]
```

### Graph Traversal Query
```cypher
// Find all specs related to a capability
MATCH (s:Specification {capability: "document-processing"})
MATCH (s)-[:HAS_THOUGHTSEED]->(ts:ThoughtSeed)
MATCH (ts)<-[:HAS_THOUGHTSEED]-(related:Specification)
RETURN s, related, ts
```

## Performance Considerations

### Ingestion Performance
- **Target**: < 10 seconds for 3 capability specs (6 files total)
- **Bottleneck**: LangGraph consciousness processing (5-10s per doc)
- **Optimization**: Parallel ingestion (asyncio), batch processing

### Storage Overhead
- **Estimate**: ~100KB per spec.md in Neo4j (text + embeddings + concepts)
- **Total**: 3 capabilities × 2 files × 100KB = 600KB
- **Negligible** compared to document processing workload

### Search Performance
- **Target**: < 500ms for semantic search across specs
- **Optimization**: Neo4j vector index on embeddings

## Testing Strategy

### Unit Tests
- Test spec file scanner (finds all .md files)
- Test metadata extraction (capability, spec_type)
- Test content hash calculation (SHA-256)
- Test requirement parser (extract ### Requirement:)

### Integration Tests
1. Create test spec file
2. Run ingestion script
3. Verify Document node in Neo4j
4. Verify requirements/scenarios extracted
5. Verify semantic search returns spec

### Manual QA
- Ingest real OpenSpec specs (document-processing, clause-multi-agent, knowledge-graph)
- Search for "LangGraph" → expect document-processing spec
- Search for "multi-agent" → expect clause-multi-agent spec
- Verify ThoughtSeeds connect related concepts
