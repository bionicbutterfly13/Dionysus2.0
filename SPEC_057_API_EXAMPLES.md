# Spec 057: Source Metadata & External Access - API Examples

Quick reference for API usage with source metadata.

---

## Upload Document (Traditional)

```bash
POST /api/documents/persist
Content-Type: application/json

{
  "document_id": "doc_upload_001",
  "filename": "my_research.pdf",
  "content_hash": "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a",
  "file_size": 2048000,
  "mime_type": "application/pdf",
  "tags": ["personal", "research"],
  "source_type": "uploaded_file",
  "daedalus_output": {...}
}
```

Response: `201 Created`

---

## Ingest from URL (Spec 056 + 057)

```bash
POST /api/documents/ingest-url
Content-Type: application/json

{
  "url": "https://arxiv.org/pdf/2024.12345.pdf",
  "tags": ["research", "ai"]
}
```

Response includes source metadata automatically:
```json
{
  "status": "success",
  "document_id": "doc_1728302400000",
  "original_url": "https://arxiv.org/pdf/2024.12345.pdf",
  "chunks_created": 15
}
```

---

## List Documents with Filter

```bash
# Get all URL-sourced documents
GET /api/documents?source_type=url

# Get all uploaded files
GET /api/documents?source_type=uploaded_file

# Get API-ingested documents
GET /api/documents?source_type=api
```

Response includes source metadata:
```json
{
  "documents": [
    {
      "document_id": "doc_001",
      "filename": "paper.pdf",
      "source_type": "url",
      "original_url": "https://arxiv.org/pdf/2024.12345.pdf",
      "connector_icon": "pdf"
    }
  ]
}
```

---

## Get Document Detail

```bash
GET /api/documents/{document_id}
```

Response includes full source metadata:
```json
{
  "document_id": "doc_001",
  "metadata": {
    "filename": "paper.pdf",
    "source_type": "url",
    "original_url": "https://arxiv.org/pdf/2024.12345.pdf",
    "connector_icon": "pdf",
    "download_metadata": {
      "status_code": 200,
      "redirected_url": "https://arxiv.org/pdf/2024.12345.pdf",
      "download_duration_ms": 567.89
    }
  }
}
```

---

## Check External Link ("Open Original" Button)

```bash
GET /api/documents/{document_id}/external-link
```

Response for URL-sourced document:
```json
{
  "available": true,
  "url": "https://arxiv.org/pdf/2024.12345.pdf",
  "source_type": "url",
  "message": "Original document available at URL"
}
```

Response for uploaded file:
```json
{
  "available": false,
  "url": null,
  "source_type": "uploaded_file",
  "message": "Document was uploaded directly (no external source)"
}
```

---

## Connector Icons

Icon hints for UI display:

| source_type | mime_type | connector_icon |
|-------------|-----------|----------------|
| uploaded_file | application/pdf | pdf |
| uploaded_file | text/html | html |
| uploaded_file | text/plain | text |
| uploaded_file | application/msword | doc |
| uploaded_file | text/markdown | markdown |
| uploaded_file | application/json | json |
| uploaded_file | (unknown) | upload |
| url | (any) | web or mime-specific |
| api | (any) | api or mime-specific |

---

## Source Type Values

Valid `source_type` values:

- `uploaded_file`: Traditional file upload
- `url`: Downloaded from web URL (Spec 056)
- `api`: Ingested via API

---

## Migration

Backfill existing documents:

```bash
cd backend
python scripts/migrate_source_metadata.py
```

All existing documents get:
- `source_type`: "uploaded_file"
- `original_url`: null
- `connector_icon`: inferred from mime_type
- `download_metadata`: null

---

**Implementation**: Spec 057
**Date**: 2025-10-07
**Status**: Production Ready
