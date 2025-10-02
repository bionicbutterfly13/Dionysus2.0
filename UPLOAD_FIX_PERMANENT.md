# PERMANENT UPLOAD FIX - Stop Rebuilding This

**Problem**: File upload has been rebuilt 5+ times and keeps breaking
**Root Cause**: No integration tests, no API contract, endpoint changes
**Solution**: Lock it down with tests, make changes IMPOSSIBLE without breaking tests

---

## The Problem

1. Frontend calls `/api/demo/process-document` ❌
2. Backend has `/api/documents` ❌
3. **THEY DON'T MATCH**
4. No tests catch this
5. Code changes break it silently
6. We rebuild it every week

---

## The PERMANENT Solution

### Step 1: Define API Contract (LOCKED)
```yaml
# backend/contracts/upload.yaml
POST /api/documents/upload:
  request:
    Content-Type: multipart/form-data
    Body:
      - file: File (required)
      - tags: string (optional, comma-separated)
  response:
    200:
      success: true
      document_id: string
      filename: string
      size: number
      status: "completed"
```

### Step 2: Backend Endpoint (DO NOT CHANGE)
```python
# backend/src/api/routes/documents.py
@router.post("/upload")  # LOCKED - tests depend on this
async def upload_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None)
):
    """PERMANENT upload endpoint - DO NOT MODIFY without updating tests"""
    # Process through Daedalus
    # Store in Neo4j
    # Return success
```

### Step 3: Frontend (DO NOT CHANGE)
```typescript
// frontend/src/services/upload.ts
const UPLOAD_ENDPOINT = '/api/documents/upload' // LOCKED

export async function uploadFile(file: File, tags?: string[]) {
  const formData = new FormData()
  formData.append('file', file)
  if (tags) formData.append('tags', tags.join(','))

  const response = await fetch(UPLOAD_ENDPOINT, {
    method: 'POST',
    body: formData
  })

  if (!response.ok) throw new Error('Upload failed')
  return response.json()
}
```

### Step 4: Integration Tests (PREVENT REGRESSION)
```typescript
// frontend/tests/upload.spec.ts
test('upload endpoint matches backend', async () => {
  // This test MUST pass - if it fails, upload is broken
  const response = await fetch('/api/documents/upload', {
    method: 'POST',
    body: createMockFile()
  })
  expect(response.status).not.toBe(404) // Endpoint exists
})

test('file uploads and appears in database', async () => {
  const file = createTestFile('test.pdf')
  const result = await uploadFile(file)

  expect(result.success).toBe(true)
  expect(result.document_id).toBeDefined()

  // Verify in database
  const doc = await fetchDocument(result.document_id)
  expect(doc).toBeDefined()
  expect(doc.filename).toBe('test.pdf')
})
```

```python
# backend/tests/test_upload_permanent.py
def test_upload_endpoint_exists():
    """CRITICAL: Upload endpoint must exist at /upload"""
    response = client.post("/api/documents/upload", files={"file": ("test.pdf", b"test")})
    assert response.status_code != 404, "Upload endpoint missing!"

def test_file_stored_in_neo4j():
    """CRITICAL: Uploaded files must reach Neo4j"""
    response = client.post("/api/documents/upload", files={"file": ("test.pdf", b"content")})
    assert response.status_code == 200

    file_id = response.json()["document_id"]

    # Query Neo4j
    with driver.session() as session:
        result = session.run("MATCH (d:Document {id: $id}) RETURN d", id=file_id)
        doc = result.single()
        assert doc is not None, "File not in Neo4j!"
```

### Step 5: CI/CD Gate (ENFORCE)
```yaml
# .github/workflows/test.yml
- name: Upload Tests (MUST PASS)
  run: |
    pytest backend/tests/test_upload_permanent.py -v
    npm run test:upload
  # If these fail, PR CANNOT merge
```

---

## Implementation Plan (DO ONCE, NEVER AGAIN)

### Task 1: Fix Endpoint Mismatch
- [ ] Change frontend from `/api/demo/process-document` → `/api/documents/upload`
- [ ] Update backend route from `/documents` → `/upload`
- [ ] Test: File uploads successfully

### Task 2: Add Integration Tests
- [ ] Backend test: Upload stores in Neo4j
- [ ] Frontend test: Upload UI → backend → database
- [ ] Test: Uploaded files appear in list

### Task 3: Lock It Down
- [ ] Add contract tests (endpoint exists)
- [ ] Add E2E test (full flow works)
- [ ] Document: "DO NOT CHANGE without updating tests"

### Task 4: Remove Confusing UI
- [ ] Remove red X icon (line 191, 414)
- [ ] Show ONLY status: uploading → processing → completed
- [ ] Green check ONLY when file is in database

---

## Why This Will Work This Time

1. **Contract tests**: If endpoint changes, tests fail immediately
2. **Integration tests**: If database storage breaks, tests fail
3. **E2E tests**: If UI → backend connection breaks, tests fail
4. **CI/CD**: Cannot merge broken upload
5. **Documentation**: Clear "DO NOT TOUCH" warnings

---

## Execute Now

```bash
# 1. Fix the endpoint
cd frontend/src/pages
# Edit DocumentUpload.tsx line 84: '/api/documents/upload'

# 2. Add backend route
cd backend/src/api/routes
# Edit documents.py: @router.post("/upload")

# 3. Test it works
curl -X POST http://localhost:9127/api/documents/upload \
  -F "file=@test.pdf"

# 4. Add tests (prevent regression)
cd backend/tests
# Create test_upload_permanent.py

cd frontend/tests
# Create upload.spec.ts

# 5. Run tests
pytest backend/tests/test_upload_permanent.py
npm run test:upload

# If all pass → DONE FOREVER
```

---

*This is the LAST time we build upload. Tests prevent regression.*
