# SPEC: Fix Flux Upload - Make It Actually Work

**Priority**: CRITICAL
**Status**: 🔴 BROKEN - Must fix now
**Owner**: Implementation team

---

## Problem Statement

**File upload in Flux doesn't work. Period.**

User uploads a file → nothing happens, or it silently fails, or it errors out. The basic functionality that worked in Dionysus 1.0 is broken. We need to:

1. ✅ Make upload work reliably
2. ✅ Show clear progress to the user
3. ✅ Store files in the database
4. ✅ Display uploaded files in the UI
5. ✅ NO SILENT FAILURES

---

## SPEC 1: Startup & Connection Manager

### Requirements

**1.1 Beautiful Loading Screen**
- Black background with subtle shine (gradient from #000000 to #1a1a1a)
- Thin luminescent threads (colored, with depth) as background animation
- Small overlay square showing loading progress

**1.2 Service Status Display**
```
┌─ Loading Dionysus ─────────────────┐
│                                    │
│ ▸ Backend (port 9127)    [████░░] │
│ ▸ Frontend (port 5173)   [██████] │
│ ▸ Neo4j (port 7687)      [████░░] │
│ ▸ Redis (port 6379)      [██████] │
│ ▸ PostgreSQL (port 5432) [░░░░░░] │
│                                    │
│ Status: Connecting...              │
└────────────────────────────────────┘
```

**1.3 Port Conflict Detection**
- If port is busy → Show dialog: "Backend is busy. Kill it?"
- User clicks "Kill" → App kills process and retries
- Same for frontend port conflicts
- NO blind failures - user always knows what's happening

**1.4 Connection Requirements**
- App DOES NOT open main interface until all services connect
- If service fails → Show diagnostic message user can copy/paste
- Diagnostic includes: port, error, suggested fix

### Tasks

- [ ] **S1-T1**: Create LoadingScreen.tsx with gradient black background
- [ ] **S1-T2**: Add luminescent thread animation (Canvas/Three.js)
- [ ] **S1-T3**: Create service status component with progress bars
- [ ] **S1-T4**: Implement port conflict detection
- [ ] **S1-T5**: Create "Kill process" dialog
- [ ] **S1-T6**: Add diagnostic message generator
- [ ] **S1-T7**: Block main UI until all services ready

### Tests

```typescript
// S1-TEST-1: Loading screen displays
test('shows loading screen on app start', async () => {
  render(<App />);
  expect(screen.getByText(/Loading Dionysus/i)).toBeInTheDocument();
});

// S1-TEST-2: Port conflict detected
test('detects backend port conflict', async () => {
  // Mock port 9127 busy
  mockPortBusy(9127);
  render(<App />);
  await waitFor(() => {
    expect(screen.getByText(/Backend is busy/i)).toBeInTheDocument();
    expect(screen.getByText(/Kill it?/i)).toBeInTheDocument();
  });
});

// S1-TEST-3: Service connection required
test('blocks main UI until services connected', async () => {
  mockServicesDisconnected();
  render(<App />);
  expect(screen.queryByText(/Document Upload/i)).not.toBeInTheDocument();
});
```

---

## SPEC 2: File Upload - MAKE IT WORK

### Requirements

**2.1 Upload Flow**
```
User selects file
  → Frontend validates (type, size)
  → Shows progress bar
  → Sends to backend
  → Backend processes
  → Stores in Neo4j/database
  → Returns confirmation
  → Frontend shows success
  → File appears in list
```

**2.2 UI Requirements**
- Drag & drop OR file picker
- Progress bar during upload (with percentage)
- Clear success message: "File uploaded successfully!"
- Error messages (not silent failures): "Upload failed: [reason]"
- Uploaded files list on left panel (most recent first)

**2.3 Backend Requirements**
- Endpoint: `POST /api/documents/upload`
- Accept: multipart/form-data
- Process through Daedalus gateway
- Store in Neo4j with metadata
- Return: `{ success: true, file_id: "...", filename: "..." }`

**2.4 NO SILENT FAILURES**
- Every error must show to user
- Console logs for debugging
- Toast notifications for status

### Tasks

- [ ] **S2-T1**: Fix backend upload endpoint (POST /api/documents/upload)
- [ ] **S2-T2**: Implement file validation (frontend + backend)
- [ ] **S2-T3**: Add progress bar to upload UI
- [ ] **S2-T4**: Connect upload to Daedalus gateway
- [ ] **S2-T5**: Store file metadata in Neo4j
- [ ] **S2-T6**: Return proper success/error responses
- [ ] **S2-T7**: Display uploaded files in left panel
- [ ] **S2-T8**: Add error toast notifications
- [ ] **S2-T9**: Add console logging for debugging

### Tests

```python
# S2-TEST-1: Upload endpoint works
def test_upload_file_success():
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")}
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "file_id" in response.json()

# S2-TEST-2: File stored in Neo4j
def test_file_in_database():
    # Upload file
    response = upload_test_file()
    file_id = response.json()["file_id"]

    # Query Neo4j
    with driver.session() as session:
        result = session.run("MATCH (d:Document {id: $id}) RETURN d", id=file_id)
        assert result.single() is not None

# S2-TEST-3: Error handling
def test_upload_invalid_file():
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.exe", b"exe content", "application/exe")}
    )
    assert response.status_code == 400
    assert "error" in response.json()
```

```typescript
// S2-TEST-4: Frontend upload flow
test('uploads file and shows in list', async () => {
  render(<DocumentUpload />);

  const file = new File(['test'], 'test.pdf', { type: 'application/pdf' });
  const input = screen.getByLabelText(/upload/i);

  await userEvent.upload(input, file);

  // Progress bar appears
  expect(screen.getByRole('progressbar')).toBeInTheDocument();

  // Success message
  await waitFor(() => {
    expect(screen.getByText(/uploaded successfully/i)).toBeInTheDocument();
  });

  // File in list
  expect(screen.getByText('test.pdf')).toBeInTheDocument();
});
```

---

## SPEC 3: Bulk Upload

### Requirements

**3.1 Bulk Upload UI**
- Button: "Upload Bulk"
- Modal/overlay showing bulk progress
- Progress: "Uploading 3 of 10 files..."
- Individual file status (✓ success, ✗ failed, ⏳ pending)
- Close button to dismiss when done

**3.2 Backend**
- Endpoint: `POST /api/documents/bulk`
- Process files sequentially or in parallel (configurable)
- Return array of results: `[{ filename, success, file_id, error }]`

**3.3 UX**
- User sees each file upload status
- Failed uploads clearly marked
- Notification when all done: "8 of 10 files uploaded successfully"
- Failed files can be retried

### Tasks

- [ ] **S3-T1**: Create bulk upload modal UI
- [ ] **S3-T2**: Implement bulk upload endpoint
- [ ] **S3-T3**: Add individual file progress tracking
- [ ] **S3-T4**: Display success/failure for each file
- [ ] **S3-T5**: Add retry for failed files
- [ ] **S3-T6**: Close modal button
- [ ] **S3-T7**: Show completion notification

### Tests

```python
# S3-TEST-1: Bulk upload works
def test_bulk_upload():
    files = [
        ("file1.pdf", b"content1", "application/pdf"),
        ("file2.pdf", b"content2", "application/pdf"),
    ]
    response = client.post("/api/documents/bulk", files=files)
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert all(r["success"] for r in results)
```

---

## SPEC 4: File List & Management

### Requirements

**4.1 Left Panel File List**
- Show most recent uploads (configurable limit, e.g., 10)
- Display: filename, upload date, size
- Click file → view details or open
- Older files: "View all" button

**4.2 File Actions**
- View file metadata
- Delete file
- Re-process through Daedalus

**4.3 Daedalus Integration**
- On upload → pass through Daedalus gateway
- Console log: Data structures passed to Daedalus
- Debug view: Show processing pipeline steps

### Tasks

- [ ] **S4-T1**: Create left panel file list component
- [ ] **S4-T2**: Fetch recent uploads from backend
- [ ] **S4-T3**: Display file metadata
- [ ] **S4-T4**: Add file actions (view, delete)
- [ ] **S4-T5**: Integrate with Daedalus gateway
- [ ] **S4-T6**: Add console logging for data structures
- [ ] **S4-T7**: Create debug view for pipeline

### Tests

```typescript
// S4-TEST-1: File list displays
test('shows recent uploads in left panel', async () => {
  mockRecentFiles([
    { id: '1', filename: 'test1.pdf', uploaded_at: '2025-10-02' },
    { id: '2', filename: 'test2.pdf', uploaded_at: '2025-10-01' },
  ]);

  render(<DocumentUpload />);

  expect(screen.getByText('test1.pdf')).toBeInTheDocument();
  expect(screen.getByText('test2.pdf')).toBeInTheDocument();
});
```

---

## SPEC 5: Error Handling & Debugging

### Requirements

**5.1 Error Display**
- NO silent failures
- All errors shown to user via toast notifications
- Error messages are actionable: "Upload failed: File too large (max 10MB)"

**5.2 Console Logging**
```javascript
console.log('[UPLOAD] Starting upload:', filename);
console.log('[UPLOAD] File size:', fileSize);
console.log('[UPLOAD] Backend response:', response);
console.log('[DAEDALUS] Processing:', data_structure);
console.log('[DAEDALUS] Pipeline step:', step_name, step_data);
```

**5.3 Debug View**
- Toggle debug panel (keyboard shortcut: Cmd+D)
- Shows: Recent API calls, responses, errors
- Exportable diagnostic report

### Tasks

- [ ] **S5-T1**: Add toast notification system
- [ ] **S5-T2**: Implement comprehensive console logging
- [ ] **S5-T3**: Create debug panel UI
- [ ] **S5-T4**: Add diagnostic report export
- [ ] **S5-T5**: Keyboard shortcut for debug view

### Tests

```typescript
// S5-TEST-1: Errors shown to user
test('displays upload error to user', async () => {
  mockUploadError('File too large');

  render(<DocumentUpload />);
  await uploadFile('large.pdf');

  expect(screen.getByText(/File too large/i)).toBeInTheDocument();
});
```

---

## Implementation Order

**DO NOT MOVE TO NEXT SPEC UNTIL CURRENT ONE WORKS**

1. ✅ **SPEC 1**: Startup & Connection (prevents blind failures)
2. ✅ **SPEC 2**: File Upload (core functionality)
3. ✅ **SPEC 3**: Bulk Upload (after single upload works)
4. ✅ **SPEC 4**: File List & Management
5. ✅ **SPEC 5**: Error Handling & Debugging

---

## Success Criteria

### SPEC 1 Success
- [ ] Loading screen displays with gradient background
- [ ] Service status shows with progress bars
- [ ] Port conflicts detected and shown to user
- [ ] App blocks until all services connected

### SPEC 2 Success
- [ ] User uploads file → sees progress bar
- [ ] File reaches backend
- [ ] File stored in Neo4j
- [ ] Success message shown
- [ ] File appears in list
- [ ] **NO SILENT FAILURES**

### SPEC 3 Success
- [ ] User uploads 10 files
- [ ] Sees progress for each file
- [ ] All files in database
- [ ] Completion notification shown

### SPEC 4 Success
- [ ] Recent files shown in left panel
- [ ] Click file → view details
- [ ] Files processed through Daedalus
- [ ] Console shows data structures

### SPEC 5 Success
- [ ] Every error has a toast notification
- [ ] Console has detailed logs
- [ ] Debug panel accessible
- [ ] Diagnostic report exportable

---

## Current Issues to Fix

1. ❌ Upload endpoint broken
2. ❌ No progress indication
3. ❌ Silent failures everywhere
4. ❌ Files not in database
5. ❌ No user feedback
6. ❌ No error messages
7. ❌ App starts before services ready

**STOP TALKING ABOUT FANCY FEATURES. FIX THE UPLOAD.**

---

*Created: 2025-10-02*
*Priority: CRITICAL*
*Next: Implement SPEC 1, test, then move to SPEC 2*
