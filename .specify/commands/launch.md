---
name: launch
description: Launch Dionysus backend server
---

# Launch Dionysus

Starts the Dionysus 2.0 backend API server.

## What it does:
- Starts FastAPI backend on port 9127
- Provides API documentation at /docs

## Usage:
```
/launch
```

## Implementation:
```bash
cd "${SPECIFY_PROJECT_ROOT}/backend"
python -m src.main
```
