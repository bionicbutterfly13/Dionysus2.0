# Spec 041: Interface Health Checks & Server Validation

## Overview
Before loading the interface, validate ALL required services are online. Display helpful error messages when services are missing. Never show a broken interface to the user.

## Problem Statement
Current system loads the frontend even when backend/Neo4j/Redis are offline, resulting in:
- Broken upload functionality
- Failed API calls with no user feedback
- Confusion about what's wrong
- Poor user experience

## Requirements

### NFR-001: Pre-flight Health Check
- Frontend MUST check backend health before rendering main UI
- Backend health endpoint MUST validate Neo4j, Redis, Daedalus
- If critical services down, show clear error page with instructions

### NFR-002: Service Status Page
- Display status of each service (Neo4j, Redis, Backend, Daedalus)
- Show which features are affected by missing services
- Provide Docker commands to start missing services
- Auto-retry connection every 5 seconds

### NFR-003: Graceful Degradation
- Read-only mode when Neo4j down (no uploads/queries)
- Show warning banner when Redis down (reduced performance)
- All API endpoints return proper error codes and messages

### NFR-004: Developer Experience
- START_FLUX.sh script checks all dependencies before starting
- Clear error messages with actionable next steps
- Playwright tests validate service availability before running UI tests

## User Stories

### US-001: Starting System
**As a** developer
**I want** to run a single command that validates all services
**So that** I don't waste time debugging missing dependencies

**Acceptance Criteria:**
- `./START_FLUX.sh` checks Neo4j, Redis before starting backend
- Script prints clear status messages
- Script exits with error code if critical services missing
- Script provides commands to start missing services

### US-002: Loading Interface
**As a** user
**I want** to see a helpful error page when services are down
**So that** I know exactly what to do to fix it

**Acceptance Criteria:**
- Frontend calls `/api/health` before showing main UI
- If health check fails, show ServiceStatus page
- ServiceStatus page shows:
  - ✅ Backend: Running
  - ❌ Neo4j: Not connected (docker start neo4j-memory)
  - ⚠️ Redis: Optional (docker run -d -p 6379:6379 redis:7-alpine)
- Auto-retry button to check again
- Auto-refresh every 5 seconds

### US-003: Running Tests
**As a** developer running Playwright tests
**I want** tests to fail fast if services are down
**So that** I don't waste time debugging test failures

**Acceptance Criteria:**
- Test setup validates backend health
- Tests skip gracefully if services unavailable
- Clear error message: "Backend not running. Start with: ./START_FLUX.sh"

## Technical Design

### Frontend: Health Check Component
```typescript
// src/components/HealthCheck.tsx
interface HealthStatus {
  overall_status: 'healthy' | 'degraded' | 'down';
  services: {
    neo4j: ServiceStatus;
    redis: ServiceStatus;
    daedalus: ServiceStatus;
  };
  can_upload: boolean;
  can_crawl: boolean;
  can_query: boolean;
  errors: string[];
}

// Check health before rendering main app
const HealthCheck: React.FC = ({ children }) => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading || health?.overall_status === 'down') {
    return <ServiceStatusPage health={health} />;
  }

  if (health?.overall_status === 'degraded') {
    return (
      <>
        <WarningBanner services={health.services} />
        {children}
      </>
    );
  }

  return children;
};
```

### Backend: Enhanced Health Endpoint
```python
# Already exists at /api/health
# Returns:
# - overall_status: "healthy" | "degraded" | "down"
# - services: dict of ServiceStatus
# - can_upload, can_crawl, can_query: bool
# - errors: list of actionable error messages
```

### START_FLUX.sh: Service Validation
```bash
#!/bin/bash

echo "🔍 Checking required services..."

# Check Neo4j
if ! curl -s http://localhost:7474 > /dev/null 2>&1; then
  echo "❌ Neo4j not running"
  echo "   Start with: docker start neo4j-memory"
  echo "   Or create: docker run -d --name neo4j-memory -p 7474:7474 -p 7687:7687 neo4j:5"
  exit 1
fi

# Check Redis (optional warning)
if ! curl -s http://localhost:6379 > /dev/null 2>&1; then
  echo "⚠️  Redis not running (optional for caching)"
  echo "   Start with: docker run -d --name redis-dionysus -p 6379:6379 redis:7-alpine"
fi

echo "✅ All critical services available"
echo "🚀 Starting backend..."

export PYTHONPATH=/Volumes/Asylum/dev/Dionysus-2.0/backend/src:$PYTHONPATH
cd /Volumes/Asylum/dev/Dionysus-2.0/backend
/Volumes/Asylum/dev/Dionysus-2.0/backend/flux-backend-env/bin/uvicorn src.app_factory:app --host 127.0.0.1 --port 9127 --reload
```

### Playwright: Test Setup
```typescript
// tests/setup/validate-services.ts
test.beforeAll(async () => {
  const health = await fetch('http://127.0.0.1:9127/api/health');
  if (!health.ok) {
    throw new Error('Backend not running. Start with: ./START_FLUX.sh');
  }

  const data = await health.json();
  if (data.overall_status === 'down') {
    throw new Error(`Services down: ${data.errors.join(', ')}`);
  }
});
```

## Test Plan

### T001: START_FLUX.sh validates Neo4j
- Stop Neo4j: `docker stop neo4j-memory`
- Run: `./START_FLUX.sh`
- Expected: Script exits with error, shows docker start command
- Verify: Backend does NOT start

### T002: Frontend shows ServiceStatus when backend down
- Stop backend
- Load http://localhost:9243
- Expected: ServiceStatus page shows "Backend: Not Connected"
- Verify: Main UI does NOT render

### T003: Frontend shows ServiceStatus when Neo4j down
- Start backend (Neo4j still down)
- Load http://localhost:9243
- Expected: ServiceStatus page shows "Neo4j: Not Connected"
- Shows docker command to start Neo4j
- Verify: Main UI does NOT render

### T004: Frontend shows degraded banner when Redis down
- Start Neo4j: `docker start neo4j-memory`
- Load http://localhost:9243
- Expected: Main UI renders with yellow banner "Redis offline - caching disabled"
- Verify: Upload functionality works

### T005: Playwright tests fail fast when services down
- Stop backend
- Run: `npx playwright test`
- Expected: Tests fail immediately with clear message
- Verify: No browser launches, no screenshots

### T006: Auto-retry connects when services come online
- Start with Neo4j down
- ServiceStatus page shows
- Start Neo4j in background
- Expected: Page auto-detects and loads main UI within 5 seconds
- Verify: No manual refresh needed

## Dependencies
- Backend `/api/health` endpoint (✅ exists)
- React frontend health check component (❌ needs implementation)
- START_FLUX.sh service validation (❌ needs enhancement)
- Playwright test setup (❌ needs implementation)

## Out of Scope
- Monitoring/alerting for production deployments
- Health check for external APIs (OpenAI, etc.)
- Historical uptime tracking

## Success Criteria
1. ✅ START_FLUX.sh refuses to start if Neo4j down
2. ✅ Frontend shows helpful ServiceStatus page when services missing
3. ✅ Playwright tests skip gracefully when services unavailable
4. ✅ Zero "connection refused" errors shown to user
5. ✅ All error messages include actionable next steps
