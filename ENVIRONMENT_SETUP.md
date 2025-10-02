# Environment Setup Guide

**Single Source of Truth for All Credentials**

All database passwords, API keys, and configuration are stored in `.env` file.

## Quick Setup

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual credentials**:
   ```bash
   # Use your preferred editor
   nano .env
   # or
   vim .env
   # or
   code .env
   ```

3. **Update the passwords** (minimum required):
   - `NEO4J_PASSWORD` - Your Neo4j password
   - `REDIS_PASSWORD` - If your Redis requires authentication
   - `SECRET_KEY` - Generate a secure random key for production

## Required Services

### Neo4j (REQUIRED for document processing)

**Start with Docker**:
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password_here \
  neo4j:latest
```

**Update `.env`**:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

**Verify connection**:
```bash
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'your_password_here'))
driver.verify_connectivity()
print('✅ Neo4j connected!')
driver.close()
"
```

### Redis (REQUIRED for consciousness processing)

**Start with Docker**:
```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7-alpine
```

**Update `.env`**:
```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
```

**Verify connection**:
```bash
redis-cli ping
# Should return: PONG
```

## Optional Services

### MongoDB (for legacy migration only)
```bash
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  mongo:latest
```

### PostgreSQL (future use)
```bash
docker run -d \
  --name postgres \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=your_password_here \
  postgres:15-alpine
```

## Configuration Hierarchy

The system reads configuration in this order:

1. **`.env` file** (primary source, gitignored)
2. **Environment variables** (overrides `.env`)
3. **Default values** in `backend/src/config/settings.py`

## Security Notes

- ✅ `.env` is gitignored and will never be committed
- ✅ `.env.example` is committed with placeholder values
- ⚠️ Never commit actual passwords or API keys
- ⚠️ Change `SECRET_KEY` in production

## Agent Access

**All agents should read from `.env`**:

```python
from backend.src.config.settings import settings

# Database connections
neo4j_uri = settings.NEO4J_URI
neo4j_user = settings.NEO4J_USER
neo4j_password = settings.NEO4J_PASSWORD

redis_url = settings.REDIS_URL

# API keys
openai_key = settings.OPENAI_API_KEY
```

## Troubleshooting

### "Neo4j connection failed"
1. Check if Neo4j is running: `docker ps | grep neo4j`
2. Verify credentials in `.env` match your Neo4j setup
3. Test connection: `python -c "from neo4j import GraphDatabase; ..."`

### "Redis connection refused"
1. Check if Redis is running: `docker ps | grep redis`
2. Verify port 6379 is not in use: `lsof -i :6379`

### "No .env file"
1. Copy the example: `cp .env.example .env`
2. Update with your credentials

## Production Deployment

For production:

1. **Generate secure secrets**:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Use environment variables** instead of `.env` file
3. **Enable authentication** on all databases
4. **Use TLS/SSL** for database connections
5. **Rotate credentials** regularly

## Current Configuration

To see your current configuration (passwords masked):

```bash
python -c "
from backend.src.config.settings import settings
import json

config = {
    'NEO4J_URI': settings.NEO4J_URI,
    'NEO4J_USER': settings.NEO4J_USER,
    'NEO4J_PASSWORD': '***' if settings.NEO4J_PASSWORD else None,
    'REDIS_URL': settings.REDIS_URL,
    'DEBUG': settings.DEBUG,
}
print(json.dumps(config, indent=2))
"
```
