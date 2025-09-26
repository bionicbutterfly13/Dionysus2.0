"""
FastAPI application for Dionysus Migration System

This is a minimal app structure to make contract tests importable.
The actual endpoints will be implemented later.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Dionysus Migration API",
    description="Distributed background migration of legacy consciousness components",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {"message": "Dionysus Migration System API"}


# Endpoints will be implemented in Phase 3.3