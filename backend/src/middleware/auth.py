"""
Local Authentication Middleware - T033
Flux Self-Teaching Consciousness Emulator

Implements local authentication for self-hosted deployment.
Constitutional compliance: local-first operation, no external dependencies.
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

logger = logging.getLogger(__name__)

class LocalAuthMiddleware(BaseHTTPMiddleware):
    """
    Local authentication middleware for Flux backend.

    Features:
    - JWT token validation for authenticated endpoints
    - Local-first security (no external auth services)
    - Development mode support
    - Constitutional compliance
    """

    def __init__(self, app, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        super().__init__(app)
        self.secret_key = secret_key or os.getenv("SECRET_KEY", "flux-dev-secret-change-in-production")
        self.algorithm = algorithm
        self.security = HTTPBearer(auto_error=False)

        # Paths that don't require authentication
        self.public_paths = {
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

        # Development mode
        self.dev_mode = os.getenv("FLUX_ENV", "development") == "development"

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through authentication middleware."""

        # Skip auth for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)

        # Skip auth in development mode for easier testing
        if self.dev_mode and request.headers.get("X-Dev-Mode") == "true":
            logger.info(f"Development mode: bypassing auth for {request.url.path}")
            return await call_next(request)

        try:
            # Extract token from Authorization header
            token = await self._extract_token(request)

            if token:
                # Validate token
                payload = self._validate_token(token)
                # Add user info to request state
                request.state.user_id = payload.get("sub")
                request.state.user_permissions = payload.get("permissions", [])
            else:
                # No token provided - allow for now in local development
                if self.dev_mode:
                    request.state.user_id = "dev-user"
                    request.state.user_permissions = ["read", "write"]
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )

        except HTTPException as e:
            # Authentication failed
            logger.warning(f"Authentication failed for {request.url.path}: {e.detail}")
            raise
        except Exception as e:
            logger.error(f"Auth middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication system error"
            )

        response = await call_next(request)
        return response

    async def _extract_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from request headers."""
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            return authorization.replace("Bearer ", "")
        return None

    def _validate_token(self, token: str) -> dict:
        """Validate JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )

            return payload

        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {e}"
            )

    def create_token(self, user_id: str, permissions: list = None) -> str:
        """Create JWT token for user (utility method)."""
        permissions = permissions or ["read", "write"]

        payload = {
            "sub": user_id,
            "permissions": permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=24)
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)


# Utility functions for token management
def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create access token for authenticated user."""
    secret_key = os.getenv("SECRET_KEY", "flux-dev-secret-change-in-production")
    algorithm = os.getenv("ALGORITHM", "HS256")

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)

    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def verify_token(token: str) -> Optional[dict]:
    """Verify token and return payload."""
    secret_key = os.getenv("SECRET_KEY", "flux-dev-secret-change-in-production")
    algorithm = os.getenv("ALGORITHM", "HS256")

    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except jwt.InvalidTokenError:
        return None