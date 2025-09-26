"""
Request Validation Middleware - T032
Flux Self-Teaching Consciousness Emulator

Handles request validation, error formatting, and constitutional compliance checks.
"""

import json
import logging
from typing import Any, Dict, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from pydantic import ValidationError
import time

logger = logging.getLogger(__name__)

class ValidationMiddleware(BaseHTTPMiddleware):
    """
    Request validation and error handling middleware.

    Features:
    - Request/response validation
    - Error standardization
    - Constitutional compliance checks
    - Request logging and metrics
    - Mock data transparency enforcement
    """

    def __init__(self, app):
        super().__init__(app)
        self.max_request_size = 50 * 1024 * 1024  # 50MB for document uploads
        self.require_mock_data_disclosure = True

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through validation middleware."""
        start_time = time.time()

        try:
            # Pre-request validation
            await self._validate_request(request)

            # Process request
            response = await call_next(request)

            # Post-response validation
            response = await self._validate_response(request, response)

            # Log request completion
            process_time = time.time() - start_time
            await self._log_request(request, response, process_time)

            return response

        except HTTPException as e:
            # Handle known HTTP exceptions
            return await self._create_error_response(e.status_code, e.detail, request)

        except ValidationError as e:
            # Handle Pydantic validation errors
            return await self._create_validation_error_response(e, request)

        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in validation middleware: {e}")
            return await self._create_error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Internal server error",
                request
            )

    async def _validate_request(self, request: Request) -> None:
        """Validate incoming request."""

        # Check request size
        if hasattr(request, 'content_length') and request.content_length:
            if request.content_length > self.max_request_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request too large. Maximum size: {self.max_request_size} bytes"
                )

        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")

            # For document upload endpoints, allow multipart/form-data
            if "/documents" in str(request.url):
                if not (content_type.startswith("multipart/form-data") or
                       content_type.startswith("application/json")):
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail="Unsupported media type. Use multipart/form-data or application/json"
                    )

            # For other endpoints, require JSON
            elif not content_type.startswith("application/json"):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Content-Type must be application/json"
                )

        # Constitutional compliance: Check for mock data transparency
        if self.require_mock_data_disclosure:
            await self._check_mock_data_compliance(request)

    async def _validate_response(self, request: Request, response: Response) -> Response:
        """Validate outgoing response."""

        # Add constitutional compliance headers
        response.headers["X-Flux-Constitutional-Compliance"] = "enabled"
        response.headers["X-Flux-Local-First"] = "true"
        response.headers["X-Flux-Evaluation-Framework"] = "active"

        # Add mock data transparency headers if applicable
        if hasattr(request.state, 'mock_data_used') and request.state.mock_data_used:
            response.headers["X-Flux-Mock-Data"] = "true"
            response.headers["X-Flux-Mock-Data-Disclosure"] = "This response contains mock data for development purposes"

        return response

    async def _check_mock_data_compliance(self, request: Request) -> None:
        """Check mock data transparency compliance."""

        # Skip for health checks and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return

        # Check if request contains mock data indicators
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Try to read body for mock data flags
                body = await request.body()
                if body:
                    # Reset stream for downstream consumption
                    request._body = body

                    try:
                        if request.headers.get("content-type", "").startswith("application/json"):
                            data = json.loads(body.decode())

                            # Check for mock_data flags in request
                            if self._contains_mock_data_flag(data):
                                request.state.mock_data_used = True
                                logger.info(f"Mock data detected in request to {request.url.path}")

                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass  # Not JSON or not decodable - skip check

            except Exception as e:
                logger.warning(f"Could not check mock data compliance: {e}")

    def _contains_mock_data_flag(self, data: Any) -> bool:
        """Recursively check for mock_data flags in request data."""
        if isinstance(data, dict):
            if data.get("mock_data") is True:
                return True
            return any(self._contains_mock_data_flag(v) for v in data.values())
        elif isinstance(data, list):
            return any(self._contains_mock_data_flag(item) for item in data)
        return False

    async def _log_request(self, request: Request, response: Response, process_time: float) -> None:
        """Log request details for monitoring."""
        log_data = {
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "process_time": round(process_time, 3),
            "user_id": getattr(request.state, 'user_id', 'anonymous'),
            "mock_data_used": getattr(request.state, 'mock_data_used', False)
        }

        # Log different levels based on status code
        if response.status_code >= 500:
            logger.error(f"Request failed: {log_data}")
        elif response.status_code >= 400:
            logger.warning(f"Client error: {log_data}")
        else:
            logger.info(f"Request processed: {log_data}")

    async def _create_error_response(self, status_code: int, detail: str, request: Request) -> JSONResponse:
        """Create standardized error response."""
        error_response = {
            "error": {
                "code": status_code,
                "message": detail,
                "timestamp": time.time(),
                "path": str(request.url.path),
                "constitutional_compliance": {
                    "evaluation_required": True,
                    "mock_data_transparent": getattr(request.state, 'mock_data_used', False),
                    "local_first_operation": True
                }
            }
        }

        return JSONResponse(
            status_code=status_code,
            content=error_response
        )

    async def _create_validation_error_response(self, error: ValidationError, request: Request) -> JSONResponse:
        """Create validation error response."""
        error_details = []
        for err in error.errors():
            error_details.append({
                "field": " -> ".join(str(x) for x in err["loc"]),
                "message": err["msg"],
                "type": err["type"]
            })

        error_response = {
            "error": {
                "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "message": "Validation failed",
                "details": error_details,
                "timestamp": time.time(),
                "path": str(request.url.path),
                "constitutional_compliance": {
                    "evaluation_required": True,
                    "validation_enforced": True
                }
            }
        }

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response
        )