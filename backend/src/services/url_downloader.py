#!/usr/bin/env python3
"""
URL Downloader Service - Spec 056

Robust URL download service with:
- HTTPS URL support only
- Retry logic with exponential backoff (3 retries, 1s/2s/4s delays)
- MIME type validation (PDF, HTML, plain text)
- Timeout handling (30s default)
- User-Agent headers
- Error handling for 404, 403, network issues
- Redirect tracking

Constitutional Compliance:
- Pure utility service, no direct Neo4j access
- Returns structured data for DocumentRepository to process

Author: Spec 056 Implementation
Created: 2025-10-07
"""

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class DownloadError(Exception):
    """Base exception for download errors."""
    pass


class NetworkError(DownloadError):
    """Network-related errors (timeout, connection failure)."""
    pass


class UnsupportedMimeTypeError(DownloadError):
    """MIME type not supported for ingestion."""
    pass


# ============================================================================
# Download Result Type
# ============================================================================

class DownloadResult:
    """
    Structured result from URL download.

    Attributes:
        content: Downloaded bytes
        mime_type: Content MIME type
        status_code: HTTP status code
        url: Original requested URL
        redirected_url: Final URL after redirects (if any)
        size_bytes: Content size in bytes
        download_duration_ms: Time taken to download
    """

    def __init__(
        self,
        content: bytes,
        mime_type: str,
        status_code: int,
        url: str,
        redirected_url: Optional[str],
        size_bytes: int,
        download_duration_ms: float
    ):
        self.content = content
        self.mime_type = mime_type
        self.status_code = status_code
        self.url = url
        self.redirected_url = redirected_url
        self.size_bytes = size_bytes
        self.download_duration_ms = download_duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "mime_type": self.mime_type,
            "status_code": self.status_code,
            "url": self.url,
            "redirected_url": self.redirected_url,
            "size_bytes": self.size_bytes,
            "download_duration_ms": self.download_duration_ms
        }


# ============================================================================
# URL Downloader Service
# ============================================================================

class URLDownloader:
    """
    Robust URL download service for document ingestion.

    Spec 056: Downloads content from HTTPS URLs with retry logic,
    MIME type validation, and comprehensive error handling.

    Example:
        >>> downloader = URLDownloader()
        >>> result = await downloader.download_url("https://example.com/doc.pdf")
        >>> print(result["mime_type"])
        'application/pdf'
    """

    DEFAULT_ALLOWED_MIME_TYPES = [
        "application/pdf",
        "text/html",
        "text/plain"
    ]

    DEFAULT_USER_AGENT = "Dionysus-DocumentIngestion/1.0 (https://github.com/dionysus)"

    def __init__(
        self,
        allowed_mime_types: Optional[List[str]] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        timeout: int = 30,
        user_agent: Optional[str] = None
    ):
        """
        Initialize URL downloader.

        Args:
            allowed_mime_types: List of allowed MIME types (default: PDF, HTML, plain text)
            max_retries: Maximum retry attempts (default: 3)
            initial_delay: Initial retry delay in seconds (default: 1.0)
            timeout: Request timeout in seconds (default: 30)
            user_agent: Custom User-Agent header (default: Dionysus bot)
        """
        self.allowed_mime_types = allowed_mime_types or self.DEFAULT_ALLOWED_MIME_TYPES
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.timeout = timeout
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT

        logger.info(
            f"URLDownloader initialized: "
            f"mime_types={len(self.allowed_mime_types)}, "
            f"max_retries={self.max_retries}, "
            f"timeout={self.timeout}s"
        )

    async def download_url(
        self,
        url: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Download content from HTTPS URL with retry logic.

        Spec 056: Core download method with exponential backoff retry.

        Args:
            url: HTTPS URL to download
            timeout: Optional custom timeout (overrides default)

        Returns:
            Dictionary with:
                - content: bytes
                - mime_type: str
                - status_code: int
                - url: str
                - redirected_url: str | None
                - size_bytes: int
                - download_duration_ms: float

        Raises:
            ValueError: Invalid URL or non-HTTPS scheme
            UnsupportedMimeTypeError: MIME type not in allowed list
            DownloadError: HTTP error (404, 403, etc.)
            NetworkError: Network/timeout errors after retries

        Example:
            >>> downloader = URLDownloader()
            >>> result = await downloader.download_url("https://arxiv.org/pdf/2301.12345.pdf")
            >>> print(f"Downloaded {result['size_bytes']} bytes")
        """
        # Validate URL
        self._validate_url(url)

        # Use custom timeout if provided
        request_timeout = timeout or self.timeout

        start_time = time.time()
        attempt = 0
        delay = self.initial_delay

        while attempt < self.max_retries:
            attempt += 1

            try:
                logger.info(f"Downloading URL (attempt {attempt}/{self.max_retries}): {url}")

                # Perform download
                result = await self._download_with_timeout(url, request_timeout)

                download_duration_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"✅ Download successful: {url} "
                    f"({result['size_bytes']} bytes, {download_duration_ms:.0f}ms)"
                )

                return {
                    **result,
                    "download_duration_ms": download_duration_ms
                }

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt >= self.max_retries:
                    # Exhausted retries
                    logger.error(
                        f"❌ Download failed after {self.max_retries} retries: {url} - {e}"
                    )
                    raise NetworkError(
                        f"Failed to download {url} after {self.max_retries} retries: {str(e)}"
                    )

                # Log retry attempt
                logger.warning(
                    f"Download attempt {attempt} failed, retrying in {delay}s: {url} - {e}"
                )

                # Exponential backoff
                await asyncio.sleep(delay)
                delay *= 2  # Double delay for next attempt (1s → 2s → 4s)

            except (UnsupportedMimeTypeError, DownloadError):
                # Don't retry for these errors
                raise

            except Exception as e:
                # Unexpected error, don't retry
                logger.error(f"❌ Unexpected download error: {url} - {e}", exc_info=True)
                raise DownloadError(f"Unexpected error downloading {url}: {str(e)}")

        # Should never reach here, but safety fallback
        raise NetworkError(f"Failed to download {url} after {self.max_retries} retries")

    def _validate_url(self, url: str) -> None:
        """
        Validate URL format and scheme.

        Args:
            url: URL to validate

        Raises:
            ValueError: If URL is invalid or not HTTPS
        """
        if not url:
            raise ValueError("URL cannot be empty")

        parsed = urlparse(url)

        if parsed.scheme != "https":
            raise ValueError(
                f"Only HTTPS URLs are supported. Got scheme: {parsed.scheme}"
            )

        if not parsed.netloc:
            raise ValueError(f"Invalid URL format: {url}")

    async def _download_with_timeout(
        self,
        url: str,
        timeout: int
    ) -> Dict[str, Any]:
        """
        Perform actual download with timeout.

        Args:
            url: URL to download
            timeout: Timeout in seconds

        Returns:
            Dictionary with download results (without duration)

        Raises:
            DownloadError: HTTP error
            UnsupportedMimeTypeError: Invalid MIME type
            aiohttp.ClientError: Network error
            asyncio.TimeoutError: Timeout
        """
        headers = {
            "User-Agent": self.user_agent
        }

        timeout_config = aiohttp.ClientTimeout(total=timeout)

        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.get(url, headers=headers, allow_redirects=True) as response:
                # Check HTTP status
                if response.status >= 400:
                    reason = response.reason or "Unknown"
                    if response.status == 404:
                        raise DownloadError(f"404 Not Found: {url}")
                    elif response.status == 403:
                        raise DownloadError(f"403 Forbidden: {url}")
                    else:
                        raise DownloadError(
                            f"HTTP {response.status} {reason}: {url}"
                        )

                # Extract MIME type
                content_type = response.headers.get("Content-Type", "")
                mime_type = self._extract_mime_type(content_type)

                # Validate MIME type
                if mime_type not in self.allowed_mime_types:
                    raise UnsupportedMimeTypeError(
                        f"MIME type '{mime_type}' is not supported. "
                        f"Allowed types: {', '.join(self.allowed_mime_types)}"
                    )

                # Read content
                content = await response.read()

                # Track redirects
                redirected_url = None
                if str(response.url) != url:
                    redirected_url = str(response.url)

                return {
                    "content": content,
                    "mime_type": mime_type,
                    "status_code": response.status,
                    "url": url,
                    "redirected_url": redirected_url,
                    "size_bytes": len(content)
                }

    def _extract_mime_type(self, content_type: str) -> str:
        """
        Extract base MIME type from Content-Type header.

        Args:
            content_type: Content-Type header value (e.g., "text/html; charset=utf-8")

        Returns:
            Base MIME type (e.g., "text/html")

        Example:
            >>> downloader = URLDownloader()
            >>> downloader._extract_mime_type("text/html; charset=utf-8")
            'text/html'
        """
        if not content_type:
            return "application/octet-stream"

        # Split on semicolon to remove parameters
        mime_type = content_type.split(";")[0].strip().lower()

        return mime_type
