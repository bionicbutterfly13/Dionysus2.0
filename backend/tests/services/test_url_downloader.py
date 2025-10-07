#!/usr/bin/env python3
"""
Test URL Downloader Service - Spec 056

Comprehensive tests for URL download functionality with:
- HTTPS URL support
- Retry logic with exponential backoff
- MIME type validation
- Timeout handling
- Error handling for 404, 403, network issues

Author: Spec 056 Implementation
Created: 2025-10-07
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Imports will work due to conftest.py path setup
from src.services.url_downloader import (
    URLDownloader,
    DownloadResult,
    DownloadError,
    UnsupportedMimeTypeError,
    NetworkError
)


class TestURLDownloaderBasics:
    """Test basic URL download functionality."""

    @pytest.mark.asyncio
    async def test_download_pdf_success(self):
        """Test successful PDF download from HTTPS URL."""
        downloader = URLDownloader()

        # Mock response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {
                'Content-Type': 'application/pdf',
                'Content-Length': '1024'
            }
            mock_response.read = AsyncMock(return_value=b'PDF content here')
            mock_response.url = "https://example.com/test.pdf"  # No redirect
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await downloader.download_url("https://example.com/test.pdf")

            assert result["status_code"] == 200
            assert result["mime_type"] == "application/pdf"
            assert result["content"] == b'PDF content here'
            assert result["size_bytes"] == 16
            assert result["url"] == "https://example.com/test.pdf"
            assert "download_duration_ms" in result
            assert result["redirected_url"] is None

    @pytest.mark.asyncio
    async def test_download_html_success(self):
        """Test successful HTML download from HTTPS URL."""
        downloader = URLDownloader()

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
            mock_response.read = AsyncMock(return_value=b'<html>Test</html>')
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await downloader.download_url("https://example.com/page.html")

            assert result["status_code"] == 200
            assert result["mime_type"] == "text/html"
            assert result["content"] == b'<html>Test</html>'

    @pytest.mark.asyncio
    async def test_download_plain_text_success(self):
        """Test successful plain text download."""
        downloader = URLDownloader()

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'text/plain'}
            mock_response.read = AsyncMock(return_value=b'Plain text content')
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await downloader.download_url("https://example.com/notes.txt")

            assert result["status_code"] == 200
            assert result["mime_type"] == "text/plain"
            assert result["content"] == b'Plain text content'


class TestURLDownloaderRetryLogic:
    """Test retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self):
        """Test that network errors trigger retry with exponential backoff."""
        downloader = URLDownloader(max_retries=3, initial_delay=0.01)

        attempt_count = [0]

        async def mock_get_with_retry(*args, **kwargs):
            attempt_count[0] += 1

            # Create async context manager
            class MockContextManager:
                async def __aenter__(self):
                    if attempt_count[0] <= 2:
                        # First 2 attempts fail
                        raise Exception(f"Network error {attempt_count[0]}")
                    else:
                        # Third attempt succeeds
                        mock_response = MagicMock()
                        mock_response.status = 200
                        mock_response.headers = {'Content-Type': 'text/plain'}
                        mock_response.read = AsyncMock(return_value=b'Success after retry')
                        mock_response.url = "https://example.com/retry.txt"
                        return mock_response

                async def __aexit__(self, *args):
                    pass

            return MockContextManager()

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get = mock_get_with_retry
            mock_session.return_value.__aexit__.return_value = AsyncMock()

            with patch('asyncio.sleep'):  # Speed up test
                result = await downloader.download_url("https://example.com/retry.txt")

            assert result["status_code"] == 200
            assert result["content"] == b'Success after retry'
            assert attempt_count[0] == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test that download fails after max retries exhausted."""
        downloader = URLDownloader(max_retries=3, initial_delay=0.01)

        async def mock_get_always_fails(*args, **kwargs):
            class MockContextManager:
                async def __aenter__(self):
                    raise Exception("Persistent network error")

                async def __aexit__(self, *args):
                    pass

            return MockContextManager()

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get = mock_get_always_fails
            mock_session.return_value.__aexit__.return_value = AsyncMock()

            with patch('asyncio.sleep'):  # Speed up test
                with pytest.raises(NetworkError) as exc_info:
                    await downloader.download_url("https://example.com/fail.txt")

                assert "after 3 retries" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test that retry delays follow exponential backoff pattern."""
        downloader = URLDownloader(max_retries=3, initial_delay=1.0)

        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        async def mock_get_always_fails(*args, **kwargs):
            class MockContextManager:
                async def __aenter__(self):
                    raise Exception("Network error")

                async def __aexit__(self, *args):
                    pass

            return MockContextManager()

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get = mock_get_always_fails
            mock_session.return_value.__aexit__.return_value = AsyncMock()

            with patch('asyncio.sleep', side_effect=mock_sleep):
                with pytest.raises(NetworkError):
                    await downloader.download_url("https://example.com/backoff.txt")

            # Should have delays: 1s, 2s, 4s (exponential backoff)
            assert len(sleep_calls) == 3
            assert sleep_calls[0] == 1.0
            assert sleep_calls[1] == 2.0
            assert sleep_calls[2] == 4.0


class TestURLDownloaderMimeValidation:
    """Test MIME type validation."""

    @pytest.mark.asyncio
    async def test_unsupported_mime_type(self):
        """Test that unsupported MIME types are rejected."""
        downloader = URLDownloader()

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'image/jpeg'}
            mock_response.read = AsyncMock(return_value=b'JPEG data')
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(UnsupportedMimeTypeError) as exc_info:
                await downloader.download_url("https://example.com/image.jpg")

            assert "image/jpeg" in str(exc_info.value)
            assert "not supported" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_mime_type_with_charset(self):
        """Test MIME type parsing with charset parameter."""
        downloader = URLDownloader()

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
            mock_response.read = AsyncMock(return_value=b'<html>Content</html>')
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await downloader.download_url("https://example.com/page.html")

            # Should extract base MIME type without charset
            assert result["mime_type"] == "text/html"

    @pytest.mark.asyncio
    async def test_custom_allowed_mime_types(self):
        """Test custom allowed MIME types configuration."""
        downloader = URLDownloader(
            allowed_mime_types=['application/json', 'text/csv']
        )

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            mock_response.read = AsyncMock(return_value=b'{"key": "value"}')
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await downloader.download_url("https://example.com/data.json")

            assert result["mime_type"] == "application/json"
            assert result["content"] == b'{"key": "value"}'


class TestURLDownloaderTimeout:
    """Test timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_default(self):
        """Test that default timeout is enforced."""
        downloader = URLDownloader(timeout=0.1)  # 100ms timeout

        with patch('aiohttp.ClientSession.get') as mock_get:
            # Simulate slow response
            async def slow_read():
                await asyncio.sleep(1.0)  # Longer than timeout
                return b'Should not reach here'

            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'text/plain'}
            mock_response.read = slow_read
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises((asyncio.TimeoutError, NetworkError)):
                await downloader.download_url("https://example.com/slow.txt")

    @pytest.mark.asyncio
    async def test_custom_timeout(self):
        """Test custom timeout configuration."""
        downloader = URLDownloader(timeout=5.0)  # 5 second timeout

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'text/plain'}
            mock_response.read = AsyncMock(return_value=b'Success')
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await downloader.download_url("https://example.com/file.txt", timeout=5)

            assert result["content"] == b'Success'


class TestURLDownloaderHTTPErrors:
    """Test HTTP error handling."""

    @pytest.mark.asyncio
    async def test_404_not_found(self):
        """Test handling of 404 Not Found errors."""
        downloader = URLDownloader()

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 404
            mock_response.reason = "Not Found"
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(DownloadError) as exc_info:
                await downloader.download_url("https://example.com/missing.pdf")

            assert "404" in str(exc_info.value)
            assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_403_forbidden(self):
        """Test handling of 403 Forbidden errors."""
        downloader = URLDownloader()

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 403
            mock_response.reason = "Forbidden"
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(DownloadError) as exc_info:
                await downloader.download_url("https://example.com/forbidden.pdf")

            assert "403" in str(exc_info.value)
            assert "forbidden" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_500_server_error(self):
        """Test handling of 500 Internal Server Error."""
        downloader = URLDownloader()

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 500
            mock_response.reason = "Internal Server Error"
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(DownloadError) as exc_info:
                await downloader.download_url("https://example.com/error.pdf")

            assert "500" in str(exc_info.value)


class TestURLDownloaderRedirects:
    """Test redirect handling."""

    @pytest.mark.asyncio
    async def test_redirect_tracking(self):
        """Test that redirects are tracked in response."""
        downloader = URLDownloader()

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/pdf'}
            mock_response.read = AsyncMock(return_value=b'PDF content')
            mock_response.url = "https://cdn.example.com/final.pdf"  # Redirected URL
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await downloader.download_url("https://example.com/redirect.pdf")

            assert result["url"] == "https://example.com/redirect.pdf"
            assert result["redirected_url"] == "https://cdn.example.com/final.pdf"


class TestURLDownloaderUserAgent:
    """Test User-Agent header configuration."""

    @pytest.mark.asyncio
    async def test_default_user_agent(self):
        """Test that default User-Agent is sent."""
        downloader = URLDownloader()

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'text/plain'}
            mock_response.read = AsyncMock(return_value=b'Content')
            mock_get.return_value.__aenter__.return_value = mock_response

            await downloader.download_url("https://example.com/test.txt")

            # Verify User-Agent was sent
            call_kwargs = mock_get.call_args[1]
            assert 'headers' in call_kwargs
            assert 'User-Agent' in call_kwargs['headers']
            assert 'Dionysus' in call_kwargs['headers']['User-Agent']

    @pytest.mark.asyncio
    async def test_custom_user_agent(self):
        """Test custom User-Agent configuration."""
        custom_agent = "CustomBot/1.0"
        downloader = URLDownloader(user_agent=custom_agent)

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'text/plain'}
            mock_response.read = AsyncMock(return_value=b'Content')
            mock_get.return_value.__aenter__.return_value = mock_response

            await downloader.download_url("https://example.com/test.txt")

            call_kwargs = mock_get.call_args[1]
            assert call_kwargs['headers']['User-Agent'] == custom_agent


class TestURLDownloaderEdgeCases:
    """Test edge cases and validation."""

    @pytest.mark.asyncio
    async def test_invalid_url_scheme(self):
        """Test that non-HTTPS URLs are rejected."""
        downloader = URLDownloader()

        with pytest.raises(ValueError) as exc_info:
            await downloader.download_url("http://example.com/file.pdf")

        assert "https" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_empty_url(self):
        """Test that empty URLs are rejected."""
        downloader = URLDownloader()

        with pytest.raises(ValueError):
            await downloader.download_url("")

    @pytest.mark.asyncio
    async def test_malformed_url(self):
        """Test that malformed URLs are rejected."""
        downloader = URLDownloader()

        with pytest.raises(ValueError):
            await downloader.download_url("not a valid url")

    @pytest.mark.asyncio
    async def test_large_file_download(self):
        """Test download of large file (10MB)."""
        downloader = URLDownloader()

        large_content = b'A' * (10 * 1024 * 1024)  # 10MB

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {
                'Content-Type': 'application/pdf',
                'Content-Length': str(len(large_content))
            }
            mock_response.read = AsyncMock(return_value=large_content)
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await downloader.download_url("https://example.com/large.pdf")

            assert result["size_bytes"] == len(large_content)
            assert result["content"] == large_content

    @pytest.mark.asyncio
    async def test_empty_response_body(self):
        """Test handling of empty response body."""
        downloader = URLDownloader()

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'text/plain'}
            mock_response.read = AsyncMock(return_value=b'')
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await downloader.download_url("https://example.com/empty.txt")

            assert result["content"] == b''
            assert result["size_bytes"] == 0
