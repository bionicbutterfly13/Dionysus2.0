"""
Web Crawl API - Simplified from Archon
Crawls URLs and passes content through Daedalus gateway
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from datetime import datetime

from services.daedalus import Daedalus

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize Daedalus
daedalus = Daedalus()

class CrawlRequest(BaseModel):
    url: str
    max_depth: int = 2  # Default to depth 2
    tags: Optional[List[str]] = None


class CrawlResponse(BaseModel):
    success: bool
    message: str
    pages_crawled: int
    documents_processed: int


@router.post("/crawl")
async def crawl_url(request: CrawlRequest):
    """
    Crawl a URL and process through Daedalus.

    Simplified from Archon - no Supabase, no fancy progress tracking.
    Just crawl → Daedalus → Neo4j.
    """
    if not request.url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    logger.info(f"Starting crawl: {request.url} (depth: {request.max_depth})")

    try:
        # Configure browser
        browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )

        # Configure crawler
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=30000,  # 30 seconds
            wait_until="domcontentloaded"
        )

        pages_crawled = 0
        documents_processed = 0
        visited_urls = set()
        urls_to_visit = [(request.url, 0)]  # (url, depth)

        async with AsyncWebCrawler(config=browser_config) as crawler:
            while urls_to_visit:
                current_url, depth = urls_to_visit.pop(0)

                # Skip if already visited or depth exceeded
                if current_url in visited_urls or depth > request.max_depth:
                    continue

                visited_urls.add(current_url)

                try:
                    logger.info(f"Crawling: {current_url} (depth {depth})")

                    # Crawl the page
                    result = await crawler.arun(
                        url=current_url,
                        config=run_config
                    )

                    if not result.success:
                        logger.warning(f"Failed to crawl {current_url}: {result.error_message}")
                        continue

                    pages_crawled += 1

                    # Pass content through Daedalus
                    # Convert markdown content to bytes-like for Daedalus
                    import io
                    content = result.markdown_v2.raw_markdown if result.markdown_v2 else result.markdown
                    file_obj = io.BytesIO(content.encode('utf-8'))
                    file_obj.name = f"{current_url.split('/')[-1] or 'index'}.md"

                    daedalus_response = daedalus.receive_perceptual_information(
                        data=file_obj,
                        tags=request.tags or [f"crawl_depth_{depth}", "web_crawl"],
                        max_iterations=2,  # Faster processing for web content
                        quality_threshold=0.6
                    )

                    if daedalus_response.get('status') != 'error':
                        documents_processed += 1
                        logger.info(f"✅ Processed {current_url}: {daedalus_response.get('extraction', {}).get('concepts', []).__len__()} concepts")
                    else:
                        logger.warning(f"⚠️ Daedalus error for {current_url}: {daedalus_response.get('error_message')}")

                    # Extract links for recursive crawling (if depth allows)
                    if depth < request.max_depth and result.links:
                        # Get internal links (same domain)
                        from urllib.parse import urlparse
                        base_domain = urlparse(request.url).netloc

                        for link_data in (result.links.get('internal', []) if isinstance(result.links, dict) else []):
                            link_url = link_data.get('href') if isinstance(link_data, dict) else link_data
                            if link_url and link_url not in visited_urls:
                                # Only crawl same domain
                                link_domain = urlparse(link_url).netloc
                                if link_domain == base_domain or not link_domain:
                                    urls_to_visit.append((link_url, depth + 1))

                except Exception as e:
                    logger.error(f"Error crawling {current_url}: {str(e)}")
                    continue

                # Rate limiting - be nice to servers
                await asyncio.sleep(0.5)

        logger.info(f"✅ Crawl complete: {pages_crawled} pages, {documents_processed} processed")

        return CrawlResponse(
            success=True,
            message=f"Crawled {pages_crawled} pages successfully",
            pages_crawled=pages_crawled,
            documents_processed=documents_processed
        )

    except Exception as e:
        logger.error(f"Crawl failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")
