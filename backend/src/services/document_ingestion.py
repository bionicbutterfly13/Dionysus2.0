"""
Multi-Format Document Ingestion Service
======================================

Handles ingestion and processing of multiple document formats:
- PDF files (PyMuPDF/pdfplumber)
- Plain text files (.txt, .md)
- Web links (BeautifulSoup)
- Document type auto-detection
- Metadata extraction and preservation

Implements Spec-022 Task 1.2 requirements.
"""

import asyncio
import logging
import mimetypes
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass, field
from urllib.parse import urlparse
import hashlib

# Document processing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    import requests
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    requests = None
    WEB_SCRAPING_AVAILABLE = False

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    WEB_URL = "web_url"
    UNKNOWN = "unknown"

class ProcessingStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"

@dataclass
class DocumentMetadata:
    """Metadata extracted from documents"""
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    url: Optional[str] = None
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    encoding: Optional[str] = None

@dataclass 
class DocumentContent:
    """Processed document content"""
    raw_text: str
    structured_content: List[Dict[str, Any]] = field(default_factory=list)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    processing_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IngestionResult:
    """Result of document ingestion process"""
    success: bool
    document_id: str
    document_type: DocumentType
    content: Optional[DocumentContent] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)

class DocumentTypeDetector:
    """Detects document types from various inputs"""
    
    @staticmethod
    def detect_from_path(file_path: Union[str, Path]) -> DocumentType:
        """Detect document type from file path"""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return DocumentType.PDF
        elif suffix in ['.txt', '.text']:
            return DocumentType.PLAIN_TEXT
        elif suffix in ['.md', '.markdown']:
            return DocumentType.MARKDOWN
        else:
            return DocumentType.UNKNOWN
    
    @staticmethod
    def detect_from_url(url: str) -> DocumentType:
        """Detect document type from URL"""
        parsed = urlparse(url)
        
        # Check for obvious file extensions in URL
        path = parsed.path.lower()
        if path.endswith('.pdf'):
            return DocumentType.PDF
        elif path.endswith(('.txt', '.text')):
            return DocumentType.PLAIN_TEXT
        elif path.endswith(('.md', '.markdown')):
            return DocumentType.MARKDOWN
        else:
            # Assume web content for other URLs
            return DocumentType.WEB_URL
    
    @staticmethod
    def detect_from_content(content: bytes, mime_type: Optional[str] = None) -> DocumentType:
        """Detect document type from content"""
        if mime_type:
            if 'pdf' in mime_type:
                return DocumentType.PDF
            elif 'text' in mime_type:
                return DocumentType.PLAIN_TEXT
            elif 'html' in mime_type:
                return DocumentType.WEB_URL
        
        # Check magic bytes
        if content.startswith(b'%PDF'):
            return DocumentType.PDF
        elif content.startswith(b'<!DOCTYPE html') or b'<html' in content[:100]:
            return DocumentType.WEB_URL
        else:
            # Try to decode as text
            try:
                content.decode('utf-8')
                return DocumentType.PLAIN_TEXT
            except UnicodeDecodeError:
                return DocumentType.UNKNOWN

class PDFProcessor:
    """Processes PDF documents using PyMuPDF and pdfplumber"""
    
    @staticmethod
    async def extract_content(file_path: str) -> DocumentContent:
        """Extract content from PDF file"""
        if not PYMUPDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise ValueError("No PDF processing library available. Install PyMuPDF or pdfplumber")
        
        # Try PyMuPDF first (faster)
        if PYMUPDF_AVAILABLE:
            return await PDFProcessor._extract_with_pymupdf(file_path)
        else:
            return await PDFProcessor._extract_with_pdfplumber(file_path)
    
    @staticmethod
    async def _extract_with_pymupdf(file_path: str) -> DocumentContent:
        """Extract using PyMuPDF"""
        doc = fitz.open(file_path)
        
        # Extract metadata
        metadata = DocumentMetadata(
            title=doc.metadata.get('title'),
            author=doc.metadata.get('author'),
            subject=doc.metadata.get('subject'),
            page_count=doc.page_count,
            creation_date=datetime.fromisoformat(doc.metadata.get('creationDate', '').replace('Z', '+00:00')) if doc.metadata.get('creationDate') else None,
            modification_date=datetime.fromisoformat(doc.metadata.get('modDate', '').replace('Z', '+00:00')) if doc.metadata.get('modDate') else None,
        )
        
        # Extract text and structure
        raw_text = ""
        structured_content = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            raw_text += page_text + "\n\n"
            
            # Extract structured elements
            page_dict = page.get_text("dict")
            structured_content.append({
                "page_number": page_num + 1,
                "text": page_text,
                "blocks": page_dict.get("blocks", []),
                "bbox": page.rect,
            })
        
        doc.close()
        
        # Count words
        word_count = len(raw_text.split())
        metadata.word_count = word_count
        
        return DocumentContent(
            raw_text=raw_text.strip(),
            structured_content=structured_content,
            metadata=metadata,
            processing_info={"processor": "PyMuPDF", "version": fitz.version}
        )
    
    @staticmethod
    async def _extract_with_pdfplumber(file_path: str) -> DocumentContent:
        """Extract using pdfplumber"""
        raw_text = ""
        structured_content = []
        
        with pdfplumber.open(file_path) as pdf:
            # Extract metadata
            metadata = DocumentMetadata(
                title=pdf.metadata.get('Title'),
                author=pdf.metadata.get('Author'),
                subject=pdf.metadata.get('Subject'),
                page_count=len(pdf.pages),
                creation_date=pdf.metadata.get('CreationDate'),
                modification_date=pdf.metadata.get('ModDate'),
            )
            
            # Extract text and structure
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                raw_text += page_text + "\n\n"
                
                # Extract tables and other elements
                tables = page.extract_tables()
                
                structured_content.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "tables": tables,
                    "bbox": page.bbox,
                })
        
        # Count words
        word_count = len(raw_text.split())
        metadata.word_count = word_count
        
        return DocumentContent(
            raw_text=raw_text.strip(),
            structured_content=structured_content,
            metadata=metadata,
            processing_info={"processor": "pdfplumber"}
        )

class TextProcessor:
    """Processes plain text and markdown files"""
    
    @staticmethod
    async def extract_content(file_path: str, encoding: str = 'utf-8') -> DocumentContent:
        """Extract content from text/markdown file"""
        path = Path(file_path)
        
        # Try different encodings if utf-8 fails
        encodings = [encoding, 'utf-8', 'latin-1', 'cp1252']
        content = None
        used_encoding = None
        
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                used_encoding = enc
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"Could not decode file {file_path} with any encoding")
        
        # Extract metadata
        metadata = DocumentMetadata(
            title=path.stem,
            file_size=path.stat().st_size,
            creation_date=datetime.fromtimestamp(path.stat().st_ctime),
            modification_date=datetime.fromtimestamp(path.stat().st_mtime),
            word_count=len(content.split()),
            encoding=used_encoding,
        )
        
        # Extract structure for markdown
        structured_content = []
        if path.suffix.lower() in ['.md', '.markdown']:
            structured_content = TextProcessor._parse_markdown_structure(content)
        
        return DocumentContent(
            raw_text=content,
            structured_content=structured_content,
            metadata=metadata,
            processing_info={"processor": "TextProcessor", "encoding": used_encoding}
        )
    
    @staticmethod
    def _parse_markdown_structure(content: str) -> List[Dict[str, Any]]:
        """Parse markdown structure"""
        lines = content.split('\n')
        structure = []
        current_section = None
        
        for line_num, line in enumerate(lines):
            # Check for headers
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                if current_section:
                    structure.append(current_section)
                
                current_section = {
                    "type": "header",
                    "level": level,
                    "title": title,
                    "line_number": line_num + 1,
                    "content": ""
                }
            elif current_section:
                current_section["content"] += line + "\n"
        
        if current_section:
            structure.append(current_section)
        
        return structure

class WebProcessor:
    """Processes web URLs and extracts content"""
    
    @staticmethod
    async def extract_content(url: str, timeout: int = 30) -> DocumentContent:
        """Extract content from web URL"""
        if not WEB_SCRAPING_AVAILABLE:
            raise ValueError("Web scraping libraries not available. Install requests and beautifulsoup4")
        
        # Fetch content
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else None
        
        # Extract meta tags
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        meta_author = soup.find('meta', attrs={'name': 'author'})
        
        metadata = DocumentMetadata(
            title=title_text,
            author=meta_author.get('content') if meta_author else None,
            subject=meta_desc.get('content') if meta_desc else None,
            keywords=meta_keywords.get('content').split(',') if meta_keywords else [],
            url=url,
            content_type=response.headers.get('content-type'),
        )
        
        # Extract main content
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content
        text_content = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = ' '.join(chunk for chunk in chunks if chunk)
        
        metadata.word_count = len(text_content.split())
        
        # Extract structured content
        structured_content = WebProcessor._extract_structure(soup)
        
        return DocumentContent(
            raw_text=text_content,
            structured_content=structured_content,
            metadata=metadata,
            processing_info={"processor": "WebProcessor", "url": url, "status_code": response.status_code}
        )
    
    @staticmethod
    def _extract_structure(soup) -> List[Dict[str, Any]]:
        """Extract structured elements from HTML"""
        structure = []
        
        # Extract headers
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            structure.append({
                "type": "header",
                "level": int(header.name[1]),
                "title": header.get_text().strip(),
                "tag": header.name
            })
        
        # Extract paragraphs
        for para in soup.find_all('p'):
            text = para.get_text().strip()
            if text:
                structure.append({
                    "type": "paragraph",
                    "text": text
                })
        
        # Extract lists
        for list_elem in soup.find_all(['ul', 'ol']):
            items = [li.get_text().strip() for li in list_elem.find_all('li')]
            structure.append({
                "type": "list",
                "list_type": list_elem.name,
                "items": items
            })
        
        return structure

class DocumentIngestionService:
    """Main service for multi-format document ingestion"""
    
    def __init__(self):
        self.detector = DocumentTypeDetector()
        self.processors = {
            DocumentType.PDF: PDFProcessor(),
            DocumentType.PLAIN_TEXT: TextProcessor(),
            DocumentType.MARKDOWN: TextProcessor(),
            DocumentType.WEB_URL: WebProcessor(),
        }
    
    async def ingest_document(self, 
                            source: Union[str, Path, bytes],
                            source_type: str = "auto",
                            **kwargs) -> IngestionResult:
        """
        Ingest a document from various sources
        
        Args:
            source: File path, URL, or bytes content
            source_type: "file", "url", "bytes", or "auto"
            **kwargs: Additional processing options
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate document ID
            if isinstance(source, (str, Path)):
                doc_id = hashlib.md5(str(source).encode()).hexdigest()
            else:
                doc_id = hashlib.md5(source).hexdigest()
            
            # Auto-detect source type
            if source_type == "auto":
                source_type = self._detect_source_type(source)
            
            # Detect document type
            doc_type = self._detect_document_type(source, source_type)
            
            if doc_type == DocumentType.UNKNOWN:
                return IngestionResult(
                    success=False,
                    document_id=doc_id,
                    document_type=doc_type,
                    error="Unsupported document type",
                    processing_time=asyncio.get_event_loop().time() - start_time
                )
            
            # Check if processor is available
            if doc_type not in self.processors:
                return IngestionResult(
                    success=False,
                    document_id=doc_id,
                    document_type=doc_type,
                    error=f"No processor available for {doc_type.value}",
                    processing_time=asyncio.get_event_loop().time() - start_time
                )
            
            # Process document
            processor = self.processors[doc_type]
            
            if source_type == "url":
                content = await processor.extract_content(str(source), **kwargs)
            elif source_type == "file":
                content = await processor.extract_content(str(source), **kwargs)
            else:
                return IngestionResult(
                    success=False,
                    document_id=doc_id,
                    document_type=doc_type,
                    error=f"Unsupported source type: {source_type}",
                    processing_time=asyncio.get_event_loop().time() - start_time
                )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return IngestionResult(
                success=True,
                document_id=doc_id,
                document_type=doc_type,
                content=content,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return IngestionResult(
                success=False,
                document_id=doc_id if 'doc_id' in locals() else "unknown",
                document_type=doc_type if 'doc_type' in locals() else DocumentType.UNKNOWN,
                error=str(e),
                processing_time=asyncio.get_event_loop().time() - start_time
            )
    
    def _detect_source_type(self, source: Union[str, Path, bytes]) -> str:
        """Detect whether source is file, url, or bytes"""
        if isinstance(source, bytes):
            return "bytes"
        elif isinstance(source, (str, Path)):
            source_str = str(source)
            if source_str.startswith(('http://', 'https://')):
                return "url"
            else:
                return "file"
        else:
            return "unknown"
    
    def _detect_document_type(self, source: Union[str, Path, bytes], source_type: str) -> DocumentType:
        """Detect document type based on source and type"""
        if source_type == "url":
            return self.detector.detect_from_url(str(source))
        elif source_type == "file":
            return self.detector.detect_from_path(source)
        elif source_type == "bytes":
            return self.detector.detect_from_content(source)
        else:
            return DocumentType.UNKNOWN
    
    async def get_supported_formats(self) -> Dict[str, bool]:
        """Get list of supported formats and their availability"""
        return {
            "pdf": PYMUPDF_AVAILABLE or PDFPLUMBER_AVAILABLE,
            "text": True,
            "markdown": True,
            "web_urls": WEB_SCRAPING_AVAILABLE,
        }

# Global service instance
document_ingestion_service = DocumentIngestionService()

# Test function
async def test_ingestion_service():
    """Test the document ingestion service"""
    service = DocumentIngestionService()
    
    # Test supported formats
    formats = await service.get_supported_formats()
    print("📋 Supported formats:")
    for fmt, available in formats.items():
        status = "✅" if available else "❌"
        print(f"  {status} {fmt}")
    
    # Test text processing
    test_text = "This is a test document with multiple sentences. It contains neuroscience concepts like synaptic plasticity."
    
    print("\n🧪 Testing text ingestion...")
    try:
        # Create temporary test file
        test_file = Path("/tmp/test_document.txt")
        test_file.write_text(test_text)
        
        result = await service.ingest_document(test_file)
        if result.success:
            print(f"✅ Text ingestion successful")
            print(f"  Document ID: {result.document_id}")
            print(f"  Type: {result.document_type.value}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            print(f"  Word count: {result.content.metadata.word_count}")
        else:
            print(f"❌ Text ingestion failed: {result.error}")
        
        # Cleanup
        test_file.unlink()
        
    except Exception as e:
        print(f"❌ Text ingestion error: {e}")
    
    print("\n🎉 Ingestion service test completed!")

if __name__ == "__main__":
    asyncio.run(test_ingestion_service())