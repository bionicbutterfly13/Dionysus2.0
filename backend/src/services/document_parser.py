"""
Document Parser - Extract concepts from uploaded files
Connects upload → parsing → knowledge extraction
"""
import io
import logging
from pathlib import Path
from typing import Dict, Any, List
import PyPDF2

logger = logging.getLogger(__name__)

class DocumentParser:
    """Parse uploaded documents and extract concepts"""

    def parse(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Parse document and extract concepts.

        Examples:
        >>> parser = DocumentParser()
        >>> result = parser.parse(pdf_bytes, "paper.pdf")
        >>> result['concepts']  # ["machine learning", "neural networks"]
        >>> result['text'][:100]  # First 100 chars of extracted text
        """
        try:
            # Determine file type
            if filename.endswith('.pdf'):
                return self._parse_pdf(file_content, filename)
            elif filename.endswith(('.txt', '.md')):
                return self._parse_text(file_content, filename)
            else:
                return {
                    'status': 'unsupported',
                    'filename': filename,
                    'error': f'Unsupported file type'
                }
        except Exception as e:
            logger.error(f"Parse error {filename}: {e}")
            return {
                'status': 'error',
                'filename': filename,
                'error': str(e)
            }

    def _parse_pdf(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Extract text and concepts from PDF"""
        reader = PyPDF2.PdfReader(io.BytesIO(content))

        # Extract text
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Extract concepts (simple: find capitalized phrases)
        concepts = self._extract_concepts(text)

        return {
            'status': 'success',
            'filename': filename,
            'text': text,
            'concepts': concepts,
            'pages': len(reader.pages),
            'word_count': len(text.split())
        }

    def _parse_text(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Extract text and concepts from text file"""
        text = content.decode('utf-8')
        concepts = self._extract_concepts(text)

        return {
            'status': 'success',
            'filename': filename,
            'text': text,
            'concepts': concepts,
            'word_count': len(text.split())
        }

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text.

        Simple extraction strategy:
        - 2-3 word noun phrases
        - Filter out common words
        - Sort by phrase length (longer = more specific)

        TODO: Use five_level_concept_extraction for deep semantic analysis
        """
        import re

        # Common words to filter out
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those'
        }

        # Extract 2-3 word phrases
        text_lower = text.lower()
        pattern = r'\b([a-z]+\s+[a-z]+(?:\s+[a-z]+)?)\b'
        phrases = re.findall(pattern, text_lower)

        # Filter phrases
        concepts = []
        for phrase in phrases:
            words = phrase.split()

            # Skip if too short
            if len(phrase) < 8:
                continue

            # Skip if starts with stopword
            if words[0] in stopwords:
                continue

            # Skip if all words are stopwords
            if all(w in stopwords for w in words):
                continue

            concepts.append(phrase)

        # Remove duplicates
        concepts = list(set(concepts))

        # Sort by length (longer phrases are more specific)
        concepts.sort(key=len, reverse=True)

        return concepts[:50]  # Top 50
