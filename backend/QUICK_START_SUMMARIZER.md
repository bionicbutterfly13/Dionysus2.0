# Quick Start: Document Summarizer

## Basic Usage

### 1. Install Dependencies
```bash
pip install openai tiktoken
```

### 2. Set API Key
```bash
export OPENAI_API_KEY=sk-your-key-here
```

### 3. Use in Code

#### Simple Summarization
```python
from src.services.document_summarizer import summarize_document

# Quick summarization
result = await summarize_document(
    document_content="Your long document text here...",
    max_tokens=150
)

print(result["summary"])
# Output: "Brief summary of the document."
```

#### Advanced Usage
```python
from src.services.document_summarizer import DocumentSummarizer, SummarizerConfig

# Custom configuration
config = SummarizerConfig(
    model="gpt-3.5-turbo",
    max_tokens=200,
    temperature=0.3,
    api_key="sk-..."
)

summarizer = DocumentSummarizer(config)

# Generate summary
result = await summarizer.generate_summary(
    document_content="Your document...",
    max_tokens=150
)

print(f"Summary: {result['summary']}")
print(f"Method: {result['method']}")  # "llm" or "extractive"
print(f"Tokens: {result['tokens_used']}")
```

#### Integration with Repository
```python
from src.services.document_repository import DocumentRepository

repo = DocumentRepository()

# Summaries are automatically generated during persistence
result = await repo.persist_document(
    final_output={
        "extracted_text": "Document content...",
        "quality": {"scores": {"overall": 0.85}},
        "concepts": {...},
        "basins": [...],
        "thoughtseeds": [...]
    },
    metadata={
        "document_id": "doc_123",
        "filename": "paper.pdf",
        "content_hash": "sha256...",
        "file_size": 1024
    }
)

# Retrieve document with summary
document = await repo.get_document("doc_123")
print(document["summary"])
print(document["summary_metadata"])
```

## API Examples

### Upload Document (Auto-generates Summary)
```bash
curl -X POST http://localhost:8000/api/documents/persist \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_123",
    "filename": "paper.pdf",
    "content_hash": "sha256...",
    "file_size": 1024,
    "mime_type": "application/pdf",
    "tags": ["research"],
    "daedalus_output": {
      "extracted_text": "Document content...",
      "quality": {"scores": {"overall": 0.85}},
      "concepts": {},
      "basins": [],
      "thoughtseeds": []
    }
  }'
```

### Get Document with Summary
```bash
curl http://localhost:8000/api/documents/doc_123
```

Response:
```json
{
  "document_id": "doc_123",
  "metadata": {...},
  "summary": "Brief summary of the document...",
  "summary_metadata": {
    "method": "llm",
    "model": "gpt-3.5-turbo",
    "tokens_used": 165,
    "generated_at": "2025-10-07T10:00:00Z"
  },
  "concepts": {...},
  "basins": [...],
  "thoughtseeds": [...]
}
```

### List Documents with Summaries
```bash
curl http://localhost:8000/api/documents?page=1&limit=10
```

Response:
```json
{
  "documents": [
    {
      "document_id": "doc_123",
      "filename": "paper.pdf",
      "summary": "Brief summary...",
      "quality_overall": 0.85,
      ...
    }
  ],
  "pagination": {...}
}
```

## Configuration Options

### Token Budgets
```python
# Concise (50 tokens)
result = await summarizer.generate_summary(text, max_tokens=50)

# Standard (150 tokens) - DEFAULT
result = await summarizer.generate_summary(text, max_tokens=150)

# Detailed (300 tokens)
result = await summarizer.generate_summary(text, max_tokens=300)
```

### Models
```python
# Fast and cheap (default)
config = SummarizerConfig(model="gpt-3.5-turbo", max_tokens=150)

# More capable
config = SummarizerConfig(model="gpt-4", max_tokens=150)

# Specific version
config = SummarizerConfig(model="gpt-3.5-turbo-0125", max_tokens=150)
```

### Temperature
```python
# Deterministic (default)
config = SummarizerConfig(temperature=0.3)

# More creative
config = SummarizerConfig(temperature=0.7)

# Very deterministic
config = SummarizerConfig(temperature=0.0)
```

## Fallback Behavior

If OpenAI API is unavailable, summarizer automatically falls back to extractive method:

```python
result = await summarizer.generate_summary(text)

if result["method"] == "extractive":
    print("Fallback used:", result.get("error"))
    # Example: "LLM failed: AuthenticationError: Invalid API key"
```

## Testing

### Run Tests
```bash
# All tests
pytest tests/test_document_summarizer.py -v

# Specific test
pytest tests/test_document_summarizer.py::TestTokenCounting -v

# With coverage
pytest tests/test_document_summarizer.py --cov
```

### Run Demo
```bash
# With API key
OPENAI_API_KEY=sk-... python demo_summarizer.py

# Without API key (extractive fallback)
python demo_summarizer.py
```

## Troubleshooting

### API Key Issues
```
Error: OpenAI API key not found
Solution: export OPENAI_API_KEY=sk-your-key
```

### Rate Limits
```
Error: Rate limit exceeded
Solution: Automatic extractive fallback activated
```

### Token Budget Exceeded
```
# Input too large? Automatic truncation at 3000 tokens
# Output too large? Reduce max_tokens parameter
result = await summarizer.generate_summary(text, max_tokens=50)
```

## Performance Tips

1. **Batch Processing**: Use asyncio.gather() for multiple documents
2. **Caching**: Store summaries with content_hash to avoid regeneration
3. **Model Selection**: Use gpt-3.5-turbo for speed, gpt-4 for quality
4. **Token Budget**: Smaller budgets = faster + cheaper

## Cost Estimates

- **gpt-3.5-turbo**: ~$0.0001 per summary
- **gpt-4**: ~$0.001 per summary
- **1000 documents**: $0.10 (3.5) or $1.00 (4)
- **Extractive fallback**: Free (no API calls)

## Support

For issues or questions:
- Review tests: `tests/test_document_summarizer.py`
- Check logs: Look for "DocumentSummarizer" entries
- Run demo: `python demo_summarizer.py`
- Read spec: `SPEC_055_AGENT_3_SUMMARY.md`
