"""
Flux API Routes
Route handlers for documents, curiosity, visualization, stats, query, and CLAUSE endpoints.
"""

from . import stats, query
# CLAUSE routes available but not auto-imported due to relative import issues
# Import manually: from src.api.routes import clause