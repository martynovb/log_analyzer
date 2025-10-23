"""
Context Retrieval Module

This module handles retrieving relevant context from codebase and documentation
based on extracted keywords. Includes interfaces for different context sources.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ContextType(Enum):
    """Types of context that can be retrieved."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    API_SPEC = "api_spec"
    ERROR_PATTERNS = "error_patterns"


@dataclass
class ContextItem:
    """Represents a piece of context with metadata."""
    content: str
    context_type: ContextType
    source: str
    relevance_score: float
    metadata: Dict[str, Any]


class ContextSource(ABC):
    """Abstract base class for context sources."""
    
    @abstractmethod
    def get_context(self, keywords: List[str]) -> List[ContextItem]:
        """
        Retrieve context based on keywords.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of context items
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of this context source."""
        pass


class MockCodebaseContextSource(ContextSource):
    """Mock implementation for codebase context retrieval."""
    
    def __init__(self):
        """Initialize with mock codebase data."""
        self.mock_codebase = {
            'authentication': [
                "class AuthService:\n    def authenticate_user(self, username, password):\n        # Implementation here\n        pass",
                "def validate_jwt_token(token):\n    # JWT validation logic\n    return True"
            ],
            'database': [
                "class DatabaseManager:\n    def __init__(self):\n        self.connection = None\n    def connect(self):\n        # Connection logic\n        pass",
                "def execute_query(query, params):\n    # Query execution\n    return results"
            ],
            'error': [
                "class ErrorHandler:\n    def handle_exception(self, exc):\n        logger.error(f'Exception occurred: {exc}')\n        return error_response",
                "def log_error(error_message, context):\n    # Error logging implementation\n    pass"
            ],
            'api': [
                "class APIController:\n    def handle_request(self, request):\n        # Request handling logic\n        return response",
                "def validate_api_key(api_key):\n    # API key validation\n    return is_valid"
            ],
            'performance': [
                "class PerformanceMonitor:\n    def track_metrics(self):\n        # Performance tracking\n        pass",
                "def optimize_query(query):\n    # Query optimization\n    return optimized_query"
            ]
        }
    
    def get_context(self, keywords: List[str]) -> List[ContextItem]:
        """Retrieve mock codebase context."""
        context_items = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Find relevant code snippets
            for category, code_snippets in self.mock_codebase.items():
                if keyword_lower in category or any(keyword_lower in snippet.lower() for snippet in code_snippets):
                    for snippet in code_snippets:
                        context_items.append(ContextItem(
                            content=snippet,
                            context_type=ContextType.CODE,
                            source=f"mock_codebase/{category}",
                            relevance_score=0.8,
                            metadata={"category": category, "keyword": keyword}
                        ))
        
        return context_items[:10]  # Limit to 10 items
    
    def get_source_name(self) -> str:
        return "Mock Codebase"


class MockDocumentationContextSource(ContextSource):
    """Mock implementation for documentation context retrieval."""
    
    def __init__(self):
        """Initialize with mock documentation data."""
        self.mock_docs = {
            'authentication': [
                "Authentication Guide:\nJWT tokens are used for user authentication. Tokens expire after 24 hours.",
                "Security Best Practices:\nAlways validate tokens on the server side. Use HTTPS for all authentication requests."
            ],
            'database': [
                "Database Configuration:\nThe application uses PostgreSQL as the primary database. Connection pooling is enabled.",
                "Query Optimization:\nUse indexes for frequently queried columns. Avoid N+1 queries by using proper joins."
            ],
            'error': [
                "Error Handling:\nAll errors should be logged with appropriate context. Use structured logging for better debugging.",
                "Exception Management:\nCatch specific exceptions rather than generic ones. Always provide meaningful error messages."
            ],
            'api': [
                "API Design:\nFollow RESTful principles. Use appropriate HTTP status codes. Include proper error responses.",
                "Rate Limiting:\nImplement rate limiting to prevent abuse. Return appropriate headers with rate limit information."
            ],
            'performance': [
                "Performance Monitoring:\nUse APM tools to monitor application performance. Set up alerts for performance degradation.",
                "Caching Strategy:\nImplement caching at multiple levels: application, database, and CDN."
            ]
        }
    
    def get_context(self, keywords: List[str]) -> List[ContextItem]:
        """Retrieve mock documentation context."""
        context_items = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Find relevant documentation
            for category, docs in self.mock_docs.items():
                if keyword_lower in category or any(keyword_lower in doc.lower() for doc in docs):
                    for doc in docs:
                        context_items.append(ContextItem(
                            content=doc,
                            context_type=ContextType.DOCUMENTATION,
                            source=f"mock_docs/{category}",
                            relevance_score=0.7,
                            metadata={"category": category, "keyword": keyword}
                        ))
        
        return context_items[:8]  # Limit to 8 items
    
    def get_source_name(self) -> str:
        return "Mock Documentation"


class MockConfigurationContextSource(ContextSource):
    """Mock implementation for configuration context retrieval."""
    
    def __init__(self):
        """Initialize with mock configuration data."""
        self.mock_configs = {
            'database': [
                "DATABASE_URL=postgresql://user:pass@localhost:5432/appdb\nDB_POOL_SIZE=20\nDB_TIMEOUT=30",
                "REDIS_URL=redis://localhost:6379\nREDIS_TTL=3600\nREDIS_MAX_CONNECTIONS=100"
            ],
            'authentication': [
                "JWT_SECRET=your-secret-key\nJWT_EXPIRY=86400\nAUTH_PROVIDER=local",
                "OAUTH_CLIENT_ID=your-client-id\nOAUTH_CLIENT_SECRET=your-secret\nOAUTH_REDIRECT_URI=http://localhost:3000/callback"
            ],
            'logging': [
                "LOG_LEVEL=INFO\nLOG_FORMAT=json\nLOG_FILE=/var/log/app.log",
                "LOG_ROTATION_SIZE=100MB\nLOG_RETENTION_DAYS=30\nLOG_COMPRESSION=true"
            ],
            'performance': [
                "CACHE_TTL=3600\nCACHE_MAX_SIZE=1000MB\nCACHE_STRATEGY=LRU",
                "MAX_CONCURRENT_REQUESTS=1000\nREQUEST_TIMEOUT=30s\nRATE_LIMIT=100/min"
            ]
        }
    
    def get_context(self, keywords: List[str]) -> List[ContextItem]:
        """Retrieve mock configuration context."""
        context_items = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Find relevant configurations
            for category, configs in self.mock_configs.items():
                if keyword_lower in category or any(keyword_lower in config.lower() for config in configs):
                    for config in configs:
                        context_items.append(ContextItem(
                            content=config,
                            context_type=ContextType.CONFIGURATION,
                            source=f"mock_config/{category}",
                            relevance_score=0.6,
                            metadata={"category": category, "keyword": keyword}
                        ))
        
        return context_items[:5]  # Limit to 5 items
    
    def get_source_name(self) -> str:
        return "Mock Configuration"


class ContextRetriever:
    """Main class for retrieving context from multiple sources."""
    
    def __init__(self):
        """Initialize with available context sources."""
        self.sources = {
            'codebase': MockCodebaseContextSource(),
            'documentation': MockDocumentationContextSource(),
            'configuration': MockConfigurationContextSource()
        }
    
    def retrieve_context(self, keywords: List[str], 
                        source_types: Optional[List[str]] = None) -> Dict[str, List[ContextItem]]:
        """
        Retrieve context from specified sources.
        
        Args:
            keywords: List of keywords to search for
            source_types: List of source types to query (None for all)
            
        Returns:
            Dictionary mapping source names to context items
        """
        if source_types is None:
            source_types = list(self.sources.keys())
        
        results = {}
        
        for source_type in source_types:
            if source_type in self.sources:
                source = self.sources[source_type]
                context_items = source.get_context(keywords)
                results[source.get_source_name()] = context_items
        
        return results
    
    def get_combined_context(self, keywords: List[str]) -> str:
        """
        Get combined context from all sources as a formatted string.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            Formatted string containing all relevant context
        """
        all_context = self.retrieve_context(keywords)
        
        formatted_context = "=== RELEVANT CONTEXT ===\n\n"
        
        for source_name, context_items in all_context.items():
            if context_items:
                formatted_context += f"## {source_name}\n\n"
                
                for item in context_items:
                    formatted_context += f"**Source:** {item.source}\n"
                    formatted_context += f"**Relevance:** {item.relevance_score:.2f}\n"
                    formatted_context += f"**Content:**\n```\n{item.content}\n```\n\n"
        
        return formatted_context
    
    def add_context_source(self, name: str, source: ContextSource):
        """
        Add a new context source.
        
        Args:
            name: Name of the source
            source: ContextSource implementation
        """
        self.sources[name] = source
    
    def get_available_sources(self) -> List[str]:
        """Get list of available context sources."""
        return list(self.sources.keys())

