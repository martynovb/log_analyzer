"""
Context Retrieval Module

This module handles retrieving relevant context from codebase and documentation
based on extracted keywords. Includes interfaces for different context sources.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


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


class JSONContextRetriever(ContextSource):
    """Context retriever that reads from JSON files."""
    
    def __init__(self, context_base_path: str = "assets/context"):
        """
        Initialize with path to context JSON files.
        
        Args:
            context_base_path: Base directory containing context JSON files
        """
        self.context_base_path = Path(context_base_path)
        self._cache = {}
    
    def _load_json_file(self, file_path: Path) -> List[Dict]:
        """Load and cache JSON file content."""
        if str(file_path) not in self._cache:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self._cache[str(file_path)] = json.load(f)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                self._cache[str(file_path)] = []
        return self._cache[str(file_path)]
    
    def _search_items(self, items: List[Dict], keywords: List[str]) -> List[Dict]:
        """
        Search items by matching keywords.
        
        Args:
            items: List of items to search
            keywords: Keywords to match
            
        Returns:
            List of matching items with relevance scores
        """
        results = []
        keyword_set = set(k.lower() for k in keywords)
        
        for item in items:
            item_keywords = item.get('keywords', [])
            item_keyword_set = set(k.lower() for k in item_keywords)
            
            # Calculate match score
            matches = keyword_set.intersection(item_keyword_set)
            if matches:
                # Calculate relevance based on number of matches
                relevance_score = len(matches) / len(keyword_set) if keyword_set else 0
                
                # Also check if keywords appear in description
                description = item.get('description', '').lower()
                for keyword in keywords:
                    if keyword.lower() in description:
                        relevance_score += 0.2
                
                # Ensure relevance doesn't exceed 1.0
                relevance_score = min(relevance_score, 1.0)
                
                results.append({
                    'item': item,
                    'relevance': relevance_score,
                    'matched_keywords': list(matches)
                })
        
        # Sort by relevance (highest first)
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results
    
    def get_context(self, keywords: List[str], 
                   context_type: str = None,
                   max_results: int = 10) -> List[ContextItem]:
        """
        Retrieve context from JSON files based on keywords.
        
        Args:
            keywords: List of keywords to search for
            context_type: Type of context to retrieve (components, docs, errors) or None for all
            max_results: Maximum number of results to return per type
            
        Returns:
            List of context items
        """
        context_items = []
        
        # Determine which context types to search
        context_types = [context_type] if context_type else ['components', 'docs', 'errors']
        
        for ctx_type in context_types:
            # Map context types to file paths
            file_mapping = {
                'components': 'components/components.json',
                'docs': 'docs/documentation.json',
                'errors': 'errors/errors.json'
            }
            
            if ctx_type not in file_mapping:
                continue
            
            file_path = self.context_base_path / file_mapping[ctx_type]
            
            if file_path.exists():
                items = self._load_json_file(file_path)
                matches = self._search_items(items, keywords)
                
                # Convert to ContextItem
                for match_data in matches[:max_results]:
                    item = match_data['item']
                    
                    # Create content string from item
                    content_parts = []
                    if 'title' in item:
                        content_parts.append(f"**Title:** {item['title']}")
                    if 'description' in item:
                        content_parts.append(f"**Description:** {item['description']}")
                    if 'content' in item:
                        content_parts.append(f"**Content:** {item['content']}")
                    if 'possible_causes' in item:
                        content_parts.append(f"**Possible Causes:** {', '.join(item['possible_causes'])}")
                    if 'solutions' in item:
                        content_parts.append(f"**Solutions:** {', '.join(item['solutions'])}")
                    if 'related_components' in item:
                        content_parts.append(f"**Related Components:** {', '.join(item['related_components'])}")
                    
                    content = '\n\n'.join(content_parts)
                    
                    context_items.append(ContextItem(
                        content=content,
                        context_type=self._map_context_type(ctx_type),
                        source=f"{ctx_type}/{item.get('id', 'unknown')}",
                        relevance_score=match_data['relevance'],
                        metadata={
                            'id': item.get('id', ''),
                            'category': item.get('category', ''),
                            'matched_keywords': match_data['matched_keywords']
                        }
                    ))
        
        return context_items
    
    def _map_context_type(self, ctx_type: str) -> ContextType:
        """Map string context type to ContextType enum."""
        mapping = {
            'components': ContextType.CODE,
            'docs': ContextType.DOCUMENTATION,
            'errors': ContextType.ERROR_PATTERNS
        }
        return mapping.get(ctx_type, ContextType.CODE)
    
    def get_source_name(self) -> str:
        return "JSON Context Database"


class ContextRetriever:
    """Main class for retrieving context from JSON database."""
    
    def __init__(self, context_base_path: str = "assets/context"):
        """
        Initialize with JSON-based context retrieval.
        
        Args:
            context_base_path: Base directory containing context JSON files
        """
        self.json_retriever = JSONContextRetriever(context_base_path)
    
    def retrieve_context(self, keywords: List[str], 
                        context_types: Optional[List[str]] = None,
                        max_results_per_type: int = 10) -> Dict[str, Any]:
        """
        Retrieve context from JSON database.
        
        Args:
            keywords: List of keywords to search for
            context_types: List of context types to query (None for all)
            max_results_per_type: Maximum results per context type
            
        Returns:
            Dictionary mapping context type to list of items
        """
        if context_types is None:
            context_types = ['components', 'docs', 'errors']
        
        results = {}
        
        for ctx_type in context_types:
            context_items = self.json_retriever.get_context(
                keywords, 
                context_type=ctx_type,
                max_results=max_results_per_type
            )
            
            # Convert to dict format
            results[ctx_type] = []
            for item in context_items:
                results[ctx_type].append({
                    'id': item.metadata.get('id', ''),
                    'title': item.metadata.get('title', ''),
                    'source': item.source,
                    'relevance': item.relevance_score,
                    'content': item.content,
                    'metadata': item.metadata
                })
        
        return results
    
    def retrieve_codebase_context(self, keywords: List[str]) -> Dict[str, Any]:
        """
        Retrieve codebase context (components).
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            Context information dictionary
        """
        results = self.retrieve_context(keywords, context_types=['components'])
        
        items = results.get('components', [])
        # Extract titles or ids for file list (backward compatibility)
        relevant_files = [item.get('title', item.get('id', 'Unknown')) for item in items]
        
        return {
            'relevant_files': relevant_files,
            'relevant_items': items,  # Also include full items
            'total_files': len(items),
            'total_items': len(items),
            'search_keywords': keywords,
            'retrieval_method': 'json_database'
        }
    
    def retrieve_documentation_context(self, keywords: List[str]) -> Dict[str, Any]:
        """
        Retrieve documentation context.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            Context information dictionary
        """
        results = self.retrieve_context(keywords, context_types=['docs'])
        
        items = results.get('docs', [])
        return {
            'relevant_documentation': [item for item in items],
            'total_docs': len(items),
            'search_keywords': keywords,
            'retrieval_method': 'json_database'
        }
    
    def retrieve_error_context(self, keywords: List[str]) -> Dict[str, Any]:
        """
        Retrieve error pattern context.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            Context information dictionary
        """
        results = self.retrieve_context(keywords, context_types=['errors'])
        
        items = results.get('errors', [])
        return {
            'relevant_errors': [item for item in items],
            'total_errors': len(items),
            'search_keywords': keywords,
            'retrieval_method': 'json_database'
        }
    
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
        
        for source_type, items in all_context.items():
            if items:
                section_title = {
                    'components': 'Codebase Components',
                    'docs': 'Documentation',
                    'errors': 'Known Error Patterns'
                }.get(source_type, source_type.capitalize())
                
                formatted_context += f"## {section_title}\n\n"
                
                for item in items[:5]:  # Limit to top 5 per type
                    item_id = item.get('id', item.get('title', 'Item'))
                    formatted_context += f"**{item_id}**\n"
                    if 'content' in item:
                        formatted_context += f"{item['content']}\n\n"
                    formatted_context += "\n"
        
        return formatted_context
    
    def add_context_source(self, name: str, source: ContextSource):
        """
        Not supported with JSON-based context retrieval.
        To add new context, update the JSON files directly.
        
        Args:
            name: Name of the source
            source: ContextSource implementation
        """
        raise NotImplementedError(
            "Adding context sources dynamically is not supported with JSON-based retrieval. "
            "Please update the JSON files directly in assets/context/."
        )
    
    def get_available_sources(self) -> List[str]:
        """Get list of available context sources."""
        return ['components', 'docs', 'errors']

