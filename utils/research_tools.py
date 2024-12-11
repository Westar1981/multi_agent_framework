"""
Research tools for online knowledge acquisition.
"""

import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ResearchTool(ABC):
    """Base class for research tools."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limits = {
            'requests_per_minute': 60,
            'concurrent_requests': 5
        }
        self.request_semaphore = asyncio.Semaphore(self.rate_limits['concurrent_requests'])
        
    async def setup(self):
        """Setup the research tool."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
            
    @abstractmethod
    async def search(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for information."""
        pass
        
    async def _rate_limited_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a rate-limited request."""
        async with self.request_semaphore:
            try:
                if not self.session:
                    await self.setup()
                    
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()
                    
            except Exception as e:
                logger.error(f"Error making request to {url}: {str(e)}")
                return {}

class CodeRepository(ResearchTool):
    """Interface with code repositories."""
    
    def __init__(self):
        super().__init__()
        self.api_endpoints = {
            'github': 'https://api.github.com/search/code',
            'gitlab': 'https://gitlab.com/api/v4/search',
            'bitbucket': 'https://api.bitbucket.org/2.0/search/code'
        }
        self.cache_path = Path(__file__).parent / 'cache' / 'code_search_cache.json'
        self.cache: Dict[str, Any] = self._load_cache()
        
    def _load_cache(self) -> Dict[str, Any]:
        """Load search cache."""
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}
        
    def _save_cache(self):
        """Save search cache."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f)
            
    async def search(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search code repositories."""
        results = []
        cache_key = f"{query}_{json.dumps(context, sort_keys=True)}"
        
        # Check cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if (datetime.now() - datetime.fromisoformat(cache_entry['timestamp'])).days < 7:
                return cache_entry['results']
                
        # Search each repository
        for repo, endpoint in self.api_endpoints.items():
            try:
                repo_results = await self._search_repository(
                    repo, endpoint, query, context
                )
                results.extend(repo_results)
            except Exception as e:
                logger.error(f"Error searching {repo}: {str(e)}")
                
        # Update cache
        self.cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        self._save_cache()
        
        return results
        
    async def _search_repository(self, repo: str, endpoint: str, 
                               query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search a specific code repository."""
        params = self._build_search_params(repo, query, context)
        response = await self._rate_limited_request(endpoint, params)
        
        return self._parse_response(repo, response)
        
    def _build_search_params(self, repo: str, query: str, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Build search parameters for repository."""
        params = {'q': query}
        
        if repo == 'github':
            # Add GitHub-specific parameters
            params.update({
                'per_page': 100,
                'sort': 'stars',
                'order': 'desc'
            })
            
        elif repo == 'gitlab':
            # Add GitLab-specific parameters
            params.update({
                'scope': 'blobs',
                'per_page': 100
            })
            
        return params
        
    def _parse_response(self, repo: str, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse repository response."""
        results = []
        
        if repo == 'github' and 'items' in response:
            for item in response['items']:
                results.append({
                    'repository': repo,
                    'url': item.get('html_url'),
                    'path': item.get('path'),
                    'score': item.get('score', 0),
                    'metadata': {
                        'repository_url': item.get('repository', {}).get('html_url'),
                        'repository_stars': item.get('repository', {}).get('stargazers_count', 0)
                    }
                })
                
        elif repo == 'gitlab':
            # Parse GitLab response
            pass
            
        return results

class AcademicDatabase(ResearchTool):
    """Interface with academic databases."""
    
    def __init__(self):
        super().__init__()
        self.api_endpoints = {
            'arxiv': 'http://export.arxiv.org/api/query',
            'semanticscholar': 'https://api.semanticscholar.org/v1/paper/search'
        }
        self.cache_path = Path(__file__).parent / 'cache' / 'academic_search_cache.json'
        self.cache: Dict[str, Any] = self._load_cache()
        
    def _load_cache(self) -> Dict[str, Any]:
        """Load search cache."""
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}
        
    def _save_cache(self):
        """Save search cache."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f)
            
    async def search(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search academic databases."""
        results = []
        cache_key = f"{query}_{json.dumps(context, sort_keys=True)}"
        
        # Check cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if (datetime.now() - datetime.fromisoformat(cache_entry['timestamp'])).days < 7:
                return cache_entry['results']
                
        # Search each database
        for db, endpoint in self.api_endpoints.items():
            try:
                db_results = await self._search_database(
                    db, endpoint, query, context
                )
                results.extend(db_results)
            except Exception as e:
                logger.error(f"Error searching {db}: {str(e)}")
                
        # Update cache
        self.cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        self._save_cache()
        
        return results
        
    async def _search_database(self, db: str, endpoint: str,
                             query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search a specific academic database."""
        params = self._build_search_params(db, query, context)
        response = await self._rate_limited_request(endpoint, params)
        
        return self._parse_response(db, response)
        
    def _build_search_params(self, db: str, query: str,
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Build search parameters for database."""
        params = {'q': query}
        
        if db == 'arxiv':
            params.update({
                'max_results': 100,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            })
            
        elif db == 'semanticscholar':
            params.update({
                'limit': 100,
                'fields': 'title,abstract,year,citations'
            })
            
        return params
        
    def _parse_response(self, db: str, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse database response."""
        results = []
        
        if db == 'arxiv':
            # Parse arXiv response
            if 'entries' in response:
                for entry in response['entries']:
                    results.append({
                        'database': db,
                        'title': entry.get('title'),
                        'abstract': entry.get('summary'),
                        'url': entry.get('id'),
                        'published': entry.get('published'),
                        'authors': [author.get('name') for author in entry.get('authors', [])],
                        'categories': entry.get('categories', [])
                    })
                    
        elif db == 'semanticscholar':
            # Parse Semantic Scholar response
            if 'papers' in response:
                for paper in response['papers']:
                    results.append({
                        'database': db,
                        'title': paper.get('title'),
                        'abstract': paper.get('abstract'),
                        'url': paper.get('url'),
                        'year': paper.get('year'),
                        'citation_count': paper.get('citations', {}).get('count', 0)
                    })
                    
        return results
``` 
</rewritten_file>