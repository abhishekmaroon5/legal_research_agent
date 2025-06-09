"""Optimized source crawler with caching and rate limiting."""
import aiohttp
import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .base_agent import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Structured search result."""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SourceCrawler(BaseAgent[List[Dict]]):
    """Optimized source crawler with caching and rate limiting."""
    
    def __init__(self):
        """Initialize the source crawler."""
        super().__init__()
        self.rate_limits = {
            'google': 1.0,  # 1 request per second
            'scholar': 1.0,  # 1 request per second
            'default': 2.0,  # 2 requests per second
        }
        self.search_engines = {
            'google': 'https://www.google.com/search',
            'bing': 'https://www.bing.com/search',
            'duckduckgo': 'https://html.duckduckgo.com/html/'
        }
        self.legal_domains = [
            'caselaw.findlaw.com',
            'supreme.justia.com',
            'law.cornell.edu',
            'scholar.google.com',
            'casetext.com',
            'courtlistener.com',
            'justia.com',
            'oyez.org',
            'scotusblog.com',
            'law.justia.com',
            'supremecourt.gov',
            'law.cornell.edu/supct',
            'scholar.google.com/scholar'
        ]
        
    async def process(self, query: str, **kwargs) -> List[Dict]:
        """Process the input query and return found sources.
        
        Args:
            query: Search query string
            **kwargs: Additional arguments including:
                - max_results: Maximum number of results to return (default: 5)
                - domains: List of domains to prioritize
                - min_relevance: Minimum relevance score (0-1)
                
        Returns:
            List of source dictionaries with metadata
        """
        max_results = kwargs.get('max_results', 5)
        domains = kwargs.get('domains')
        min_relevance = kwargs.get('min_relevance', 0.5)
        
        return await self.find_sources(
            query=query,
            max_results=max_results,
            domains=domains,
            min_relevance=min_relevance
        )
        self.legal_domains = [
            'scholar.google.com',
            'www.casemine.com',
            'www.justia.com',
            'www.law.cornell.edu',
            'supreme.justia.com',
            'www.oyez.org',
            'www.scotusblog.com',
            'www.law360.com',
            'www.lexisnexis.com',
            'westlaw.com'
        ]
    
    async def find_sources(
        self,
        query: str,
        max_results: int = 5,
        domains: Optional[List[str]] = None,
        min_relevance: float = 0.5
    ) -> List[Dict]:
        """Find relevant sources for a given query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            domains: List of domains to prioritize
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            List of source dictionaries
        """
        cache_key = self.cache_key('find_sources', query, max_results, domains, min_relevance)
        if cached := await self.get_cached(cache_key):
            return cached
        
        try:
            # Prepare search queries for different engines
            queries = self._prepare_queries(query, domains)
            
            # Execute searches in parallel
            search_tasks = []
            for engine, q in queries.items():
                search_tasks.append(
                    self._search_engine(engine, q, max_results)
                )
            
            # Gather results from all search engines
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and deduplicate results
            combined = []
            seen_urls = set()
            
            for engine_results in results:
                if isinstance(engine_results, Exception):
                    logger.warning(f"Search error: {engine_results}")
                    continue
                    
                for result in engine_results:
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        combined.append(result)
            
            # Score and sort results
            scored_results = await self._score_results(combined, query, domains)
            
            # Filter by minimum relevance and get top results
            filtered = [
                {
                    'title': r.title,
                    'url': r.url,
                    'snippet': r.snippet,
                    'source': r.source,
                    'relevance_score': r.metadata.get('relevance_score', 0),
                    'timestamp': r.timestamp
                }
                for r in scored_results
                if r.metadata.get('relevance_score', 0) >= min_relevance
            ][:max_results]
            
            # Cache the results
            await self.set_cached(cache_key, filtered, ttl=86400)  # Cache for 24 hours
            return filtered
            
        except Exception as e:
            logger.error(f"Error finding sources: {e}")
            return []
    
    def _prepare_queries(
        self,
        query: str,
        domains: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Prepare search queries for different engines."""
        queries = {}
        
        # Add site: filters for each domain
        site_filters = []
        if domains:
            site_filters.extend([f"site:{d}" for d in domains])
        
        # Add legal domain filters
        legal_filters = [f"site:{d}" for d in self.legal_domains]
        
        # Google search with site filters
        google_query = f"{query} {' '.join(site_filters)} {' '.join(legal_filters)} filetype:pdf OR filetype:docx"
        queries['google'] = google_query
        
        # Bing search (slightly different format)
        bing_query = f"{query} {' '.join(f'site:{d}' for d in (domains or []) + self.legal_domains)}"
        queries['bing'] = bing_query
        
        # DuckDuckGo search
        ddg_query = f"{query} {' '.join(f'site:{d}' for d in (domains or []) + self.legal_domains)}"
        queries['duckduckgo'] = ddg_query
        
        return queries
    
    async def _search_google(
        self,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Search using Google."""
        results = []
        
        try:
            params = {
                'q': query,
                'num': max_results,
                'hl': 'en',
                'as_sitesearch': '|'.join(self.legal_domains)
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with self.rate_limiter('google'):
                async with self.session.get(
                    'https://www.google.com/search',
                    params=params,
                    headers=headers,
                    timeout=10
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Google search failed with status {response.status}")
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Parse search results
                    for result in soup.select('div.g'):
                        try:
                            link = result.select_one('a[href^="/url?q="]')
                            if not link:
                                continue
                                
                            url = link['href'].split('&')[0].replace('/url?q=', '')
                            title = link.text.strip()
                            snippet = result.select_one('div.VwiC3b')
                            
                            results.append(SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet.text if snippet else '',
                                source='google',
                                metadata={
                                    'engine': 'google',
                                    'position': len(results) + 1
                                }
                            ))
                            
                            if len(results) >= max_results:
                                break
                                
                        except Exception as e:
                            logger.debug(f"Error parsing Google result: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error searching Google: {e}")
        
        return results
    
    async def _search_engine(
        self,
        engine: str,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Search using the specified engine."""
        if engine == 'google':
            return await self._search_google(query, max_results)
        # Add other search engines here
        
        return []
    
    async def _score_results(
        self,
        results: List[SearchResult],
        query: str,
        domains: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Score search results based on relevance."""
        if not results:
            return []
        
        # Score each result
        scored = []
        for i, result in enumerate(results, 1):
            score = 0.0
            
            # Position-based scoring (earlier results are better)
            position_score = 1.0 / (i ** 0.5)
            score += 0.3 * position_score
            
            # Domain-based scoring
            if domains and any(d in result.url for d in domains):
                score += 0.3
            
            # Legal domain boost
            if any(d in result.url for d in self.legal_domains):
                score += 0.2
            
            # Query term matching in title and snippet
            query_terms = set(query.lower().split())
            title_terms = set(result.title.lower().split())
            snippet_terms = set(result.snippet.lower().split())
            
            title_match = len(query_terms & title_terms) / len(query_terms) if query_terms else 0
            snippet_match = len(query_terms & snippet_terms) / len(query_terms) if query_terms else 0
            
            score += 0.2 * title_match + 0.1 * snippet_match
            
            # File type bonus
            if any(ext in result.url.lower() for ext in ['.pdf', '.docx', '.doc']):
                score += 0.1
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
            
            # Update result with score
            result.metadata['relevance_score'] = score
            scored.append(result)
        
        # Sort by score (descending)
        return sorted(scored, key=lambda x: x.metadata['relevance_score'], reverse=True)
    
    async def close(self):
        """Clean up resources."""
        await super().close()
