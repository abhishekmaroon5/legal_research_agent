"""Optimized citation chainer for legal research."""
import re
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

from .base_agent import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalCitation:
    """Structured legal citation."""
    citation: str
    case_name: str = ""
    year: Optional[int] = None
    court: str = ""
    url: str = ""
    relevance_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

class CitationChainer(BaseAgent[List[Dict]]):
    """Optimized citation chainer for legal research."""
    
    def __init__(self):
        """Initialize the citation chainer."""
        super().__init__()
        self.rate_limits = {
            'default': 2.0,  # 2 requests per second
            'google_scholar': 1.0,  # 1 request per second
            'caselaw': 2.0  # 2 requests per second
        }
        
    async def process(self, document_text: str, **kwargs) -> List[Dict]:
        """Process the document text and return related citations.
        
        Args:
            document_text: Text content of the document to analyze
            **kwargs: Additional arguments including:
                - document_url: URL of the document (optional)
                - max_citations: Maximum number of citations to return (default: 10)
                - min_relevance: Minimum relevance score (0-1, default: 0.5)
                
        Returns:
            List of citation dictionaries with metadata
        """
        document_url = kwargs.get('document_url', '')
        max_citations = kwargs.get('max_citations', 10)
        min_relevance = kwargs.get('min_relevance', 0.5)
        
        return await self.find_related_citations(
            document_text=document_text,
            document_url=document_url,
            max_citations=max_citations,
            min_relevance=min_relevance
        )
        
        # Common citation patterns
        self.citation_patterns = {
            'us_supreme': r'(\d+)\s+U\.?\s*S\.?\s+(\d+)',
            'federal': r'(\d+)\s+F\.?\s+(?:2d|3d|4th)?\s*\d+',
            'federal_supp': r'(\d+)\s+F\.?\s*Supp\.?\s*(?:2d|3d)?\s*\d+',
            'state': r'\d+\s+[A-Za-z]+\.?\s+\d+',
            'law_journal': r'\d+\s+[A-Za-z\.\s]+\d+',
            'statute': r'\d+\s+U\.?S\.?C\.?\s+§?\s*\d+',
            'cfr': r'\d+\s+C\.?F\.?R\.?\s+§?\s*\d+'
        }
        
        # Legal research APIs and endpoints
        self.legal_apis = {
            'google_scholar': 'https://scholar.google.com/scholar',
            'case_law': 'https://api.case.law/v1/cases/',
            'court_listener': 'https://www.courtlistener.com/api/rest/v3/',
            'findlaw': 'https://caselaw.findlaw.com/'
        }
        
        # Cache for citation lookups
        self.citation_cache: Dict[str, Dict] = {}
    
    async def find_related_citations(
        self,
        document_text: str = "",
        document_url: str = "",
        max_citations: int = 10,
        min_relevance: float = 0.5
    ) -> List[Dict]:
        """Find and analyze citations in a document.
        
        Args:
            document_text: Text content of the document
            document_url: URL of the document (for context)
            max_citations: Maximum number of citations to return
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            List of citation dictionaries with metadata
        """
        cache_key = self.cache_key(
            'find_related_citations', 
            document_text[:1000] if document_text else document_url,
            max_citations,
            min_relevance
        )
        
        if cached := await self.get_cached(cache_key):
            return cached
        
        try:
            # Extract citations from document text
            citations = await self.extract_citations(document_text)
            
            if not citations and document_url:
                # If no citations found in text, try to fetch document content
                doc_content = await self._fetch_document_content(document_url)
                if doc_content:
                    citations = await self.extract_citations(doc_content)
            
            # Get related cases for each citation
            related_cases = await self.get_related_cases(citations, max_citations)
            
            # Score and sort results
            scored_results = await self._score_citations(related_cases, document_text)
            
            # Filter by relevance and limit results
            filtered = [
                {
                    'citation': c.citation,
                    'case_name': c.case_name,
                    'year': c.year,
                    'court': c.court,
                    'url': c.url,
                    'relevance_score': c.relevance_score,
                    'metadata': c.metadata
                }
                for c in scored_results
                if c.relevance_score >= min_relevance
            ][:max_citations]
            
            # Cache the results
            await self.set_cached(cache_key, filtered, ttl=86400)  # Cache for 24 hours
            return filtered
            
        except Exception as e:
            logger.error(f"Error finding related citations: {e}")
            return []
    
    async def extract_citations(self, text: str) -> List[LegalCitation]:
        """Extract legal citations from text."""
        if not text:
            return []
            
        citations = set()
        
        # Check for common citation patterns
        for pattern_name, pattern in self.citation_patterns.items():
            for match in re.finditer(pattern, text):
                citation = match.group(0).strip()
                if len(citation) > 5:  # Filter out very short matches
                    citations.add(citation)
        
        # Convert to LegalCitation objects
        return [LegalCitation(citation=c) for c in citations]
    
    async def get_related_cases(
        self,
        citations: List[LegalCitation],
        max_results: int = 10
    ) -> List[LegalCitation]:
        """Get related cases for the given citations."""
        if not citations:
            return []
        
        # Process citations in parallel
        tasks = [self._lookup_citation(c) for c in citations[:20]]  # Limit to first 20
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and flatten results
        related_cases = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"Error looking up citation: {result}")
                continue
            if result and isinstance(result, list):
                related_cases.extend(result)
        
        # Deduplicate by case name or citation
        unique_cases = {}
        for case in related_cases:
            key = case.case_name or case.citation
            if key and key not in unique_cases:
                unique_cases[key] = case
        
        return list(unique_cases.values())[:max_results]
    
    async def _lookup_citation(self, citation: LegalCitation) -> List[LegalCitation]:
        """Look up a single citation and return related cases."""
        # Check cache first
        if citation.citation in self.citation_cache:
            return self.citation_cache[citation.citation]
        
        try:
            # Try Google Scholar first
            results = await self._search_google_scholar(citation.citation)
            
            if not results:
                # Fall back to other sources
                results = await self._search_other_sources(citation.citation)
            
            # Cache the results
            self.citation_cache[citation.citation] = results
            return results
            
        except Exception as e:
            logger.error(f"Error looking up citation {citation.citation}: {e}")
            return []
    
    async def _search_google_scholar(self, citation: str) -> List[LegalCitation]:
        """Search for a citation on Google Scholar."""
        results = []
        
        try:
            params = {
                'q': f'"{citation}"',
                'hl': 'en',
                'as_sdt': '2006',
                'as_vis': '1'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with self.rate_limiter('google_scholar'):
                async with self.session.get(
                    self.legal_apis['google_scholar'],
                    params=params,
                    headers=headers,
                    timeout=10
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Google Scholar search failed with status {response.status}")
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Parse search results
                    for result in soup.select('div.gs_ri'):
                        try:
                            title_elem = result.select_one('h3.gs_rt')
                            if not title_elem or not title_elem.a:
                                continue
                                
                            title = title_elem.get_text().strip()
                            url = title_elem.a['href']
                            snippet = result.select_one('div.gs_rs')
                            
                            # Extract citation info
                            citation_info = self._extract_citation_info(title, url, snippet.text if snippet else "")
                            
                            results.append(LegalCitation(
                                citation=citation_info.get('citation', ''),
                                case_name=citation_info.get('case_name', ''),
                                year=citation_info.get('year'),
                                court=citation_info.get('court', ''),
                                url=url,
                                metadata={
                                    'source': 'google_scholar',
                                    'snippet': snippet.text if snippet else ''
                                }
                            ))
                            
                        except Exception as e:
                            logger.debug(f"Error parsing Google Scholar result: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}")
        
        return results
    
    async def _search_other_sources(self, citation: str) -> List[LegalCitation]:
        """Search for citations in other legal databases."""
        # Implement other legal database searches here
        return []
    
    def _extract_citation_info(
        self,
        title: str,
        url: str,
        snippet: str
    ) -> Dict[str, Any]:
        """Extract citation information from search result."""
        # This is a simplified implementation
        # In a real application, you'd use a more sophisticated citation parser
        
        # Try to extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', title + " " + snippet)
        year = int(year_match.group(0)) if year_match else None
        
        # Try to extract court
        court = ""
        if 'U.S.' in title or 'Supreme Court' in snippet:
            court = "U.S. Supreme Court"
        elif 'F.2d' in title or 'F.3d' in title or 'F.4th' in title:
            court = "U.S. Court of Appeals"
        elif 'F. Supp' in title:
            court = "U.S. District Court"
        
        return {
            'citation': title.split('.')[0].strip(),
            'case_name': title,
            'year': year,
            'court': court,
            'url': url
        }
    
    async def _score_citations(
        self,
        citations: List[LegalCitation],
        context: str = ""
    ) -> List[LegalCitation]:
        """Score citations based on relevance to the context."""
        if not citations:
            return []
        
        # Simple scoring based on citation features
        for citation in citations:
            score = 0.0
            
            # Higher score for more recent cases
            if citation.year and citation.year >= 2000:
                score += 0.3
            elif citation.year and citation.year >= 1990:
                score += 0.2
            else:
                score += 0.1
            
            # Higher score for higher courts
            if 'supreme' in citation.court.lower():
                score += 0.4
            elif 'appeals' in citation.court.lower():
                score += 0.3
            elif 'district' in citation.court.lower():
                score += 0.2
            
            # Context relevance (simple keyword matching)
            if context:
                context_terms = set(context.lower().split())
                citation_terms = set(
                    (citation.case_name + " " + (citation.citation or "")).lower().split()
                )
                
                # Calculate Jaccard similarity
                intersection = len(context_terms & citation_terms)
                union = len(context_terms | citation_terms)
                
                if union > 0:
                    similarity = intersection / union
                    score += 0.3 * similarity
            
            # Ensure score is between 0 and 1
            citation.relevance_score = max(0.0, min(1.0, score))
        
        # Sort by score (descending)
        return sorted(citations, key=lambda x: x.relevance_score, reverse=True)
    
    async def _fetch_document_content(self, url: str) -> Optional[str]:
        """Fetch document content from a URL."""
        if not url:
            return None
            
        try:
            async with self.rate_limiter('default'):
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        
                        if 'application/pdf' in content_type:
                            # Handle PDF content
                            return await response.text(encoding='utf-8', errors='ignore')
                        else:
                            # Handle HTML content
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()
                                
                            return soup.get_text(separator='\n', strip=True)
                    
                    logger.warning(f"Failed to fetch document: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching document content: {e}")
            return None
    
    async def close(self):
        """Clean up resources."""
        await super().close()
