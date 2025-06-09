from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import aiohttp
import json
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urljoin, urlparse
import logging
from dataclasses import dataclass
from datetime import datetime

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Source:
    title: str
    url: str
    content: str
    citations: List[str]
    relevance_score: float
    date: Optional[datetime] = None

class CitationChainer:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Common citation patterns
        self.citation_patterns = [
            r'\b\d+\s+U\.S\.\s+\d+\b',  # Supreme Court
            r'\b\d+\s+F\.\s+\d+\b',     # Federal
            r'\b\d+\s+F\.2d\s+\d+\b',   # Federal 2nd
            r'\b\d+\s+F\.3d\s+\d+\b',   # Federal 3rd
            r'\b\d+\s+F\.4th\s+\d+\b',  # Federal 4th
            r'\b\d+\s+S\.\s+Ct\.\s+\d+\b',  # Supreme Court Reporter
            r'\b\d+\s+L\.\s+Ed\.\s+\d+\b',  # Lawyers' Edition
            r'\b\d+\s+L\.\s+Ed\.\s+2d\s+\d+\b',  # Lawyers' Edition 2nd
        ]
        
        # Legal research sites to search
        self.legal_sites = {
            'cornell': {
                'base_url': 'https://www.law.cornell.edu/search/site/',
                'result_selector': '.search-result',
                'title_selector': 'h3',
                'link_selector': 'a',
                'content_selector': '.search-snippet'
            },
            'justia': {
                'base_url': 'https://law.justia.com/search?q=',
                'result_selector': '.search-result',
                'title_selector': '.title',
                'link_selector': 'a',
                'content_selector': '.snippet'
            },
            'oyez': {
                'base_url': 'https://www.oyez.org/search?q=',
                'result_selector': '.search-result',
                'title_selector': '.case-name',
                'link_selector': 'a',
                'content_selector': '.case-description'
            }
        }
        
        # Headers to mimic browser requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    async def chain_citations(self, sources: List[Source], base_arguments: Dict[str, Any]) -> List[Source]:
        """Chain citations from a list of sources."""
        all_sources = []
        for s in sources:
            if isinstance(s, dict):
                all_sources.append(Source(
                    title=s['title'],
                    url=s['url'],
                    content=s['content'],
                    citations=self.extract_citations(s['content']),
                    relevance_score=s.get('relevance_score', 0.0),
                    date=datetime.fromtimestamp(s.get('timestamp', datetime.now().timestamp()))
                ))
            else:
                all_sources.append(s)
        processed_citations = set()
        for source in all_sources:
            for citation in source.citations:
                if citation not in processed_citations:
                    try:
                        related_sources = await self._follow_citation(citation, base_arguments)
                        all_sources.extend(related_sources)
                        processed_citations.add(citation)
                    except Exception as e:
                        logger.error(f"Error processing citation {citation}: {str(e)}")
                        continue
        print(f"All sources: {len(all_sources)}")
        print(f"All sources: {all_sources}")
        return all_sources

    def extract_citations(self, content: str) -> List[str]:
        """Extract citations from text content."""
        citations = []
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, content)
            citations.extend(match.group() for match in matches)
        return list(set(citations))  # Remove duplicates

    async def _follow_citation(self, citation: str, base_arguments: Dict[str, Any]) -> List[Source]:
        """Follow a citation to find related sources."""
        try:
            sources = await self._search_case_law(citation)
            relevant_sources = []
            
            for source in sources:
                try:
                    relevance = await self._assess_relevance(source, base_arguments)
                    if relevance > 0.5:  # Threshold for relevance
                        source.relevance_score = relevance
                        relevant_sources.append(source)
                except Exception as e:
                    logger.error(f"Error assessing relevance: {str(e)}")
                    continue
            
            return relevant_sources
        except Exception as e:
            logger.error(f"Error following citation {citation}: {str(e)}")
            return []

    async def _search_case_law(self, citation: str) -> List[Source]:
        """Search for case law using the citation."""
        try:
            results = await self._search_legal_sites(citation, "case law")
            sources = []
            
            for result in results:
                try:
                    content = await self._fetch_content(result['url'])
                    citations = self.extract_citations(content)
                    
                    source = Source(
                        title=result['title'],
                        url=result['url'],
                        content=content,
                        citations=citations,
                        relevance_score=0.0,
                        date=datetime.now()
                    )
                    sources.append(source)
                except Exception as e:
                    logger.error(f"Error processing search result: {str(e)}")
                    continue
            
            return sources
        except Exception as e:
            logger.error(f"Error searching case law: {str(e)}")
            return []

    async def _search_legal_sites(self, query: str, site_type: str) -> List[Dict[str, str]]:
        """Search multiple legal sites for the query."""
        results = []
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = []
            for site_name, site_config in self.legal_sites.items():
                search_url = f"{site_config['base_url']}{query}"
                tasks.append(self._search_site(session, search_url, site_config))
            
            site_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for site_result in site_results:
                if isinstance(site_result, Exception):
                    logger.error(f"Error searching site: {str(site_result)}")
                    continue
                results.extend(site_result)
        
        return results

    async def _search_site(self, session: aiohttp.ClientSession, url: str, site_config: Dict[str, str]) -> List[Dict[str, str]]:
        """Search a specific legal site."""
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Error fetching {url}: Status {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'lxml')
                results = []
                
                for result in soup.select(site_config['result_selector']):
                    try:
                        title_elem = result.select_one(site_config['title_selector'])
                        link_elem = result.select_one(site_config['link_selector'])
                        content_elem = result.select_one(site_config['content_selector'])
                        
                        if title_elem and link_elem:
                            results.append({
                                'title': title_elem.get_text(strip=True),
                                'url': link_elem.get('href', ''),
                                'content': content_elem.get_text(strip=True) if content_elem else ''
                            })
                    except Exception as e:
                        logger.error(f"Error parsing search result: {str(e)}")
                        continue
                
                return results
        except Exception as e:
            logger.error(f"Error searching site {url}: {str(e)}")
            return []

    async def _fetch_content(self, url: str) -> str:
        """Fetch and parse content from a URL."""
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching content from {url}: Status {response.status}")
                        return ""
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text(separator=' ', strip=True)
                    return text
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return ""

    async def _assess_relevance(self, source: Source, base_arguments: Dict[str, Any]) -> float:
        """Assess the relevance of a source to the base arguments."""
        try:
            # Simple keyword matching for now
            keywords = set(base_arguments.get('keywords', []))
            content_words = set(source.content.lower().split())
            
            if not keywords:
                return 0.5  # Default relevance if no keywords provided
            
            matches = len(keywords.intersection(content_words))
            return min(1.0, matches / len(keywords))
        except Exception as e:
            logger.error(f"Error assessing relevance: {str(e)}")
            return 0.0
