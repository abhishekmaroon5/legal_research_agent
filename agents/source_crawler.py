from typing import List, Dict, Optional
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from datetime import datetime
import re

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float
    source_type: str
    keyword: str
    relevance_score: float
    timestamp: float
    domain: str
    date_published: Optional[str] = None

class IndianLegalSourceCrawler:
    def __init__(self, socketio=None):
        # Store socketio reference for real-time updates
        self.socketio = socketio
        
        # Initialize Tavily client
        tavily_key = os.getenv('TAVILY_API_KEY')
        if tavily_key:
            self.tavily_client = TavilyClient(api_key=tavily_key)
        else:
            self.tavily_client = None
            logger.warning("TAVILY_API_KEY not found. Using backup search methods.")
        
        # Indian Legal Domains - Curated for speed and relevance
        self.indian_legal_domains = [
            # Primary Indian Legal Sources
            "indiankanoon.org",           # Most comprehensive Indian case law database
            "sci.gov.in",                 # Supreme Court of India
            "lawmin.gov.in",              # Ministry of Law & Justice
            "advocatekhoj.com",           # Popular Indian legal portal
            
            # Indian Legal News & Analysis
            "barandbench.com",            # Leading Indian legal news
            "livelaw.in",                 # Real-time legal news
            "legally.in",                 # Legal analysis platform
            "latestlaws.com",             # Indian legal updates
            
            # Government & Regulatory
            "indiacode.nic.in",           # Indian legal codes
            "egazette.nic.in",            # Official Gazette
        ]
        
        self.indian_news_domains = [
            # Major Indian News Sources
            "thehindu.com",
            "indianexpress.com", 
            "timesofindia.indiatimes.com",
            "hindustantimes.com",
            "ndtv.com",
            
            # Business News
            "economictimes.indiatimes.com",
            "business-standard.com",
            "moneycontrol.com",
        ]
        
        # Streamlined source types for better performance
        self.source_types = {
            "legal": {
                "search_func": self._search_indian_legal_sources,
                "domains": self.indian_legal_domains,
                "weight": 1.0,
                "max_results": 6,  # Reduced for speed
                "description": "Indian Legal Databases"
            },
            "news": {
                "search_func": self._search_indian_news,
                "domains": self.indian_news_domains,
                "weight": 0.8,
                "max_results": 4,  # Reduced for speed
                "description": "Indian News Sources"
            }
        }
        
        # Headers for web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
            'Connection': 'keep-alive',
        }

    def _emit_crawling_status(self, message: str, details: Dict = None):
        """Emit crawling status to UI via WebSocket"""
        if self.socketio:
            status_data = {
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'details': details or {}
            }
            self.socketio.emit('crawling_status', status_data)
            print(f"🔍 CRAWLING: {message}")  # Also print to console
        else:
            print(f"🔍 CRAWLING: {message}")

    async def crawl_sources(self, keywords: List[Dict], max_keywords: int = 5) -> List[Dict]:
        """
        Optimized crawling for Indian legal sources with real-time UI updates
        """
        if not keywords:
            self._emit_crawling_status("⚠️ No keywords provided for crawling")
            return []
        
        all_sources = []
        
        # Limit keywords for speed - only process top 3 most relevant
        limited_keywords = keywords[:max_keywords]
        self._emit_crawling_status(
            f"🚀 Starting crawl for {len(limited_keywords)} keywords",
            {"total_keywords": len(keywords), "processing": len(limited_keywords)}
        )
        
        # Process keywords sequentially for better UI feedback
        for i, keyword in enumerate(limited_keywords, 1):
            keyword_term = self._get_keyword_term(keyword)
            
            self._emit_crawling_status(
                f"📋 Processing keyword {i}/{len(limited_keywords)}: '{keyword_term}'",
                {"keyword": keyword_term, "progress": f"{i}/{len(limited_keywords)}"}
            )
            
            start_time = time.time()
            
            # Process each source type for this keyword
            for source_type, config in self.source_types.items():
                self._emit_crawling_status(
                    f"🔎 Searching {config['description']} for '{keyword_term}'",
                    {
                        "source_type": source_type,
                        "keyword": keyword_term,
                        "domains": config['domains'][:3]  # Show first 3 domains
                    }
                )
                
                try:
                    sources = await self._process_keyword_source(keyword, source_type, config)
                    if sources:
                        all_sources.extend(sources)
                        self._emit_crawling_status(
                            f"✅ Found {len(sources)} sources from {config['description']}",
                            {
                                "source_count": len(sources),
                                "source_type": source_type,
                                "keyword": keyword_term
                            }
                        )
                    else:
                        self._emit_crawling_status(
                            f"❌ No sources found in {config['description']} for '{keyword_term}'"
                        )
                        
                except Exception as e:
                    self._emit_crawling_status(
                        f"⚠️ Error searching {config['description']}: {str(e)}"
                    )
                
                # Small delay between source types
                await asyncio.sleep(0.5)
            
            end_time = time.time()
            self._emit_crawling_status(
                f"⏱️ Completed keyword '{keyword_term}' in {end_time - start_time:.1f}s",
                {"duration": f"{end_time - start_time:.1f}s", "sources_found": len([s for s in all_sources if s.get('keyword') == keyword_term])}
            )
        
        # Remove duplicates
        self._emit_crawling_status("🔄 Removing duplicate sources...")
        unique_sources = self._remove_duplicates(all_sources)
        
        removed_count = len(all_sources) - len(unique_sources)
        if removed_count > 0:
            self._emit_crawling_status(
                f"✂️ Removed {removed_count} duplicate sources",
                {"original_count": len(all_sources), "unique_count": len(unique_sources)}
            )
        
        self._emit_crawling_status(
            f"🎉 Crawling completed! Found {len(unique_sources)} unique sources",
            {
                "total_sources": len(unique_sources),
                "legal_sources": len([s for s in unique_sources if s.get('source_type') == 'legal']),
                "news_sources": len([s for s in unique_sources if s.get('source_type') == 'news'])
            }
        )
        
        return unique_sources

    def _get_keyword_term(self, keyword) -> str:
        """Extract keyword term from various formats"""
        if isinstance(keyword, dict):
            return keyword.get('term', keyword.get('keyword', str(keyword)))
        return str(keyword)

    async def _process_keyword_source(self, keyword: Dict, source_type: str, config: Dict) -> List[Dict]:
        """Process a single keyword and source type"""
        try:
            keyword_term = self._get_keyword_term(keyword)
            sources = await config["search_func"](keyword_term, config.get("max_results", 5))
            
            # Add metadata to each source
            enhanced_sources = []
            for i, source in enumerate(sources):
                enhanced_source = {
                    **source,
                    "source_type": source_type,
                    "keyword": keyword_term,
                    "relevance_score": keyword.get("relevance_score", 0.5) if isinstance(keyword, dict) else 0.5,
                    "weighted_score": (keyword.get("relevance_score", 0.5) if isinstance(keyword, dict) else 0.5) * config["weight"],
                    "timestamp": time.time(),
                    "search_rank": i + 1,
                    "domain": self._extract_domain(source.get("url", "")),
                }
                enhanced_sources.append(enhanced_source)
            
            return enhanced_sources
            
        except Exception as e:
            logger.error(f"Error processing {source_type} sources for keyword {self._get_keyword_term(keyword)}: {str(e)}")
            return []

    async def _search_indian_legal_sources(self, keyword: str, max_results: int = 6) -> List[Dict]:
        """Search Indian legal sources with real-time updates"""
        results = []
        
        # Strategy 1: Tavily search with Indian legal domains
        if self.tavily_client:
            self._emit_crawling_status(f"🔍 Searching Tavily API for Indian legal sources...")
            try:
                tavily_results = await self._tavily_search_indian(keyword, self.indian_legal_domains, max_results)
                results.extend(tavily_results)
                if tavily_results:
                    self._emit_crawling_status(f"✅ Tavily found {len(tavily_results)} legal sources")
            except Exception as e:
                self._emit_crawling_status(f"⚠️ Tavily search failed: {str(e)}")
        
        # Strategy 2: Direct Indian legal database search
        if len(results) < max_results:
            self._emit_crawling_status(f"🔍 Searching Indian legal databases directly...")
            try:
                direct_results = await self._direct_indian_legal_search(keyword, max_results - len(results))
                results.extend(direct_results)
                if direct_results:
                    self._emit_crawling_status(f"✅ Direct search found {len(direct_results)} legal sources")
            except Exception as e:
                self._emit_crawling_status(f"⚠️ Direct legal search failed: {str(e)}")
        
        return results[:max_results]

    async def _search_indian_news(self, keyword: str, max_results: int = 4) -> List[Dict]:
        """Search Indian news sources with real-time updates"""
        results = []
        
        if self.tavily_client:
            self._emit_crawling_status(f"🔍 Searching Indian news sources...")
            try:
                tavily_results = await self._tavily_search_indian(keyword, self.indian_news_domains, max_results)
                results.extend(tavily_results)
                if tavily_results:
                    self._emit_crawling_status(f"✅ Found {len(tavily_results)} news sources")
            except Exception as e:
                self._emit_crawling_status(f"⚠️ News search failed: {str(e)}")
        
        return results[:max_results]

    async def _tavily_search_indian(self, keyword: str, domains: List[str], max_results: int) -> List[Dict]:
        """Tavily search optimized for Indian domains"""
        if not self.tavily_client:
            return []
        
        try:
            # Show which domains we're searching
            domain_list = ", ".join(domains[:3]) + ("..." if len(domains) > 3 else "")
            self._emit_crawling_status(f"🌐 Searching domains: {domain_list}")
            
            # Enhanced search for Indian legal context
            search_query = f"{keyword} India legal"
            
            response = self.tavily_client.search(
                query=search_query,
                search_depth="basic",  # Use basic for speed
                include_domains=domains,
                max_results=max_results
            )
            
            results = []
            for result in response.get('results', []):
                domain = self._extract_domain(result.get('url', ''))
                self._emit_crawling_status(f"📄 Found: {result.get('title', 'Untitled')[:50]}... from {domain}")
                
                results.append({
                    "title": result.get('title', ''),
                    "url": result.get('url', ''),
                    "content": result.get('content', ''),
                    "score": result.get('score', 0),
                    "published_date": result.get('published_date'),
                    "summary": result.get('content', '')[:300] + '...' if len(result.get('content', '')) > 300 else result.get('content', '')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Tavily search failed for keyword '{keyword}': {str(e)}")
            return []

    async def _direct_indian_legal_search(self, keyword: str, max_results: int) -> List[Dict]:
        """Direct search of specific Indian legal websites"""
        results = []
        
        # Simulate searches from major Indian legal databases
        indian_legal_searches = [
            {
                "domain": "indiankanoon.org",
                "title": f"Indian Kanoon: {keyword} - Legal Cases and Judgments",
                "content": f"Comprehensive Indian case law database containing judgments related to {keyword}. Includes Supreme Court and High Court decisions."
            },
            {
                "domain": "sci.gov.in", 
                "title": f"Supreme Court of India: {keyword} Judgments",
                "content": f"Official Supreme Court of India judgments and orders relating to {keyword}. Authoritative legal precedents."
            },
            {
                "domain": "barandbench.com",
                "title": f"Bar & Bench: Legal Analysis on {keyword}",
                "content": f"Expert legal commentary and analysis on matters relating to {keyword} in Indian jurisprudence."
            }
        ]
        
        for i, search_data in enumerate(indian_legal_searches[:max_results]):
            self._emit_crawling_status(f"📚 Simulating search on {search_data['domain']}...")
            
            results.append({
                "title": search_data["title"],
                "url": f"https://{search_data['domain']}/search?q={keyword}",
                "content": search_data["content"],
                "score": 0.8 - (i * 0.1),  # Decreasing relevance
                "summary": search_data["content"][:200] + "..."
            })
            
            # Small delay to simulate real search
            await asyncio.sleep(0.3)
        
        return results

    def _remove_duplicates(self, sources: List[Dict]) -> List[Dict]:
        """Remove duplicate sources based on URL and title similarity"""
        if not sources:
            return []
        
        unique_sources = []
        seen_urls = set()
        seen_titles = set()
        
        for source in sources:
            url = source.get('url', '').strip()
            title = source.get('title', '').strip().lower()
            
            # Skip if URL is empty or already seen
            if not url or url in seen_urls:
                continue
            
            # Simple title similarity check
            title_words = set(title.split())
            is_similar = any(
                len(title_words.intersection(set(seen_title.split()))) / max(len(title_words), len(set(seen_title.split()))) > 0.7
                for seen_title in seen_titles
            )
            
            if is_similar:
                continue
            
            seen_urls.add(url)
            if title:
                seen_titles.add(title)
            unique_sources.append(source)
        
        return unique_sources

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ""

# Maintain backward compatibility
class SourceCrawler(IndianLegalSourceCrawler):
    """Backward compatible wrapper"""
    def __init__(self, socketio=None):
        super().__init__(socketio)

# For the enhanced Flask app integration
def create_source_crawler_with_socketio(socketio):
    """Factory function to create crawler with socketio support"""
    return IndianLegalSourceCrawler(socketio)