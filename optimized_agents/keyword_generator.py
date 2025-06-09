"""Optimized keyword generator with caching and rate limiting."""
from typing import Dict, List, Any, Optional
import asyncio
import json
import logging
from dataclasses import dataclass
from .base_agent import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Keyword:
    """Structured keyword with metadata."""
    keyword: str
    score: float = 1.0
    metadata: Dict[str, Any] = None

class KeywordGenerator(BaseAgent[List[Dict]]):
    """Optimized keyword generator with caching and rate limiting."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        """Initialize the keyword generator.
        
        Args:
            model_name: Name of the model to use
        """
        super().__init__(model_name=model_name)
    
    async def process(self, context: Dict[str, Any], **kwargs) -> List[Dict]:
        """Process the input context and return generated keywords.
        
        Args:
            context: Dictionary containing research context
            **kwargs: Additional arguments
            
        Returns:
            List of keyword dictionaries with scores
        """
        return await self.generate_keywords(context, **kwargs)
        
    async def generate_keywords(
        self,
        context: Dict[str, Any],
        max_keywords: int = 10,
        min_score: float = 0.5
    ) -> List[Dict]:
        """Generate relevant keywords based on the research context.
        
        Args:
            context: Research context containing base paper and research angle
            max_keywords: Maximum number of keywords to generate
            min_score: Minimum relevance score for keywords (0-1)
            
        Returns:
            List of keyword dictionaries with scores
        """
        cache_key = self.cache_key('generate_keywords', context, max_keywords, min_score)
        if cached := await self.get_cached(cache_key):
            return cached
        
        try:
            # Extract key information from context
            base_paper = context.get('base_paper', {})
            research_angle = context.get('new_angle', {})
            
            # Prepare prompt
            prompt = self._build_keyword_prompt(base_paper, research_angle, max_keywords)
            
            # Generate keywords using the model
            keywords = await self._generate_with_model(prompt, max_keywords)
            
            # Filter by minimum score
            filtered_keywords = [
                kw for kw in keywords 
                if kw.get('score', 0) >= min_score
            ][:max_keywords]
            
            # Cache the results
            await self.set_cached(cache_key, filtered_keywords, ttl=86400)  # Cache for 24 hours
            return filtered_keywords
            
        except Exception as e:
            logger.error(f"Error generating keywords: {e}")
            # Fallback to simple keyword extraction
            return self._fallback_keyword_extraction(context, max_keywords)
    
    def _build_keyword_prompt(
        self,
        base_paper: Dict[str, Any],
        research_angle: Dict[str, Any],
        max_keywords: int
    ) -> str:
        """Build the prompt for keyword generation."""
        return f"""
        Generate {max_keywords} relevant search keywords for legal research based on the following:
        
        Base Paper:
        - Thesis: {base_paper.get('thesis', 'N/A')}
        - Key Findings: {', '.join(base_paper.get('findings', []))}
        
        Research Angle:
        - Builds Upon: {', '.join(research_angle.get('builds_upon', []))}
        - Diverges: {', '.join(research_angle.get('diverges', []))}
        - Gaps Addressed: {', '.join(research_angle.get('gaps_addressed', []))}
        
        Return the response as a JSON array of objects with 'keyword' and 'score' (0-1) fields.
        Focus on legal concepts, case names, statutes, and specific legal terms.
        """
    
    async def _generate_with_model(self, prompt: str, max_keywords: int) -> List[Dict]:
        """Generate keywords using the model."""
        # In a real implementation, this would call the actual model
        # For now, we'll simulate it with a simple implementation
        await asyncio.sleep(0.1)  # Simulate API call
        
        # This is a simplified example - in reality, you'd parse the model's response
        return [
            {"keyword": "artificial intelligence", "score": 0.95},
            {"keyword": "intellectual property rights", "score": 0.92},
            {"keyword": "AI patent law", "score": 0.88},
            {"keyword": "machine learning copyright", "score": 0.85},
            {"keyword": "AI authorship", "score": 0.82}
        ][:max_keywords]
    
    def _fallback_keyword_extraction(
        self,
        context: Dict[str, Any],
        max_keywords: int
    ) -> List[Dict]:
        """Fallback method for keyword extraction if model fails."""
        # Extract text from context
        text = ""
        if 'base_paper' in context:
            paper = context['base_paper']
            text += f"{paper.get('thesis', '')} {' '.join(paper.get('findings', []))}"
        
        if 'new_angle' in context:
            angle = context['new_angle']
            text += f"{' '.join(angle.get('builds_upon', []))} "
            text += f"{' '.join(angle.get('diverges', []))} "
            text += f"{' '.join(angle.get('gaps_addressed', []))}"
        
        # Simple keyword extraction (in a real implementation, use a proper NLP library)
        words = text.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only consider words longer than 3 characters
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"keyword": word, "score": min(1.0, count/10)} 
                for word, count in sorted_words[:max_keywords]]
    
    async def close(self):
        """Clean up resources."""
        await super().close()
