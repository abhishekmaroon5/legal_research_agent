"""Optimized relevance scorer for legal research."""
import re
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from .base_agent import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

@dataclass
class ScoredItem:
    """Container for scored items with metadata."""
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[np.ndarray] = None

class RelevanceScorer(BaseAgent[Dict]):
    """Optimized relevance scorer for legal research."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        """Initialize the relevance scorer.
        
        Args:
            model_name: Name of the model to use for semantic scoring
        """
        super().__init__(model_name=model_name)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.nlp = nlp
        
    async def process(self, source: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process a source and return its relevance score.
        
        Args:
            source: Source document with title, content, etc.
            **kwargs: Additional arguments including:
                - research_angle: Research angle or question
                - base_arguments: Base arguments from the original paper
                - use_semantic: Whether to use semantic similarity (default: True)
                
        Returns:
            Source dict with added relevance_score and other metadata
        """
        research_angle = kwargs.get('research_angle', '')
        base_arguments = kwargs.get('base_arguments', {})
        use_semantic = kwargs.get('use_semantic', True)
        
        return await self.score_source(
            source=source,
            research_angle=research_angle,
            base_arguments=base_arguments,
            use_semantic=use_semantic
        )
        
    async def score_source(
        self,
        source: Dict[str, Any],
        research_angle: str,
        base_arguments: Dict[str, Any],
        use_semantic: bool = True
    ) -> Dict[str, Any]:
        """Score a source's relevance to the research angle and base arguments.
        
        Args:
            source: Source document with title, content, etc.
            research_angle: Research angle or question
            base_arguments: Base arguments from the original paper
            use_semantic: Whether to use semantic similarity (slower but more accurate)
            
        Returns:
            Source dict with added relevance_score and other metadata
        """
        cache_key = self.cache_key(
            'score_source',
            f"{source.get('title', '')[:100]}{source.get('url', '')}",
            research_angle[:200] if research_angle else '',
            json.dumps(base_arguments, sort_keys=True) if base_arguments else ''
        )
        
        if cached := await self.get_cached(cache_key):
            return {**source, **cached}
        
        try:
            # Extract text components for scoring
            title = source.get('title', '')
            snippet = source.get('snippet', '')
            content = source.get('content', '')
            
            # Combine all text for full analysis
            full_text = f"{title}. {snippet} {content}"
            
            # Initialize scores
            scores = {
                'keyword_overlap': 0.0,
                'semantic_similarity': 0.0,
                'source_quality': 0.7,  # Base quality score
                'freshness': 0.5,  # Default freshness score
                'relevance_score': 0.0  # Will be calculated as weighted sum
            }
            
            # 1. Keyword overlap score
            scores['keyword_overlap'] = self._calculate_keyword_overlap(
                full_text, research_angle, base_arguments
            )
            
            # 2. Semantic similarity (more computationally expensive)
            if use_semantic:
                scores['semantic_similarity'] = await self._calculate_semantic_similarity(
                    full_text, research_angle, base_arguments
                )
            
            # 3. Source quality score
            scores['source_quality'] = self._calculate_source_quality(source)
            
            # 4. Freshness score (if publication date is available)
            if 'date' in source:
                scores['freshness'] = self._calculate_freshness_score(source['date'])
            
            # Calculate weighted relevance score
            weights = {
                'keyword_overlap': 0.3,
                'semantic_similarity': 0.4 if use_semantic else 0.0,
                'source_quality': 0.2,
                'freshness': 0.1
            }
            
            # Normalize weights
            total_weight = sum(weights.values())
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            # Calculate final score
            scores['relevance_score'] = sum(
                scores[k] * normalized_weights[k] 
                for k in normalized_weights
            )
            
            # Ensure score is between 0 and 1
            scores['relevance_score'] = max(0.0, min(1.0, scores['relevance_score']))
            
            # Cache the results
            await self.set_cached(cache_key, scores, ttl=86400)  # Cache for 24 hours
            
            # Return source with added scores
            return {
                **source,
                **scores,
                'scoring_metadata': {
                    'scoring_method': 'hybrid',
                    'model_used': model_name,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error scoring source: {e}")
            return {
                **source,
                'relevance_score': 0.0,
                'error': str(e)
            }
    
    def _calculate_keyword_overlap(
        self,
        text: str,
        research_angle: str,
        base_arguments: Dict[str, Any]
    ) -> float:
        """Calculate keyword overlap score."""
        if not text or not (research_angle or base_arguments):
            return 0.0
        
        # Extract keywords from research angle and base arguments
        query_terms = set()
        
        # Add research angle terms
        if research_angle:
            doc = self.nlp(research_angle.lower())
            query_terms.update(
                token.lemma_ 
                for token in doc 
                if not token.is_stop and not token.is_punct and token.is_alpha
            )
        
        # Add base argument terms
        if base_arguments:
            if isinstance(base_arguments, dict):
                for value in base_arguments.values():
                    if isinstance(value, str):
                        doc = self.nlp(value.lower())
                        query_terms.update(
                            token.lemma_
                            for token in doc
                            if not token.is_stop and not token.is_punct and token.is_alpha
                        )
        
        if not query_terms:
            return 0.0
        
        # Process text
        doc = self.nlp(text.lower())
        text_terms = set(
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        )
        
        # Calculate Jaccard similarity
        intersection = len(query_terms & text_terms)
        union = len(query_terms | text_terms)
        
        return intersection / union if union > 0 else 0.0
    
    async def _calculate_semantic_similarity(
        self,
        text: str,
        research_angle: str,
        base_arguments: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity using embeddings."""
        if not text or not (research_angle or base_arguments):
            return 0.0
        
        try:
            # Prepare query text
            query_parts = []
            if research_angle:
                query_parts.append(research_angle)
            
            if base_arguments:
                if isinstance(base_arguments, dict):
                    query_parts.extend(str(v) for v in base_arguments.values() if v)
                elif isinstance(base_arguments, (list, tuple)):
                    query_parts.extend(str(item) for item in base_arguments if item)
            
            query_text = " ".join(query_parts)
            
            if not query_text.strip():
                return 0.0
            
            # Get embeddings
            query_embedding = await self._get_embedding(query_text)
            text_embedding = await self._get_embedding(text)
            
            if query_embedding is None or text_embedding is None:
                return 0.0
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                text_embedding.reshape(1, -1)
            )[0][0]
            
            # Ensure the score is between 0 and 1
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}")
            return 0.0
    
    def _calculate_source_quality(self, source: Dict[str, Any]) -> float:
        """Calculate source quality score based on metadata."""
        score = 0.7  # Base score
        
        # Domain-based scoring
        domain = source.get('domain', '').lower()
        if any(d in domain for d in ['.edu', '.gov', '.org']):
            score += 0.1
        
        # Known legal domains
        legal_domains = [
            'caselaw.findlaw.com',
            'supreme.justia.com',
            'law.cornell.edu',
            'scholar.google.com',
            'casetext.com',
            'courtlistener.com'
        ]
        
        if any(legal_domain in domain for legal_domain in legal_domains):
            score += 0.2
        
        # Length-based scoring (longer content is generally better)
        content_length = len(source.get('content', ''))
        if content_length > 5000:
            score += 0.1
        elif content_length > 2000:
            score += 0.05
            
        # Citation count (if available)
        if 'citation_count' in source and isinstance(source['citation_count'], (int, float)):
            if source['citation_count'] > 100:
                score += 0.1
            elif source['citation_count'] > 10:
                score += 0.05
        
        return min(1.0, score)
    
    def _calculate_freshness_score(self, date_str: str) -> float:
        """Calculate freshness score based on publication date."""
        try:
            from datetime import datetime
            
            # Parse date (handle different formats)
            date_formats = [
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%d/%m/%Y',
                '%Y',
                '%Y-%m',
                '%Y/%m'
            ]
            
            pub_date = None
            for fmt in date_formats:
                try:
                    pub_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if not pub_date:
                return 0.5  # Default score if date parsing fails
            
            # Calculate days since publication
            days_since_pub = (datetime.now() - pub_date).days
            
            # Calculate score (exponential decay)
            # Half-life of 5 years (1825 days)
            half_life = 1825
            score = 0.5 ** (days_since_pub / half_life)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.debug(f"Error calculating freshness score: {e}")
            return 0.5  # Default score on error
    
    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for the given text."""
        if not text.strip():
            return None
            
        try:
            # In a real implementation, this would call an embedding API
            # For now, we'll use a simple TF-IDF vectorizer as a placeholder
            vector = self.vectorizer.fit_transform([text])
            return vector.toarray()[0]
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    async def close(self):
        """Clean up resources."""
        await super().close()
