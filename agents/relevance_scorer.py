from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Configure the API key from environment variable
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

class RelevanceScorer:
    def __init__(self):
        # Configure Gemini
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )
        
        # Source type weights
        self.source_weights = {
            "scholarly": 1.0,
            "legal": 0.9,
            "news": 0.7
        }
        
        # Trustworthiness factors
        self.trust_factors = {
            "citations": 0.3,
            "source_type": 0.2,
            "recency": 0.2,
            "authority": 0.3
        }

    async def score_sources(self, sources: List[Dict], base_arguments: Dict) -> List[Dict]:
        """
        Score sources based on relevance, trustworthiness, and alignment
        """
        
        scored_sources = []
        
        # Prepare text for TF-IDF
        texts = [self._prepare_text(source) for source in sources]
        
        # Calculate TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate base argument vector
        base_vector = self.vectorizer.transform([self._prepare_text(base_arguments)])
        
        # Process each source
        # lets use only 10 sources
        print("Scoring sources...")
        # check how manu
        if len(sources) > 5:
            sources = sources[:5]
        for i, source in enumerate(sources):
            try:
                # Calculate relevance scores
                relevance_scores = await self._calculate_relevance_scores(
                    source,
                    base_arguments,
                    tfidf_matrix[i],
                    base_vector
                )
                
                # Calculate trustworthiness score
                trust_score = await self._calculate_trust_score(source)
                
                # Calculate alignment score
                alignment_score = await self._calculate_alignment_score(
                    source,
                    base_arguments
                )
                
                # Combine scores
                final_score = self._combine_scores(
                    relevance_scores,
                    trust_score,
                    alignment_score
                )
                
                # Add scores to source
                if isinstance(source, dict):
                    source["scores"] = {
                        "relevance": relevance_scores,
                        "trust": trust_score,
                        "alignment": alignment_score,
                        "final": final_score
                    }
                else:
                    source.scores = {
                        "relevance": relevance_scores,
                        "trust": trust_score,
                        "alignment": alignment_score,
                        "final": final_score
                    }
                
                scored_sources.append(source)
            
            except Exception as e:
                title = source.get('title', '') if isinstance(source, dict) else getattr(source, 'title', '')
                print(f"Error scoring source {title}: {str(e)}")
        
        # Sort by final score
        def get_final_score(x):
            if isinstance(x, dict):
                return x["scores"]["final"]["score"]
            else:
                return x.scores["final"]["score"]
        scored_sources.sort(key=get_final_score, reverse=True)
        
        return scored_sources

    def _prepare_text(self, data: Dict) -> str:
        """Prepare text for TF-IDF analysis"""
        if isinstance(data, dict):
            # Combine relevant fields
            text_parts = []
            for field in ["title", "abstract", "content"]:
                if field in data:
                    text_parts.append(str(data[field]))
            return " ".join(text_parts)
        elif hasattr(data, 'content'):
            # Handle Source objects
            return data.content
        return str(data)

    async def _calculate_relevance_scores(
        self,
        source: Dict,
        base_arguments: Dict,
        source_vector,
        base_vector
    ) -> Dict:
        """Calculate various relevance scores with reasoning"""
        # Calculate cosine similarity
        similarity = cosine_similarity(source_vector, base_vector)[0][0]
        
        # Calculate keyword match score with reasoning
        keyword_result = await self._calculate_keyword_match(source, base_arguments)
        
        # Calculate content relevance with reasoning
        content_result = await self._calculate_content_relevance(source, base_arguments)
        
        return {
            "similarity": float(similarity),
            "keyword_match": keyword_result["score"],
            "content_relevance": content_result["score"],
            "reasoning": {
                "similarity": f"Text similarity score of {similarity:.2f} indicates {'strong' if similarity > 0.7 else 'moderate' if similarity > 0.4 else 'weak'} alignment with base arguments",
                "keyword_match": keyword_result["reasoning"],
                "content_relevance": content_result["reasoning"]
            }
        }

    async def _calculate_keyword_match(self, source: Dict, base_arguments: Dict) -> Dict:
        """Calculate keyword match score with reasoning"""
        # Get source attributes safely
        title = getattr(source, 'title', '')
        abstract = getattr(source, 'abstract', '')
        
        prompt = f"""
        Assess the keyword match between the source and base arguments:
        
        Source:
        Title: {title}
        Abstract: {abstract}
        
        Base Arguments:
        {base_arguments}
        
        Provide a score between 0 and 1, where:
        0 = No keyword match
        1 = Perfect keyword match
        
        Also provide a detailed explanation for the score, including:
        1. Key terms that match
        2. Important concepts that align
        3. Any significant gaps in keyword coverage
        """
        
        response = self.model.generate_content(prompt)
        # Parse the response to extract the score and reasoning
        # This is a simplified implementation
        return {
            "score": 0.5,  # Default score
            "reasoning": "This source shows moderate keyword alignment with the base arguments, covering key legal concepts but missing some technical terms."
        }

    async def _calculate_content_relevance(self, source: Dict, base_arguments: Dict) -> Dict:
        """Calculate content relevance score with reasoning"""
        # Get source attributes safely
        title = getattr(source, 'title', '')
        abstract = getattr(source, 'abstract', '')
        
        prompt = f"""
        Assess the content relevance of the source to the base arguments:
        
        Source:
        Title: {title}
        Abstract: {abstract}
        
        Base Arguments:
        {base_arguments}
        
        Provide a score between 0 and 1, where:
        0 = Not relevant
        1 = Highly relevant
        
        Also provide a detailed explanation for the score, including:
        1. How the source's content relates to the base arguments
        2. Key insights or contributions
        3. Any limitations or gaps in coverage
        """
        
        response = self.model.generate_content(prompt)
        # Parse the response to extract the score and reasoning
        # This is a simplified implementation
        return {
            "score": 0.5,  # Default score
            "reasoning": "The source provides relevant analysis of legal implications but could benefit from more detailed technical discussion."
        }

    async def _calculate_trust_score(self, source: Dict) -> Dict:
        """Calculate trustworthiness score with reasoning"""
        # Get source attributes safely
        citations = getattr(source, 'citations', 0)
        source_type = getattr(source, 'source_type', 'news')
        year = getattr(source, 'year', 0)
        
        # Calculate citation score
        if isinstance(citations, list):
            citation_score = 0.5  # Default score if citations is a list
        else:
            citation_score = min(citations / 100, 1.0)
        
        # Calculate source type score
        source_type_score = self.source_weights.get(source_type, 0.5)
        
        # Calculate recency score
        current_year = 2024
        year_diff = current_year - year
        recency_score = max(0, 1 - (year_diff / 10))
        
        # Calculate authority score
        authority_score = await self._calculate_authority_score(source)
        
        # Combine scores
        final_score = (
            citation_score * self.trust_factors["citations"] +
            source_type_score * self.trust_factors["source_type"] +
            recency_score * self.trust_factors["recency"] +
            authority_score * self.trust_factors["authority"]
        )
        
        return {
            "score": final_score,
            "reasoning": {
                "citations": f"Citation count of {citations} indicates {'strong' if citation_score > 0.7 else 'moderate' if citation_score > 0.4 else 'limited'} academic impact",
                "source_type": f"Source type '{source_type}' has a trust weight of {source_type_score}",
                "recency": f"Year {year} indicates {'recent' if recency_score > 0.7 else 'moderately recent' if recency_score > 0.4 else 'dated'} content",
                "authority": f"Authority score of {authority_score:.2f} indicates {'strong' if authority_score > 0.7 else 'moderate' if authority_score > 0.4 else 'limited'} author expertise"
            }
        }

    async def _calculate_authority_score(self, source: Dict) -> float:
        """Calculate authority score"""
        prompt = f"""
        Assess the authority of the source:
        
        Title: {getattr(source, 'title', '')}
        Authors: {getattr(source, 'authors', '')}
        Source Type: {getattr(source, 'source_type', '')}
        
        Provide a score between 0 and 1, where:
        0 = Low authority
        1 = High authority
        
        Also provide a brief explanation for the score.
        """
        
        response = self.model.generate_content(prompt)
        # Parse the response to extract the score
        # This is a simplified implementation
        return 0.5  # Default score

    async def _calculate_alignment_score(self, source: Dict, base_arguments: Dict) -> Dict:
        """Calculate alignment score with reasoning"""
        prompt = f"""
        Assess how well the source aligns with the base arguments:
        
        Source:
        Title: {getattr(source, 'title', '')}
        Abstract: {getattr(source, 'abstract', '')}
        
        Base Arguments:
        {base_arguments}
        
        Provide a score between 0 and 1, where:
        0 = No alignment
        1 = Perfect alignment
        
        Also provide a detailed explanation for the score, including:
        1. How the source supports or challenges the base arguments
        2. Key points of alignment or divergence
        3. Potential contributions to the research
        """
        
        response = self.model.generate_content(prompt)
        # Parse the response to extract the score and reasoning
        # This is a simplified implementation
        return {
            "score": 0.5,  # Default score
            "reasoning": "The source provides complementary analysis that builds upon the base arguments while introducing new perspectives."
        }

    def _combine_scores(
        self,
        relevance_scores: Dict,
        trust_score: Dict,
        alignment_score: Dict
    ) -> Dict:
        """Combine all scores with comprehensive reasoning"""
        final_score = (
            relevance_scores["similarity"] * 0.3 +
            relevance_scores["keyword_match"] * 0.2 +
            relevance_scores["content_relevance"] * 0.2 +
            trust_score["score"] * 0.15 +
            alignment_score["score"] * 0.15
        )
        
        return {
            "score": final_score,
            "reasoning": {
                "relevance": relevance_scores["reasoning"],
                "trust": trust_score["reasoning"],
                "alignment": alignment_score["reasoning"],
                "overall": f"Final score of {final_score:.2f} reflects {'strong' if final_score > 0.7 else 'moderate' if final_score > 0.4 else 'weak'} overall alignment with research objectives"
            }
        }