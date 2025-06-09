from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
import os
import itertools
from collections import Counter

load_dotenv()

class KeywordGenerator:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Common legal terms and connectors
        self.legal_terms = [
            "jurisdiction", "precedent", "statute", "regulation",
            "constitution", "amendment", "legislation", "case law",
            "legal doctrine", "legal principle", "legal theory"
        ]
        
        self.connectors = [
            "and", "or", "versus", "versus", "v.", "in re",
            "ex parte", "et al.", "et seq."
        ]

    async def generate_keywords(self, base_arguments: Dict, seed_keywords: List[str] = None) -> List[Dict]:
        """
        Generate and rank search keyword permutations based on base arguments and seed keywords
        """
        # 1. Extract key terms from base arguments
        base_terms = await self._extract_key_terms(base_arguments)
        
        # 2. Combine with seed keywords
        all_terms = self._combine_terms(base_terms, seed_keywords)
        
        # 3. Generate permutations
        permutations = self._generate_permutations(all_terms)
        
        # 4. Rank permutations
        ranked_keywords = await self._rank_keywords(permutations, base_arguments)
        
        return ranked_keywords

    async def _extract_key_terms(self, base_arguments: Dict) -> List[str]:
        """Extract key terms from base arguments using Gemini"""
        prompt = f"""
        Extract key legal terms and concepts from the following arguments:
        
        {base_arguments}
        
        Focus on:
        1. Legal concepts
        2. Key cases
        3. Important statutes
        4. Legal doctrines
        5. Jurisdictional terms
        
        Provide a list of terms, one per line.
        """
        
        response = self.model.generate_content(prompt)
        terms = [term.strip() for term in response.text.split('\n') if term.strip()]
        return terms

    def _combine_terms(self, base_terms: List[str], seed_keywords: List[str] = None) -> List[str]:
        """Combine base terms with seed keywords and legal terms"""
        all_terms = set(base_terms)
        
        if seed_keywords:
            all_terms.update(seed_keywords)
        
        # Add relevant legal terms
        all_terms.update(self.legal_terms)
        
        return list(all_terms)

    def _generate_permutations(self, terms: List[str]) -> List[str]:
        """Generate keyword permutations"""
        permutations = []
        
        # Single terms
        permutations.extend(terms)
        
        # Two-term combinations
        for term1, term2 in itertools.combinations(terms, 2):
            permutations.append(f"{term1} {term2}")
        
        # Three-term combinations (limited to most relevant terms)
        top_terms = terms[:10]  # Limit to top 10 terms for three-word combinations
        for term1, term2, term3 in itertools.combinations(top_terms, 3):
            permutations.append(f"{term1} {term2} {term3}")
        
        return permutations

    async def _rank_keywords(self, permutations: List[str], base_arguments: Dict) -> List[Dict]:
        """Rank keyword permutations based on relevance"""
        prompt = f"""
        Rank the following search terms based on their relevance to the research context:
        
        Research context:
        {base_arguments}
        
        Search terms:
        {permutations}
        
        For each term, provide:
        1. Relevance score (0-1)
        2. Brief explanation
        3. Suggested use case
        """
        
        response = self.model.generate_content(prompt)
        ranked_terms = self._parse_ranking_response(response.text, permutations)
        
        # Sort by relevance score
        ranked_terms.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return ranked_terms

    def _parse_ranking_response(self, response: str, permutations: List[str]) -> List[Dict]:
        """Parse the ranking response into structured format"""
        # Implementation for parsing the response
        # This would convert the text response into a structured list of dictionaries
        return [
            {
                "term": term,
                "relevance_score": 0.0,
                "explanation": "",
                "use_case": ""
            }
            for term in permutations
        ]
