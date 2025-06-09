from typing import List, Dict
import json
import os
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MemoryAgent:
    def __init__(self):
        self.memory_file = "data/memory.json"
        self.memory = self._load_memory()
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )

    def _load_memory(self) -> Dict:
        """Load memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading memory: {str(e)}")
        
        # Initialize empty memory
        return {
            "accepted_sources": [],
            "rejected_sources": [],
            "user_preferences": {
                "source_types": {},
                "topics": {},
                "authors": {},
                "journals": {}
            },
            "search_history": [],
            "last_updated": datetime.now().isoformat()
        }

    def _save_memory(self):
        """Save memory to file"""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {str(e)}")

    async def update_memory(self, sources: List[Dict]):
        """Update memory with new sources"""
        # Update search history
        self.memory["search_history"].append({
            "timestamp": datetime.now().isoformat(),
            "sources": sources
        })
        
        # Keep only last 100 searches
        if len(self.memory["search_history"]) > 100:
            self.memory["search_history"] = self.memory["search_history"][-100:]
        
        # Update last_updated
        self.memory["last_updated"] = datetime.now().isoformat()
        
        # Save memory
        self._save_memory()

    async def accept_source(self, source: Dict):
        """Record an accepted source"""
        self.memory["accepted_sources"].append({
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update user preferences
        await self._update_preferences(source, accepted=True)
        
        # Save memory
        self._save_memory()

    async def reject_source(self, source: Dict):
        """Record a rejected source"""
        self.memory["rejected_sources"].append({
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update user preferences
        await self._update_preferences(source, accepted=False)
        
        # Save memory
        self._save_memory()

    async def _update_preferences(self, source: Dict, accepted: bool):
        """Update user preferences based on source acceptance/rejection"""
        # Update source type preferences
        source_type = source.get("source_type", "")
        if source_type:
            if source_type not in self.memory["user_preferences"]["source_types"]:
                self.memory["user_preferences"]["source_types"][source_type] = 0
            self.memory["user_preferences"]["source_types"][source_type] += 1 if accepted else -1
        
        # Update topic preferences
        topics = self._extract_topics(source)
        for topic in topics:
            if topic not in self.memory["user_preferences"]["topics"]:
                self.memory["user_preferences"]["topics"][topic] = 0
            self.memory["user_preferences"]["topics"][topic] += 1 if accepted else -1
        
        # Update author preferences
        authors = source.get("authors", "").split(", ")
        for author in authors:
            if author and author not in self.memory["user_preferences"]["authors"]:
                self.memory["user_preferences"]["authors"][author] = 0
            self.memory["user_preferences"]["authors"][author] += 1 if accepted else -1
        
        # Update journal preferences
        journal = source.get("journal", "")
        if journal:
            if journal not in self.memory["user_preferences"]["journals"]:
                self.memory["user_preferences"]["journals"][journal] = 0
            self.memory["user_preferences"]["journals"][journal] += 1 if accepted else -1

    def _extract_topics(self, source: Dict) -> List[str]:
        """Extract topics from source"""
        # This is a simplified implementation
        # In a real system, you might use NLP to extract topics
        topics = []
        
        # Extract from title
        title = source.get("title", "").lower()
        if "copyright" in title:
            topics.append("copyright")
        if "patent" in title:
            topics.append("patent")
        if "trademark" in title:
            topics.append("trademark")
        
        return topics

    async def get_recommendations(self, query: str, num_recommendations: int = 5) -> List[Dict]:
        """Get personalized recommendations based on user preferences"""
        # Prepare text for TF-IDF
        accepted_sources = [s["source"] for s in self.memory["accepted_sources"]]
        texts = [self._prepare_text(source) for source in accepted_sources]
        
        if not texts:
            return []
        
        # Calculate TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate query vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[-num_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:
            source = accepted_sources[idx]
            source["similarity_score"] = float(similarities[idx])
            recommendations.append(source)
        
        return recommendations

    def _prepare_text(self, source: Dict) -> str:
        """Prepare text for TF-IDF analysis"""
        text_parts = []
        for field in ["title", "abstract", "content"]:
            if field in source:
                text_parts.append(str(source[field]))
        return " ".join(text_parts)

    def get_user_preferences(self) -> Dict:
        """Get current user preferences"""
        return self.memory["user_preferences"]

    def get_search_history(self) -> List[Dict]:
        """Get search history"""
        return self.memory["search_history"]

    def clear_memory(self):
        """Clear all memory"""
        self.memory = {
            "accepted_sources": [],
            "rejected_sources": [],
            "user_preferences": {
                "source_types": {},
                "topics": {},
                "authors": {},
                "journals": {}
            },
            "search_history": [],
            "last_updated": datetime.now().isoformat()
        }
        self._save_memory()
