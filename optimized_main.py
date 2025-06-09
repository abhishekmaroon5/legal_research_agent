"""Optimized main application for legal research agent."""
import asyncio
import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal
from dotenv import load_dotenv

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime and date objects."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

# Import optimized agents
from optimized_agents.argument_extractor import ArgumentExtractor
from optimized_agents.keyword_generator import KeywordGenerator
from optimized_agents.source_crawler import SourceCrawler
from optimized_agents.citation_chainer import CitationChainer
from optimized_agents.relevance_scorer import RelevanceScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Type definitions
class ResearchState:
    """State container for the research workflow."""
    
    def __init__(self):
        self.paper_content: str = ""
        self.research_angle: str = ""
        self.base_arguments: Dict[str, Any] = {}
        self.research_analysis: Dict[str, Any] = {}
        self.combined_context: Dict[str, Any] = {}
        self.keywords: List[Dict[str, Any]] = []
        self.sources: List[Dict[str, Any]] = []
        self.scored_sources: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            'start_time': datetime.now(),
            'end_time': None,
            'duration': None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            'paper_content': self.paper_content,
            'research_angle': self.research_angle,
            'base_arguments': self.base_arguments,
            'research_analysis': self.research_analysis,
            'combined_context': self.combined_context,
            'keywords': self.keywords,
            'sources': self.sources,
            'scored_sources': self.scored_sources,
            'metadata': {
                **self.metadata,
                'end_time': datetime.now().isoformat(),
                'duration': (datetime.now() - self.metadata['start_time']).total_seconds()
            }
        }

class ResearchWorkflow:
    """Optimized research workflow with parallel processing."""
    
    def __init__(self):
        """Initialize the research workflow with optimized agents."""
        self.argument_extractor = ArgumentExtractor()
        self.keyword_generator = KeywordGenerator()
        self.source_crawler = SourceCrawler()
        self.citation_chainer = CitationChainer()
        self.relevance_scorer = RelevanceScorer()
    
    async def process_paper(
        self,
        paper_path: str,
        research_angle: str,
        max_sources: int = 5
    ) -> Dict[str, Any]:
        """Process a paper through the research workflow.
        
        Args:
            paper_path: Path to the paper or URL
            research_angle: Research angle or question
            max_sources: Maximum number of sources to return
            
        Returns:
            Dict containing research results
        """
        state = ResearchState()
        state.paper_content = paper_path
        state.research_angle = research_angle
        
        try:
            # Step 1: Extract base arguments
            logger.info("Extracting base arguments...")
            state.base_arguments = await self.argument_extractor._extract_base_arguments(paper_path)
            
            # Step 2: Analyze research angle with the extracted base arguments
            logger.info("Analyzing research angle...")
            state.research_analysis = await self.argument_extractor._analyze_research_angle(
                research_angle, 
                state.base_arguments
            )
            
            state.combined_context = {
                'base_paper': state.base_arguments,
                'new_angle': state.research_analysis
            }
            
            # Step 2: Generate keywords
            logger.info("Generating keywords...")
            state.keywords = await self.keyword_generator.generate_keywords(
                state.combined_context
            )
            
            # Step 3: Find and process sources in parallel
            logger.info("Finding relevant sources...")
            source_tasks = []
            for keyword in state.keywords[:5]:  # Limit to top 5 keywords
                if isinstance(keyword, dict):
                    kw = keyword.get('keyword', '')
                else:
                    kw = str(keyword)
                source_tasks.append(
                    self.source_crawler.find_sources(kw, max_results=3)
                )
            
            # Gather all sources and flatten the list
            source_results = await asyncio.gather(*source_tasks)
            state.sources = [src for sublist in source_results for src in sublist]
            
            # Step 4: Score sources by relevance
            logger.info("Scoring sources...")
            scoring_tasks = []
            for source in state.sources:
                scoring_tasks.append(
                    self.relevance_scorer.score_source(
                        source,
                        state.research_angle,
                        state.base_arguments
                    )
                )
            
            # Get scored sources and sort by score
            scored_sources = await asyncio.gather(*scoring_tasks)
            state.scored_sources = sorted(
                scored_sources,
                key=lambda x: x.get('relevance_score', 0),
                reverse=True
            )[:max_sources]  # Return only top N sources
            
            # Step 5: Chain citations for top sources
            logger.info("Chaining citations...")
            citation_tasks = []
            for source in state.scored_sources[:3]:  # Limit to top 3 sources for citation chaining
                citation_tasks.append(
                    self.citation_chainer.find_related_citations(
                        source.get('title', ''),
                        source.get('url', '')
                    )
                )
            
            # Add related citations to sources
            related_citations = await asyncio.gather(*citation_tasks)
            for i, citations in enumerate(related_citations):
                if i < len(state.scored_sources):
                    state.scored_sources[i]['related_citations'] = citations
            
            logger.info("Research completed successfully!")
            return state.to_dict()
            
        except Exception as e:
            logger.error(f"Error in research workflow: {e}", exc_info=True)
            state.metadata['error'] = str(e)
            return state.to_dict()
        finally:
            # Ensure all resources are cleaned up
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        await asyncio.gather(
            self.argument_extractor.close(),
            self.keyword_generator.close(),
            self.source_crawler.close(),
            self.citation_chainer.close(),
            self.relevance_scorer.close()
        )

async def main():
    """Main entry point for the application."""
    # Define paper path and research angle
    paper_path = "https://3fdef50c-add3-4615-a675-a91741bcb5c0.usrfiles.com/ugd/3fdef5_1c44141f484c4967a4259eb97b5333c2.pdf"
    research_angle = "Analyzing the impact of AI on intellectual property rights"
    
    # Optional: Uncomment to use command line arguments instead
    # import argparse
    # parser = argparse.ArgumentParser(description='Legal Research Agent')
    # parser.add_argument('--paper', type=str, default=paper_path, help='Path to the paper or URL')
    # parser.add_argument('--angle', type=str, default=research_angle, help='Research angle or question')
    # parser.add_argument('--max-sources', type=int, default=5, help='Maximum number of sources to return')
    # parser.add_argument('--output', type=str, default='research_results.json', help='Output file path')
    # args = parser.parse_args()
    
    # Use hardcoded values
    output_file = 'research_results.json'
    max_sources = 5
    
    workflow = ResearchWorkflow()
    
    try:
        logger.info(f"Starting research workflow for paper: {paper_path}")
        logger.info(f"Research angle: {research_angle}")
        
        results = await workflow.process_paper(
            paper_path=paper_path,
            research_angle=research_angle,
            max_sources=max_sources
        )
        
        # Save results to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, cls=DateTimeEncoder)
            
        logger.info(f"Research completed. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in research workflow: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up resources
        await workflow.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
