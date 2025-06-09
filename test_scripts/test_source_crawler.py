import asyncio
from agents.source_crawler import SourceCrawler
import time

async def test_source_crawler():
    # Initialize the crawler
    crawler = SourceCrawler()
    
    # Test keywords with relevance scores
    test_keywords = [
        {"term": "AI copyright infringement", "relevance_score": 0.9},
        {"term": "machine learning patent law", "relevance_score": 0.85},
        {"term": "artificial intelligence legal liability", "relevance_score": 0.8},
        {"term": "deep learning intellectual property", "relevance_score": 0.75},
        {"term": "neural networks legal framework", "relevance_score": 0.7}
    ]
    
    print("Starting source crawl with 5 test keywords...")
    print("=" * 80)
    start_time = time.time()
    
    sources = await crawler.crawl_sources(test_keywords)
    end_time = time.time()
    print(f"\nSearch completed in {end_time - start_time:.2f} seconds")
    print(f"\nFound {len(sources)} unique sources")
    print("=" * 80)
    
    # Print sources grouped by type (legal and news only)
    for source_type in ["legal", "news"]:
        type_sources = [s for s in sources if s["source_type"] == source_type]
        print(f"\n{source_type.upper()} SOURCES ({len(type_sources)}):")
        print("-" * 40)
        for source in type_sources:
            print(f"\nTitle: {source.get('title', 'N/A')}")
            print(f"URL: {source.get('url', 'N/A')}")
            print(f"Relevance Score: {source.get('relevance_score', 0):.2f}")
            print(f"Weighted Score: {source.get('weighted_score', 0):.2f}")
            print(f"Timestamp: {source.get('timestamp', 'N/A')}")
            print("-" * 40)

if __name__ == "__main__":
    asyncio.run(test_source_crawler()) 