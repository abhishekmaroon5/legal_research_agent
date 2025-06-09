from semanticscholar import SemanticScholar
import time
import random
import logging
import sys
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sch = SemanticScholar()

def format_result(result: Any) -> str:
    # result is a Paper object
    return f"""
Title: {getattr(result, 'title', 'N/A')}
Authors: {', '.join([a.name for a in getattr(result, 'authors', [])])}
Year: {getattr(result, 'year', 'N/A')}
Citations: {getattr(result, 'citationCount', 0)}
URL: {getattr(result, 'url', 'N/A')}
Abstract: {getattr(result, 'abstract', 'N/A')[:200]}...
"""

def test_semantic_scholar_search():
    test_queries = [
        "artificial intelligence legal implications",
        "intellectual property rights AI",
        "machine learning copyright law"
    ]
    for query in test_queries:
        logger.info(f"\nTesting search for: {query}")
        try:
            delay = random.uniform(1, 2)
            logger.info(f"Waiting {delay:.1f} seconds before search...")
            time.sleep(delay)
            logger.info("Initiating search...")
            results = sch.search_paper(query, limit=1)
            first_result = next(iter(results), None)
            if first_result:
                logger.info("Success! First result:")
                print(format_result(first_result))
            else:
                logger.warning("No results found for this query")
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
        print("-" * 80)

def main():
    try:
        print("Testing Semantic Scholar Search Functionality")
        print("=" * 80)
        test_semantic_scholar_search()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 