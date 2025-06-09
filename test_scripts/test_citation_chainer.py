import asyncio
import pytest
from agents.citation_chainer import CitationChainer, Source
from datetime import datetime
import os
from dotenv import load_dotenv
from unittest.mock import AsyncMock, patch

load_dotenv()

async def test_citation_chaining():
    # Print API credentials (without full key)
    print("\nAPI Configuration:")
    print(f"GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY', 'Not found')[:10]}...")
    print(f"GOOGLE_CSE_ID: {os.getenv('GOOGLE_CSE_ID', 'Not found')}")
    
    # Initialize the citation chainer
    chainer = CitationChainer()
    
    # Create a test source
    test_source = Source(
        title="Test Case Law Document",
        url="https://example.com/test-case",
        content="This case cites 410 U.S. 113 and 505 U.S. 833",
        citations=["410 U.S. 113", "505 U.S. 833"],
        relevance_score=0.0,
        date=datetime.now()
    )
    
    # Base arguments for relevance assessment
    base_arguments = {
        "keywords": ["abortion", "rights", "constitutional"],
        "topic": "Reproductive Rights"
    }
    
    # Test citation chaining
    results = await chainer.chain_citations([test_source], base_arguments)
    
    # Print results for inspection
    print("\nCitation Chaining Results:")
    print(f"Found {len(results)} sources")
    
    for source in results:
        print(f"\nSource: {source.title}")
        print(f"URL: {source.url}")
        print(f"Relevance Score: {source.relevance_score}")
        print(f"Citations: {source.citations}")
    
    # Basic assertions
    assert len(results) > 0, "Should find at least one source"
    assert all(isinstance(source, Source) for source in results), "All results should be Source objects"
    assert all(0 <= source.relevance_score <= 1 for source in results), "Relevance scores should be between 0 and 1"

@pytest.mark.asyncio
async def test_citation_chaining_with_mocked_network():
    chainer = CitationChainer()

    # Create a test source with citations
    test_source = Source(
        title="Test Case Law Document",
        url="https://example.com/test-case",
        content="This case cites 410 U.S. 113 and 505 U.S. 833",
        citations=["410 U.S. 113", "505 U.S. 833"],
        relevance_score=0.0,
        date=datetime.now()
    )

    # Base arguments for relevance assessment
    base_arguments = {
        "keywords": ["abortion", "rights", "constitutional"],
        "topic": "Reproductive Rights"
    }

    # Mocked Source objects for citations
    fake_sources = {
        "410 U.S. 113": [
            Source(
                title='Roe v. Wade',
                url='https://www.law.cornell.edu/supremecourt/text/410/113',
                content='Landmark Supreme Court decision on abortion rights.',
                citations=[],
                relevance_score=0.8,
                date=datetime.now()
            )
        ],
        "505 U.S. 833": [
            Source(
                title='Planned Parenthood v. Casey',
                url='https://www.law.cornell.edu/supremecourt/text/505/833',
                content='Another important case on reproductive rights.',
                citations=[],
                relevance_score=0.7,
                date=datetime.now()
            )
        ]
    }

    # Mock _follow_citation to return our fake sources
    async def mock_follow_citation(self, citation: str, base_args: dict) -> list:
        return fake_sources.get(citation, [])

    with patch.object(CitationChainer, '_follow_citation', new=mock_follow_citation):
        try:
            results = await chainer.chain_citations([test_source], base_arguments)
            
            # Print results for inspection
            print("\nCitation Chaining Results:")
            print(f"Found {len(results)} sources")
            for source in results:
                print(f"\nSource: {source.title}")
                print(f"URL: {source.url}")
                print(f"Relevance Score: {source.relevance_score}")
                print(f"Citations: {source.citations}")

            # Assertions
            assert len(results) == 3  # 1 original + 2 chained
            titles = [s.title for s in results]
            assert "Roe v. Wade" in titles
            assert "Planned Parenthood v. Casey" in titles
            assert all(isinstance(source, Source) for source in results)
            assert all(0 <= source.relevance_score <= 1 for source in results)
            
        except Exception as e:
            print(f"\nError during test: {str(e)}")
            raise

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_citation_chaining_with_mocked_network()) 