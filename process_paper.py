import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio

from agents.argument_extractor import ArgumentExtractor
from agents.keyword_generator import KeywordGenerator
from agents.source_crawler import SourceCrawler
from agents.citation_chainer import CitationChainer
from agents.relevance_scorer import RelevanceScorer
from agents.memory_agent import MemoryAgent

# Load environment variables
load_dotenv()

class LegalResearchProcessor:
    def __init__(self):
        # Initialize LangChain components
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Initialize Tavily search tool
        self.search_tool = TavilySearchResults(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=5
        )
        
        # Initialize our agents
        self.argument_extractor = ArgumentExtractor()
        self.keyword_generator = KeywordGenerator()
        self.source_crawler = SourceCrawler()
        self.citation_chainer = CitationChainer()
        self.relevance_scorer = RelevanceScorer()
        self.memory_agent = MemoryAgent()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    async def process_paper(self, pdf_path: str, research_angle: str = None):
        """
        Process a legal paper and perform research
        """
        # 1. Load and split the PDF
        print("Loading PDF...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
        chunks = self.text_splitter.split_text(text)
        
        # 2. Create vector store
        print("Creating vector store...")
        vectorstore = FAISS.from_texts(chunks, self.embeddings)
        
        # 3. Extract arguments
        print("Extracting arguments...")
        base_arguments = await self.argument_extractor.extract_arguments(
            text,
            research_angle
        )
        
        # 4. Generate keywords
        print("Generating keywords...")
        keywords = await self.keyword_generator.generate_keywords(
            base_arguments,
            None  # No seed keywords
        )
        
        # 5. Crawl sources
        print("Crawling sources...")
        initial_sources = await self.source_crawler.crawl_sources(keywords)
        
        # 6. Chain citations
        print("Chaining citations...")
        expanded_sources = await self.citation_chainer.chain_citations(
            initial_sources,
            base_arguments
        )
        
        # 7. Score relevance
        print("Scoring sources...")
        scored_sources = await self.relevance_scorer.score_sources(
            expanded_sources,
            base_arguments
        )
        
        # 8. Update memory
        print("Updating memory...")
        await self.memory_agent.update_memory(scored_sources)
        
        return {
            "base_arguments": base_arguments,
            "keywords": keywords,
            "sources": scored_sources
        }

async def main():
    # Initialize processor
    processor = LegalResearchProcessor()
    
    # Process the paper
    result = await processor.process_paper(
        pdf_path="law_paper1.pdf",
        research_angle="Analyzing the impact of AI on intellectual property rights"
    )
    
    # Print results
    print("\nBase Arguments:")
    print(result["base_arguments"])
    
    print("\nTop Keywords:")
    for keyword in result["keywords"][:5]:
        print(f"- {keyword['term']} (Score: {keyword['relevance_score']:.2f})")
    
    print("\nTop Sources:")
    for source in result["sources"][:5]:
        print(f"\nTitle: {source['title']}")
        print(f"Score: {source['scores']['final']:.2f}")
        print(f"URL: {source['url']}")

if __name__ == "__main__":
    asyncio.run(main()) 