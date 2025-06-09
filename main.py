from langsmith import traceable
import asyncio
from dotenv import load_dotenv
import os
from agents.argument_extractor import ArgumentExtractor
from agents.keyword_generator import KeywordGenerator
from agents.source_crawler import SourceCrawler
from agents.citation_chainer import CitationChainer
from agents.relevance_scorer import RelevanceScorer
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Environment configuration
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "legal-research")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
TRACING_ENABLED = os.getenv("TRACING_ENABLED", "true").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configuration
MODEL_NAME = "models/gemini-1.5-flash-latest"

# Print configuration
print("Configuration:")
print(f"LANGSMITH_API_KEY: {LANGSMITH_API_KEY[:10]}...")
print(f"LANGSMITH_PROJECT: {LANGSMITH_PROJECT}")
print(f"LANGSMITH_ENDPOINT: {LANGSMITH_ENDPOINT}")
print(f"TRACING_ENABLED: {TRACING_ENABLED}")
print(f"MODEL_NAME: {MODEL_NAME}")

# Define the state schema
class ResearchState(TypedDict):
    """The state of the research process."""
    paper_content: str
    research_angle: str
    base_arguments: Dict
    research_analysis: Dict
    combined_context: Dict
    keywords: List[Dict]
    sources: List[Dict]
    scored_sources: List[Dict]
    messages: List[Dict]

# Initialize the model
model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_output_tokens=2048,
)

# Initialize the agents
extractor = ArgumentExtractor(model=model)
keyword_generator = KeywordGenerator()
source_crawler = SourceCrawler()
citation_chainer = CitationChainer()
relevance_scorer = RelevanceScorer()

# Define the nodes
async def extract_base_arguments(state: ResearchState) -> ResearchState:
    """Extract base arguments from the paper content."""
    print("\nExtracting base arguments...")
    base_arguments = await extractor._extract_base_arguments(state["paper_content"])
    return {**state, "base_arguments": base_arguments}

async def analyze_research_angle(state: ResearchState) -> ResearchState:
    """Analyze how the research angle relates to base arguments."""
    print("\nAnalyzing research angle...")
    research_analysis = await extractor._analyze_research_angle(
        state["research_angle"],
        state["base_arguments"]
    )
    return {**state, "research_analysis": research_analysis}

def combine_context(state: ResearchState) -> ResearchState:
    """Combine base arguments and research analysis into final context."""
    print("\nCombining context...")
    combined_context = {
        "base_paper": state["base_arguments"],
        "new_angle": state["research_analysis"]
    }
    return {**state, "combined_context": combined_context}

async def generate_keywords(state: ResearchState) -> ResearchState:
    """Generate keywords for research."""
    print("\nGenerating keywords...")
    keywords = await keyword_generator.generate_keywords(
        state["combined_context"],
        None  # No seed keywords
    )
    return {**state, "keywords": keywords}

async def crawl_sources(state: ResearchState) -> ResearchState:
    """Crawl sources based on keywords."""
    print("\nCrawling sources...")
    sources = await source_crawler.crawl_sources(state["keywords"])
    print(sources)
    return {**state, "sources": sources}

async def chain_citations(state: ResearchState) -> ResearchState:
    """Chain citations to find related sources."""
    print("\nChaining citations...")
    expanded_sources = await citation_chainer.chain_citations(
        state["sources"],
        state["combined_context"]
    )
    return {**state, "sources": expanded_sources}

async def score_sources(state: ResearchState) -> ResearchState:
    """Score sources based on relevance and provide reasoning."""
    print("\nScoring sources...")
    scored_sources = await relevance_scorer.score_sources(
        state["sources"],
        state["combined_context"]
    )
    
    # Add reasoning for each source
    for source in scored_sources:
        # Generate reasoning based on scores and context
        reasoning = {
            "relevance": f"This source is relevant because it has a relevance score of {source.relevance_score}",
            "quality": f"The source quality is based on citations count: {len(source.citations)}",
            "impact": f"Impact score based on date: {source.date}",
            "overall": f"Overall selection rationale: Source titled '{source.title}' with URL: {source.url}"
        }
        source.reasoning = reasoning
    
    return {**state, "scored_sources": scored_sources}

def print_results(state: ResearchState) -> ResearchState:
    """Print the final results with detailed reasoning."""
    print("\nExtracted Arguments:")
    print("\n1. base_arguments")
    for k, v in state["base_arguments"].items():
        print(f"  {k}: {v}\n")
    print("\n2. research_analysis")
    for k, v in state["research_analysis"].items():
        print(f"  {k}: {v}\n")
    print("\n3. combined_context")
    for k, v in state["combined_context"].items():
        print(f"  {k}: {v}\n")
    
    print("\n4. Keywords:")
    for keyword in state["keywords"][:5]:  # Show top 5 keywords
        print(f"  - {keyword['term']} (Score: {keyword['relevance_score']:.2f})")
    
    print("\n5. Top Sources with Reasoning:")
    for source in state["scored_sources"][:5]:  # Show top 5 sources
        print(f"\n  Title: {source.title}")
        print(f"  URL: {source.url}")
        print(f"  Content: {source.content}")
        print(f"  Relevance Score: {source.relevance_score}")
        print(f"  Date: {source.date}")
        print("\n  Selection Reasoning:")
        print(f"  - Relevance: {source.reasoning['relevance']}")
        print(f"  - Quality: {source.reasoning['quality']}")
        print(f"  - Impact: {source.reasoning['impact']}")
        print(f"  - Overall: {source.reasoning['overall']}")
        print("-" * 80)
    
    return state

# Create the graph
def create_research_graph() -> Graph:
    """Create the research workflow graph."""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("extract_base_arguments", extract_base_arguments)
    workflow.add_node("analyze_research_angle", analyze_research_angle)
    workflow.add_node("combine_context", combine_context)
    workflow.add_node("generate_keywords", generate_keywords)
    workflow.add_node("crawl_sources", crawl_sources)
    workflow.add_node("chain_citations", chain_citations)
    workflow.add_node("score_sources", score_sources)
    workflow.add_node("print_results", print_results)
    
    # Add edges
    workflow.add_edge("extract_base_arguments", "analyze_research_angle")
    workflow.add_edge("analyze_research_angle", "combine_context")
    workflow.add_edge("combine_context", "generate_keywords")
    workflow.add_edge("generate_keywords", "crawl_sources")
    workflow.add_edge("crawl_sources", "chain_citations")
    workflow.add_edge("chain_citations", "score_sources")
    workflow.add_edge("score_sources", "print_results")
    
    # Set entry and exit points
    workflow.set_entry_point("extract_base_arguments")
    workflow.set_finish_point("print_results")
    
    return workflow.compile()

@traceable(run_name="research_workflow", project_name=LANGSMITH_PROJECT)
async def process_paper(paper_path: str, research_angle: str):
    """Process a paper through the research workflow."""
    # Initialize the graph
    graph = create_research_graph()
    
    # Create initial state
    initial_state = {
        "paper_content": paper_path,  # This will be fetched by the extractor
        "research_angle": research_angle,
        "base_arguments": {},
        "research_analysis": {},
        "combined_context": {},
        "keywords": [],
        "sources": [],
        "scored_sources": [],
        "messages": []
    }
    
    # Run the graph
    final_state = await graph.ainvoke(initial_state)
    return final_state

async def main():
    if not TRACING_ENABLED:
        print("Tracing is disabled")
        return
    
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in environment variables")
        return
    
    # Process the law paper
    paper_path = "https://3fdef50c-add3-4615-a675-a91741bcb5c0.usrfiles.com/ugd/3fdef5_1c44141f484c4967a4259eb97b5333c2.pdf"
    research_angle = "Analyzing the impact of AI on intellectual property rights"
    
    print(f"\nProcessing paper: {paper_path}")
    print(f"Research angle: {research_angle}")
    
    try:
        final_state = await process_paper(paper_path, research_angle)
    except Exception as e:
        print(f"Error processing paper: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
