from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import asyncio
import json
import os
import threading
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import uuid
from typing import Dict, List, TypedDict

# Import LangGraph and LangChain components
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Import your existing agents
from agents.argument_extractor import ArgumentExtractor
from agents.keyword_generator import KeywordGenerator
from agents.source_crawler import create_source_crawler_with_socketio
from agents.citation_chainer import CitationChainer
from agents.relevance_scorer import RelevanceScorer

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'legal-research-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*")

# Define the state schema for LangGraph
class ResearchState(TypedDict):
    """The state of the research process."""
    conversation_id: str
    paper_content: str
    research_angle: str
    base_arguments: Dict
    research_analysis: Dict
    combined_context: Dict
    keywords: List[Dict]
    sources: List[Dict]
    scored_sources: List[Dict]
    messages: List[Dict]
    progress: int
    current_step: str
    status: str

# Database setup
def init_db():
    """Initialize SQLite database for conversation history"""
    conn = sqlite3.connect('research_history.db')
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            paper_content TEXT NOT NULL,
            research_angle TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'completed',
            is_favorite BOOLEAN DEFAULT FALSE
        )
    ''')
    
    # Create research_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS research_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            base_arguments TEXT,
            research_analysis TEXT,
            keywords TEXT,
            sources TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
    ''')
    
    # Create user_preferences table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            preference_key TEXT NOT NULL,
            preference_value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Global variables to store research state
current_research = {
    'status': 'idle',
    'progress': 0,
    'current_step': '',
    'results': None,
    'error': None,
    'conversation_id': None
}

# Thread-safe lock for research operations
import threading
research_lock = threading.Lock()

# Model configuration
MODEL_NAME = "models/gemini-1.5-flash-latest"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize agents (these are used for the main thread and backup)
if GEMINI_API_KEY:
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
    )
    
    # Initialize agents
    extractor = ArgumentExtractor(model=model)
    keyword_generator = KeywordGenerator()
    source_crawler = create_source_crawler_with_socketio(socketio)  # Pass socketio for real-time updates
    citation_chainer = CitationChainer()
    relevance_scorer = RelevanceScorer()
else:
    print("Warning: GEMINI_API_KEY not found. Some features may not work.")
    # Create dummy agents to prevent errors
    extractor = None
    keyword_generator = None
    source_crawler = None
    citation_chainer = None
    relevance_scorer = None

# Create the LangGraph workflow
def create_research_graph():
    """Create the research workflow graph using global agents"""
    return create_research_graph_with_agents(
        extractor, keyword_generator, source_crawler, citation_chainer, relevance_scorer
    )

def create_research_graph_with_agents(extractor_agent, keyword_agent, crawler_agent, chainer_agent, scorer_agent):
    """Create the research workflow graph with specific agent instances"""
    workflow = StateGraph(ResearchState)
    
    # Create node functions that use the provided agents
    async def extract_base_arguments_local(state: ResearchState) -> ResearchState:
        """Extract base arguments from the paper content."""
        print("\nExtracting base arguments...")
        
        # Update progress
        state["progress"] = 15
        state["current_step"] = "Extracting base arguments..."
        state["status"] = "running"
        
        # Emit progress update
        socketio.emit('research_progress', {
            'progress': state["progress"],
            'current_step': state["current_step"],
            'status': state["status"]
        })
        
        base_arguments = await extractor_agent._extract_base_arguments(state["paper_content"])
        return {**state, "base_arguments": base_arguments}

    async def analyze_research_angle_local(state: ResearchState) -> ResearchState:
        """Analyze how the research angle relates to base arguments."""
        print("\nAnalyzing research angle...")
        
        # Update progress
        state["progress"] = 30
        state["current_step"] = "Analyzing research angle..."
        
        # Emit progress update
        socketio.emit('research_progress', {
            'progress': state["progress"],
            'current_step': state["current_step"],
            'status': state["status"]
        })
        
        research_analysis = await extractor_agent._analyze_research_angle(
            state["research_angle"],
            state["base_arguments"]
        )
        return {**state, "research_analysis": research_analysis}

    def combine_context_local(state: ResearchState) -> ResearchState:
        """Combine base arguments and research analysis into final context."""
        print("\nCombining context...")
        
        combined_context = {
            "base_paper": state["base_arguments"],
            "new_angle": state["research_analysis"]
        }
        return {**state, "combined_context": combined_context}

    async def generate_keywords_local(state: ResearchState) -> ResearchState:
        """Generate keywords for research."""
        print("\nGenerating keywords...")
        
        # Update progress
        state["progress"] = 50
        state["current_step"] = "Generating search keywords..."
        
        # Emit progress update
        socketio.emit('research_progress', {
            'progress': state["progress"],
            'current_step': state["current_step"],
            'status': state["status"]
        })
        
        keywords = await keyword_agent.generate_keywords(
            state["combined_context"],
            None  # No seed keywords
        )
        return {**state, "keywords": keywords}

    async def crawl_sources_local(state: ResearchState) -> ResearchState:
        """Crawl sources based on keywords."""
        print("\nCrawling sources...")
        
        # Update progress
        state["progress"] = 70
        state["current_step"] = "Searching for sources..."
        
        # Emit progress update
        socketio.emit('research_progress', {
            'progress': state["progress"],
            'current_step': state["current_step"],
            'status': state["status"]
        })
        
        sources = await crawler_agent.crawl_sources(state["keywords"])
        print(f"Found {len(sources)} sources")
        return {**state, "sources": sources}

    async def chain_citations_local(state: ResearchState) -> ResearchState:
        """Chain citations to find related sources."""
        print("\nChaining citations...")
        
        # Update progress
        state["progress"] = 85
        state["current_step"] = "Following citation chains..."
        
        # Emit progress update
        socketio.emit('research_progress', {
            'progress': state["progress"],
            'current_step': state["current_step"],
            'status': state["status"]
        })
        
        expanded_sources = await chainer_agent.chain_citations(
            state["sources"],
            state["combined_context"]
        )
        return {**state, "sources": expanded_sources}

    async def score_sources_local(state: ResearchState) -> ResearchState:
        """Score sources based on relevance and provide reasoning."""
        print("\nScoring sources...")
        
        # Update progress
        state["progress"] = 95
        state["current_step"] = "Scoring and ranking sources..."
        
        # Emit progress update
        socketio.emit('research_progress', {
            'progress': state["progress"],
            'current_step': state["current_step"],
            'status': state["status"]
        })
        
        scored_sources = await scorer_agent.score_sources(
            state["sources"],
            state["combined_context"]
        )
        
        # Add reasoning for each source
        for source in scored_sources:
            if hasattr(source, 'relevance_score') and hasattr(source, 'citations') and hasattr(source, 'date'):
                reasoning = {
                    "relevance": f"This source is relevant because it has a relevance score of {source.relevance_score}",
                    "quality": f"The source quality is based on citations count: {len(source.citations) if source.citations else 0}",
                    "impact": f"Impact score based on date: {source.date}",
                    "overall": f"Overall selection rationale: Source titled '{source.title}' with URL: {source.url}"
                }
                source.reasoning = reasoning
        
        return {**state, "scored_sources": scored_sources}

    def finalize_results_local(state: ResearchState) -> ResearchState:
        """Finalize the research results."""
        print("\nFinalizing results...")
        
        # Update progress
        state["progress"] = 100
        state["current_step"] = "Research completed!"
        state["status"] = "completed"
        
        print(f"\nExtracted Arguments:")
        print(f"1. base_arguments: {len(state['base_arguments']) if isinstance(state['base_arguments'], dict) else 'N/A'} items")
        print(f"2. research_analysis: {len(state['research_analysis']) if isinstance(state['research_analysis'], dict) else 'N/A'} items")
        print(f"3. keywords: {len(state['keywords'])} generated")
        print(f"4. sources: {len(state['scored_sources'])} sources found and scored")
        
        return state
    
    # Add nodes with local functions
    workflow.add_node("extract_base_arguments", extract_base_arguments_local)
    workflow.add_node("analyze_research_angle", analyze_research_angle_local)
    workflow.add_node("combine_context", combine_context_local)
    workflow.add_node("generate_keywords", generate_keywords_local)
    workflow.add_node("crawl_sources", crawl_sources_local)
    workflow.add_node("chain_citations", chain_citations_local)
    workflow.add_node("score_sources", score_sources_local)
    workflow.add_node("finalize_results", finalize_results_local)
    
    # Add edges
    workflow.add_edge("extract_base_arguments", "analyze_research_angle")
    workflow.add_edge("analyze_research_angle", "combine_context")
    workflow.add_edge("combine_context", "generate_keywords")
    workflow.add_edge("generate_keywords", "crawl_sources")
    workflow.add_edge("crawl_sources", "chain_citations")
    workflow.add_edge("chain_citations", "score_sources")
    workflow.add_edge("score_sources", "finalize_results")
    
    # Set entry and exit points
    workflow.set_entry_point("extract_base_arguments")
    workflow.set_finish_point("finalize_results")
    
    return workflow.compile()

# Flask routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get all conversation history"""
    try:
        conn = sqlite3.connect('research_history.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, paper_content, research_angle, created_at, status, is_favorite
            FROM conversations 
            ORDER BY created_at DESC
        ''')
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'id': row[0],
                'title': row[1],
                'paper_content': row[2][:200] + '...' if len(row[2]) > 200 else row[2],
                'research_angle': row[3],
                'created_at': row[4],
                'status': row[5],
                'is_favorite': bool(row[6])
            })
        
        conn.close()
        return jsonify(conversations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get specific conversation with results"""
    try:
        conn = sqlite3.connect('research_history.db')
        cursor = conn.cursor()
        
        # Get conversation details
        cursor.execute('''
            SELECT id, title, paper_content, research_angle, created_at, status, is_favorite
            FROM conversations WHERE id = ?
        ''', (conversation_id,))
        
        conv_row = cursor.fetchone()
        if not conv_row:
            conn.close()
            return jsonify({'error': 'Conversation not found'}), 404
        
        conversation = {
            'id': conv_row[0],
            'title': conv_row[1],
            'paper_content': conv_row[2],
            'research_angle': conv_row[3],
            'created_at': conv_row[4],
            'status': conv_row[5],
            'is_favorite': bool(conv_row[6])
        }
        
        # Get research results
        cursor.execute('''
            SELECT base_arguments, research_analysis, keywords, sources, summary
            FROM research_results WHERE conversation_id = ?
            ORDER BY created_at DESC LIMIT 1
        ''', (conversation_id,))
        
        result_row = cursor.fetchone()
        if result_row:
            conversation['results'] = {
                'base_arguments': json.loads(result_row[0]) if result_row[0] else {},
                'research_analysis': json.loads(result_row[1]) if result_row[1] else {},
                'keywords': json.loads(result_row[2]) if result_row[2] else [],
                'sources': json.loads(result_row[3]) if result_row[3] else [],
                'summary': json.loads(result_row[4]) if result_row[4] else {}
            }
        
        conn.close()
        return jsonify(conversation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<conversation_id>/favorite', methods=['POST'])
def toggle_favorite(conversation_id):
    """Toggle favorite status of a conversation"""
    try:
        conn = sqlite3.connect('research_history.db')
        cursor = conn.cursor()
        
        # Get current favorite status
        cursor.execute('SELECT is_favorite FROM conversations WHERE id = ?', (conversation_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Toggle favorite status
        new_status = not bool(row[0])
        cursor.execute('''
            UPDATE conversations SET is_favorite = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (new_status, conversation_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({'is_favorite': new_status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation and its results"""
    try:
        conn = sqlite3.connect('research_history.db')
        cursor = conn.cursor()
        
        # Delete research results first (foreign key constraint)
        cursor.execute('DELETE FROM research_results WHERE conversation_id = ?', (conversation_id,))
        
        # Delete conversation
        cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Conversation not found'}), 404
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Conversation deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/research', methods=['POST'])
def start_research():
    """Start the research process using LangGraph workflow"""
    global current_research
    
    # Check if research is already running
    with research_lock:
        if current_research['status'] == 'running':
            return jsonify({'error': 'Research already in progress. Please wait for the current research to complete.'}), 400
        
        # Mark research as starting
        current_research['status'] = 'starting'
    
    data = request.get_json()
    paper_content = data.get('paper_content', '')
    research_angle = data.get('research_angle', '')
    conversation_id = data.get('conversation_id')  # Optional for continuing existing conversation
    
    if not paper_content or not research_angle:
        current_research['status'] = 'idle'  # Reset status
        return jsonify({'error': 'Paper content and research angle are required'}), 400
    
    # Generate new conversation ID if not provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        
        # Save conversation to database
        try:
            conn = sqlite3.connect('research_history.db')
            cursor = conn.cursor()
            
            # Generate title from research angle (first 50 chars)
            title = research_angle[:50] + '...' if len(research_angle) > 50 else research_angle
            
            cursor.execute('''
                INSERT INTO conversations (id, title, paper_content, research_angle, status)
                VALUES (?, ?, ?, ?, 'running')
            ''', (conversation_id, title, paper_content, research_angle))
            
            conn.commit()
            conn.close()
        except Exception as e:
            current_research['status'] = 'idle'  # Reset status
            return jsonify({'error': f'Failed to save conversation: {str(e)}'}), 500
    
    # Reset research state
    with research_lock:
        current_research = {
            'status': 'running',
            'progress': 0,
            'current_step': 'Starting research...',
            'results': None,
            'error': None,
            'conversation_id': conversation_id
        }
    
    # Start research in background thread using LangGraph
    thread = threading.Thread(
        target=run_langgraph_research,
        args=(paper_content, research_angle, conversation_id),
        daemon=True  # Make thread daemon so it doesn't prevent app shutdown
    )
    thread.start()
    
    return jsonify({
        'message': 'Research started successfully',
        'conversation_id': conversation_id
    })

@app.route('/api/status')
def get_status():
    """Get current research status"""
    return jsonify(current_research)

@app.route('/api/results')
def get_results():
    """Get research results"""
    if current_research['results']:
        return jsonify(current_research['results'])
    return jsonify({'error': 'No results available'}), 404

@app.route('/api/research/reset', methods=['POST'])
def reset_research():
    """Reset research state (useful if stuck)"""
    global current_research
    
    with research_lock:
        current_research = {
            'status': 'idle',
            'progress': 0,
            'current_step': '',
            'results': None,
            'error': None,
            'conversation_id': None
        }
    
    return jsonify({'message': 'Research state reset successfully'})

def run_langgraph_research(paper_content, research_angle, conversation_id):
    """Run the LangGraph research workflow with proper async handling"""
    # Create a completely new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the research workflow
        loop.run_until_complete(execute_research_workflow(paper_content, research_angle, conversation_id))
        
    except Exception as e:
        print(f"Error in LangGraph research: {str(e)}")
        
        # Update global state with thread safety
        with research_lock:
            current_research['status'] = 'error'
            current_research['error'] = str(e)
        
        # Emit error event
        socketio.emit('research_error', {'error': str(e)})
        
        # Update conversation status in database
        try:
            conn = sqlite3.connect('research_history.db')
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE conversations SET status = 'error', updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (conversation_id,))
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"Database update error: {str(db_error)}")
    
    finally:
        # Always close the loop when done
        try:
            # Cancel any remaining tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Wait for tasks to be cancelled
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Close the loop
            loop.close()
        except Exception as cleanup_error:
            print(f"Error during loop cleanup: {str(cleanup_error)}")

async def execute_research_workflow(paper_content, research_angle, conversation_id):
    """Execute the actual research workflow with proper async context"""
    try:
        # Create fresh instances of agents for this loop
        model = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )
        
        # Initialize agents with the current loop's context
        extractor_local = ArgumentExtractor(model=model)
        keyword_generator_local = KeywordGenerator()
        source_crawler_local = create_source_crawler_with_socketio(socketio)
        citation_chainer_local = CitationChainer()
        relevance_scorer_local = RelevanceScorer()
        
        # Create the research graph with local agents
        graph = create_research_graph_with_agents(
            extractor_local,
            keyword_generator_local, 
            source_crawler_local,
            citation_chainer_local,
            relevance_scorer_local
        )
        
        # Create initial state
        initial_state = {
            "conversation_id": conversation_id,
            "paper_content": paper_content,
            "research_angle": research_angle,
            "base_arguments": {},
            "research_analysis": {},
            "combined_context": {},
            "keywords": [],
            "sources": [],
            "scored_sources": [],
            "messages": [],
            "progress": 0,
            "current_step": "Initializing...",
            "status": "running"
        }
        
        # Run the graph
        final_state = await graph.ainvoke(initial_state)
        
        # Format results for UI
        results = format_results_for_ui(final_state)
        
        # Save results to database
        save_research_results(conversation_id, results)
        
        # Update global state with thread safety
        with research_lock:
            current_research['status'] = 'completed'
            current_research['progress'] = 100
            current_research['current_step'] = 'Research completed!'
            current_research['results'] = results
        
        # Emit completion event
        socketio.emit('research_complete', results)
        
    except Exception as e:
        print(f"Error in research workflow execution: {str(e)}")
        
        # Update global state with thread safety
        with research_lock:
            current_research['status'] = 'error'
            current_research['error'] = str(e)
        
        # Emit error event
        socketio.emit('research_error', {'error': str(e)})
        
        # Update conversation status in database
        try:
            conn = sqlite3.connect('research_history.db')
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE conversations SET status = 'error', updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (conversation_id,))
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"Database update error: {str(db_error)}")
        
        raise  # Re-raise the exception

def format_results_for_ui(state):
    """Format the LangGraph state results for UI display"""
    return {
        'conversation_id': state['conversation_id'],
        'base_arguments': state['base_arguments'],
        'research_analysis': state['research_analysis'],
        'keywords': state['keywords'][:10] if state['keywords'] else [],
        'sources': format_sources_for_ui(state['scored_sources'][:20] if state['scored_sources'] else []),
        'summary': generate_research_summary(
            state['base_arguments'], 
            state['research_analysis'], 
            state['scored_sources']
        ),
        'timestamp': datetime.now().isoformat()
    }

def format_sources_for_ui(sources):
    """Format sources for UI display"""
    formatted_sources = []
    for source in sources:
        if hasattr(source, 'title'):
            # Handle Source objects
            formatted_source = {
                'title': source.title,
                'url': source.url,
                'content': source.content[:500] + '...' if len(source.content) > 500 else source.content,
                'relevance_score': getattr(source, 'relevance_score', 0),
                'date': source.date.isoformat() if source.date else None,
                'citations': len(source.citations) if source.citations else 0,
                'reasoning': getattr(source, 'reasoning', {})
            }
        else:
            # Handle dict objects
            formatted_source = {
                'title': source.get('title', ''),
                'url': source.get('url', ''),
                'content': source.get('content', '')[:500] + '...' if len(source.get('content', '')) > 500 else source.get('content', ''),
                'relevance_score': source.get('relevance_score', 0),
                'date': None,
                'citations': 0,
                'reasoning': source.get('reasoning', {})
            }
        formatted_sources.append(formatted_source)
    return formatted_sources

def generate_research_summary(base_arguments, research_analysis, sources):
    """Generate a summary of the research"""
    return {
        'total_sources': len(sources) if sources else 0,
        'key_themes': list(base_arguments.keys()) if isinstance(base_arguments, dict) else [],
        'research_focus': research_analysis.get('focus_areas', []) if isinstance(research_analysis, dict) else [],
        'top_source_types': ['legal', 'news'],
    }

def save_research_results(conversation_id, results):
    """Save research results to database"""
    try:
        conn = sqlite3.connect('research_history.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO research_results 
            (conversation_id, base_arguments, research_analysis, keywords, sources, summary)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            conversation_id,
            json.dumps(results.get('base_arguments', {})),
            json.dumps(results.get('research_analysis', {})),
            json.dumps(results.get('keywords', [])),
            json.dumps(results.get('sources', [])),
            json.dumps(results.get('summary', {}))
        ))
        
        # Update conversation status
        cursor.execute('''
            UPDATE conversations SET status = 'completed', updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (conversation_id,))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving research results: {str(e)}")

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'data': 'Connected to research server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("🚀 Starting Enhanced Legal Research Assistant with LangGraph...")
    print(f"🔗 Web Interface: http://localhost:5000")
    print(f"🤖 LangGraph Workflow: Enabled")
    print(f"💾 Memory System: Enabled")
    print(f"📊 Real-time Progress: Enabled")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)