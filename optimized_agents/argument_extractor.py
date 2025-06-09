"""Optimized argument extractor with caching and rate limiting."""
from typing import Dict, List, Any, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import os
import aiohttp
import asyncio
import PyPDF2
import io
import re
import json
from bs4 import BeautifulSoup
import logging
from .base_agent import BaseAgent
from dataclasses import dataclass, field
from datetime import datetime

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PaperAnalysis:
    """Structured analysis of a legal paper."""
    thesis: str
    supporting_points: List[str]
    methodology: str
    findings: List[str]
    limitations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class ArgumentExtractor(BaseAgent[Dict]):
    """Optimized argument extractor with caching and rate limiting."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        """Initialize the argument extractor.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        super().__init__(model_name=model_name)
        
        # Verify API key is set
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
            
        try:
            # Configure the API
            genai.configure(api_key=api_key)
            
            # List available models to verify API key works
            available_models = [m.name for m in genai.list_models()]
            logger.debug(f"Available models: {available_models}")
            
            if f"models/{model_name}" not in available_models:
                logger.warning(f"Model {model_name} not found. Using default model.")
                model_name = "gemini-1.5-flash-latest"  # Fallback to default model
                
            # Initialize the model
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
        
        # Configure rate limits
        self.rate_limits = {
            'gemini': 2,  # 2 requests per second
            'http': 10,   # 10 HTTP requests per second
        }
        
        # Set model configuration
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    
    async def process(self, paper_content_or_url: str, research_angle: str = "") -> Dict:
        """Extract and analyze arguments from paper content or URL.
        
        Args:
            paper_content_or_url: Either the full text content or a URL to fetch
            research_angle: Optional research angle to analyze against
            
        Returns:
            Dict containing analysis results
        """
        # Check if input is a URL
        if self._is_url(paper_content_or_url):
            paper_content = await self._fetch_paper_content(paper_content_or_url)
        else:
            paper_content = paper_content_or_url
        
        # Create default research analysis
        default_analysis = {
            'builds_upon': [],
            'diverges': [],
            'gaps_addressed': [],
            'focus_areas': []
        }
        
        # First, extract base arguments
        base_analysis = await self._extract_base_arguments(paper_content)
        
        # Then analyze research angle if provided
        if research_angle and base_analysis:
            try:
                research_analysis = await self._analyze_research_angle(
                    research_angle, 
                    base_analysis
                )
            except Exception as e:
                logger.error(f"Error analyzing research angle: {e}")
                research_analysis = default_analysis
        else:
            research_analysis = default_analysis
        
        return {
            'base_arguments': base_analysis,
            'research_analysis': research_analysis,
            'combined_context': {
                'base_paper': base_analysis,
                'new_angle': research_analysis
            }
        }
    
    async def _extract_base_arguments(self, paper_content: str) -> Dict:
        """Extract core arguments from the paper content.
        
        Args:
            paper_content: Either the text content or a URL to a PDF/HTML document
            
        Returns:
            Dict containing the extracted arguments
        """
        # Default response in case of errors
        default_response = {
            'thesis': '',
            'supporting_points': [],
            'methodology': '',
            'findings': [],
            'limitations': []
        }
        
        # Validate input
        if not paper_content or not isinstance(paper_content, str):
            logger.error("Invalid paper content provided")
            return default_response
            
        # Use cached version if available
        cache_key = self.cache_key('_extract_base_arguments', paper_content[:1000])
        if cached := await self.get_cached(cache_key):
            logger.debug("Using cached base arguments")
            return cached
        
        try:
            # Check if the input is a URL
            if self._is_url(paper_content):
                logger.info(f"Fetching content from URL: {paper_content}")
                paper_content = await self._fetch_paper_content(paper_content)
                if not paper_content:
                    logger.error("Failed to fetch content from URL")
                    return default_response
            
            # Process in chunks if content is large
            if len(paper_content) > 4000:
                chunks = self._chunk_text(paper_content, 4000)
                content_to_analyze = chunks[0] + "\n\n[...truncated...]\n\n" + chunks[-1]
            else:
                content_to_analyze = paper_content
            
            logger.debug(f"Processing paper content (length: {len(content_to_analyze)} chars)")
            
            # Create a more structured prompt
            system_prompt = """You are a legal research assistant. Analyze the provided legal paper and extract the following information in JSON format:
            {
                "thesis": "Main thesis or central argument of the paper",
                "supporting_points": ["Key supporting argument 1", "Key supporting argument 2", ...],
                "methodology": "Research methodology used in the paper",
                "findings": ["Key finding 1", "Key finding 2", ...],
                "limitations": ["Limitation 1", "Limitation 2", ...]
            }"""
            
            user_prompt = f"""Analyze the following legal paper and extract the requested information in the specified JSON format:
            
            {content_to_analyze}"""
            
            logger.debug("Sending request to Gemini model")
            
            try:
                # Debug: Print model info
                logger.debug(f"Model info: {self.model}")
                logger.debug(f"Model name: {getattr(self.model, '_model_name', 'unknown')}")
                
                # Initialize chat with the model and generation config
                chat = self.model.start_chat(history=[])
                logger.debug("Chat session started")
                
                # Prepare the messages in the format expected by the model
                messages = [
                    {"role": "user", "parts": [user_prompt]}
                ]
                
                # Send the request with generation config
                logger.debug("Sending message to Gemini API")
                response = await asyncio.to_thread(
                    chat.send_message,
                    user_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=8192,
                    )
                )
                
                logger.debug("Received response from Gemini model")
                
                # Debug: Print response object structure
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response attributes: {dir(response)}")
                
                # Get the response text
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    response_text = response.candidates[0].content.parts[0].text
                elif hasattr(response, 'parts') and response.parts:
                    response_text = response.parts[0].text
                elif hasattr(response, 'content') and hasattr(response.content, 'parts') and response.content.parts:
                    response_text = response.content.parts[0].text
                else:
                    response_text = str(response)
                
                logger.debug(f"Extracted response text (first 200 chars): {response_text[:200]}...")
                
                # Parse the response
                result = self._parse_json_response(response_text)
                logger.debug(f"Parsed JSON result: {result}")
                
                # Ensure all required keys are present
                required_keys = ['thesis', 'supporting_points', 'methodology', 'findings', 'limitations']
                if not all(key in result for key in required_keys):
                    logger.warning("Response missing required keys, using default structure")
                    result = {
                        'thesis': result.get('thesis', ''),
                        'supporting_points': result.get('supporting_points', []),
                        'methodology': result.get('methodology', ''),
                        'findings': result.get('findings', []),
                        'limitations': result.get('limitations', [])
                    }
                
                # Cache the result for 1 hour
                await self.set_cached(cache_key, result, ttl=3600)
                return result
                
            except Exception as e:
                logger.error(f"Error in Gemini API call: {str(e)}", exc_info=True)
                return default_response
                    
        except Exception as e:
            logger.error(f"Error in _extract_base_arguments: {str(e)}", exc_info=True)
            return default_response
    
    async def _analyze_research_angle(self, research_angle: str, base_arguments: Dict) -> Dict:
        """Analyze how the research angle relates to base arguments."""
        cache_key = self.cache_key('_analyze_research_angle', research_angle, str(base_arguments))
        if cached := await self.get_cached(cache_key):
            return cached
        
        # Create a more structured prompt
        system_prompt = """You are a legal research assistant. Analyze how the provided research angle relates to the base paper's arguments. Provide your analysis in the following JSON format:
        {
            "builds_upon": ["How the research angle builds on the paper's arguments"],
            "diverges": ["How the research angle differs from the paper's focus"],
            "gaps_addressed": ["What gaps in the paper the research angle addresses"],
            "focus_areas": ["Suggested focus areas for the research"]
        }"""
        
        user_prompt = f"""Base paper arguments:
        {json.dumps(base_arguments, indent=2) if base_arguments else 'No base arguments provided'}
        
        Research angle to analyze:
        {research_angle}
        
        Please provide your analysis in the requested JSON format."""
        
        try:
            async with self.rate_limiter('gemini'):
                # Use the chat interface for better structured responses
                chat = self.model.start_chat(history=[])
                response = await asyncio.to_thread(
                    chat.send_message,
                    [system_prompt, user_prompt]
                )
                
                # Parse the response
                result = self._parse_json_response(response.text)
                
                # Ensure all required keys are present
                required_keys = ['builds_upon', 'diverges', 'gaps_addressed', 'focus_areas']
                if not all(key in result for key in required_keys):
                    logger.warning("Response missing required keys, using default structure")
                    result = {
                        'builds_upon': result.get('builds_upon', []),
                        'diverges': result.get('diverges', []),
                        'gaps_addressed': result.get('gaps_addressed', []),
                        'focus_areas': result.get('focus_areas', [])
                    }
                
                # Cache the result for 1 hour
                await self.set_cached(cache_key, result, ttl=3600)
                return result
                
        except Exception as e:
            logger.error(f"Error analyzing research angle: {e}")
            return {
                'builds_upon': [],
                'diverges': [],
                'gaps_addressed': [],
                'focus_areas': []
            }
    
    async def _fetch_paper_content(self, url: str) -> str:
        """Fetch and parse paper content from URL with caching."""
        try:
            logger.info(f"Attempting to fetch content from URL: {url}")
            
            # Set a custom user agent to avoid 403 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
            
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                try:
                    async with session.get(url, allow_redirects=True, ssl=False) as response:
                        response.raise_for_status()  # Raise an exception for HTTP errors
                        
                        content_type = response.headers.get('Content-Type', '').lower()
                        logger.debug(f"Content-Type: {content_type}")
                        
                        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                            logger.info("Downloading PDF content...")
                            pdf_content = io.BytesIO(await response.read())
                            text = self._extract_text_from_pdf(pdf_content)
                            if not text.strip():
                                logger.warning("Extracted text from PDF is empty")
                            return text
                        else:
                            logger.info("Downloading HTML content...")
                            html = await response.text()
                            text = self._extract_text_from_html(html)
                            if not text.strip():
                                logger.warning("Extracted text from HTML is empty")
                            return text
                            
                except aiohttp.ClientError as e:
                    logger.error(f"HTTP error while fetching {url}: {str(e)}")
                    return ""
                except asyncio.TimeoutError:
                    logger.error(f"Timeout while fetching {url}")
                    return ""
                except Exception as e:
                    logger.error(f"Unexpected error while fetching {url}: {str(e)}")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error in _fetch_paper_content: {str(e)}", exc_info=True)
            return ""
    
    @staticmethod
    def _extract_text_from_pdf(pdf_file: io.BytesIO) -> str:
        """Extract text from PDF file with improved error handling."""
        try:
            logger.info("Attempting to extract text from PDF...")
            
            # Ensure we're at the start of the file
            pdf_file.seek(0)
            
            try:
                # First try with PyPDF2
                reader = PyPDF2.PdfReader(pdf_file)
                text = []
                
                for i, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text.append(page_text)
                        else:
                            logger.warning(f"No text extracted from page {i}")
                    except Exception as page_e:
                        logger.warning(f"Error extracting text from page {i}: {str(page_e)}")
                
                result = '\n'.join(text).strip()
                if not result:
                    logger.warning("No text extracted from PDF using PyPDF2")
                    
                    # Try alternative approach with pdfplumber if available
                    try:
                        import pdfplumber
                        pdf_file.seek(0)
                        with pdfplumber.open(pdf_file) as pdf:
                            alt_text = []
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    alt_text.append(page_text)
                            result = '\n'.join(alt_text).strip()
                            if result:
                                logger.info("Successfully extracted text using pdfplumber")
                    except ImportError:
                        logger.warning("pdfplumber not available, using PyPDF2 results")
                    except Exception as plumber_e:
                        logger.warning(f"Error using pdfplumber: {str(plumber_e)}")
                
                return result if result else ""
                
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                return ""
                
        except Exception as e:
            logger.error(f"Unexpected error in _extract_text_from_pdf: {str(e)}", exc_info=True)
            return ""
    
    @staticmethod
    def _extract_text_from_html(html: str) -> str:
        """Extract text from HTML content."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of specified size."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    @staticmethod
    def _parse_json_response(text: str) -> Dict:
        """Parse JSON response from model, handling markdown code blocks."""
        try:
            # Extract JSON from markdown code block if present
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {text}")
            return {}
    
    @staticmethod
    def _is_url(text: str) -> bool:
        """Check if text is a URL."""
        return text.strip().lower().startswith(("http://", "https://"))
