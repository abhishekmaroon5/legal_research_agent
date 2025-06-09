from typing import Dict, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io

load_dotenv()

class ArgumentExtractor:
    def __init__(self, model=None):
        # Use provided model or create default Gemini model
        if model:
            self.model = model
        else:
            # Configure Gemini
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')

    async def extract_arguments(self, base_paper_content_or_url: str, research_angle: str) -> Dict:
        """
        Extract core arguments from the base paper and research angle
        Accepts either the full text content or a URL. If the input looks like a URL, fetch it; otherwise, use as text.
        """
        # 1. Use the text directly if it's not a URL
        if base_paper_content_or_url.strip().lower().startswith(("http://", "https://")):
            paper_content = await self._fetch_paper_content(base_paper_content_or_url)
        else:
            paper_content = base_paper_content_or_url
        
        # 2. Extract core arguments using Gemini
        base_arguments = await self._extract_base_arguments(paper_content)
        
        # 3. Analyze research angle
        research_analysis = await self._analyze_research_angle(research_angle, base_arguments)
        
        return {
            "base_arguments": base_arguments,
            "research_analysis": research_analysis,
            "combined_context": {
                "base_paper": base_arguments,
                "new_angle": research_analysis
            }
        }

    async def _fetch_paper_content(self, url: str) -> str:
        """Fetch and parse paper content from URL"""
        try:
            response = requests.get(url)
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/pdf' in content_type:
                # Handle PDF
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            else:
                # Handle HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text()
        except Exception as e:
            raise Exception(f"Error fetching paper content: {str(e)}")

    async def _extract_base_arguments(self, paper_content: str) -> Dict:
        """Extract core arguments from the base paper using Gemini"""
        prompt = f"""
        Analyze the following legal paper and extract:
        1. Main thesis/argument
        2. Key supporting points
        3. Methodology used
        4. Key findings
        5. Limitations identified
        
        Paper content:
        {paper_content[:4000]}  # Limit content length for API
        
        Provide the analysis in a structured format.
        """
        
        response = await self.model.ainvoke(prompt)
        return self._parse_argument_response(response.content)

    async def _analyze_research_angle(self, research_angle: str, base_arguments: Dict) -> Dict:
        """Analyze how the research angle relates to or diverges from base arguments"""
        prompt = f"""
        Analyze how the following research angle relates to or diverges from the base paper:
        
        Base paper arguments:
        {base_arguments}
        
        New research angle:
        {research_angle}
        
        Provide:
        1. How it builds upon the base paper
        2. Where it diverges
        3. Potential gaps it addresses
        4. Suggested focus areas
        """
        
        response = await self.model.ainvoke(prompt)
        return self._parse_angle_response(response.content)

    def _parse_argument_response(self, response: str) -> Dict:
        """Parse the Gemini response into structured format"""
        try:
            # Clean up response
            response = response.replace("**", "").strip()
            
            # Split response into sections
            sections = response.split('\n\n')
            
            # Initialize result dictionary
            result = {
                "thesis": "",
                "supporting_points": [],
                "methodology": "",
                "findings": [],
                "limitations": []
            }
            
            current_section = None
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                # Check for section headers
                if "1. Main thesis/argument" in section or "Thesis" in section:
                    current_section = "thesis"
                    content = section.split(":", 1)[-1].strip()
                    result["thesis"] = self._clean_text(content)
                elif "2. Key supporting points" in section or "Supporting points" in section:
                    current_section = "supporting_points"
                    content = section.split(":", 1)[-1].strip()
                    points = [p.strip() for p in content.split('\n') if p.strip()]
                    result["supporting_points"] = [self._clean_text(p) for p in points]
                elif "3. Methodology used" in section or "Methodology" in section:
                    current_section = "methodology"
                    content = section.split(":", 1)[-1].strip()
                    result["methodology"] = self._clean_text(content)
                elif "4. Key findings" in section or "Findings" in section:
                    current_section = "findings"
                    content = section.split(":", 1)[-1].strip()
                    findings = [f.strip() for f in content.split('\n') if f.strip()]
                    result["findings"] = [self._clean_text(f) for f in findings]
                elif "5. Limitations identified" in section or "Limitations" in section:
                    current_section = "limitations"
                    content = section.split(":", 1)[-1].strip()
                    limitations = [l.strip() for l in content.split('\n') if l.strip()]
                    result["limitations"] = [self._clean_text(l) for l in limitations]
                elif current_section and current_section != "thesis" and current_section != "methodology":
                    # Append to current list section
                    if isinstance(result[current_section], list):
                        result[current_section].append(self._clean_text(section))
            
            return result
        except Exception as e:
            print(f"Error parsing argument response: {str(e)}")
            return {
                "thesis": self._clean_text(response),  # Return full response as thesis if parsing fails
                "supporting_points": [],
                "methodology": "",
                "findings": [],
                "limitations": []
            }

    def _parse_angle_response(self, response: str) -> Dict:
        """Parse the research angle analysis into structured format"""
        try:
            # Clean up response
            response = response.replace("**", "").strip()
            
            # Split response into sections
            sections = response.split('\n\n')
            
            # Initialize result dictionary
            result = {
                "builds_upon": [],
                "diverges": [],
                "gaps_addressed": [],
                "focus_areas": []
            }
            
            current_section = None
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                # Check for section headers
                if "1. How it builds upon" in section or "Builds upon" in section:
                    current_section = "builds_upon"
                    content = section.split(":", 1)[-1].strip()
                    points = [p.strip() for p in content.split('\n') if p.strip()]
                    result["builds_upon"] = [self._clean_text(p) for p in points]
                elif "2. Where it diverges" in section or "Diverges" in section:
                    current_section = "diverges"
                    content = section.split(":", 1)[-1].strip()
                    points = [p.strip() for p in content.split('\n') if p.strip()]
                    result["diverges"] = [self._clean_text(p) for p in points]
                elif "3. Potential gaps" in section or "Gaps addressed" in section:
                    current_section = "gaps_addressed"
                    content = section.split(":", 1)[-1].strip()
                    points = [p.strip() for p in content.split('\n') if p.strip()]
                    result["gaps_addressed"] = [self._clean_text(p) for p in points]
                elif "4. Suggested focus areas" in section or "Focus areas" in section:
                    current_section = "focus_areas"
                    content = section.split(":", 1)[-1].strip()
                    points = [p.strip() for p in content.split('\n') if p.strip()]
                    result["focus_areas"] = [self._clean_text(p) for p in points]
                elif current_section:
                    # Append to current list section
                    if isinstance(result[current_section], list):
                        result[current_section].append(self._clean_text(section))
            
            return result
        except Exception as e:
            print(f"Error parsing angle response: {str(e)}")
            return {
                "builds_upon": [],
                "diverges": [],
                "gaps_addressed": [],
                "focus_areas": []
            }

    def _clean_text(self, text: str) -> str:
        """Clean up text by removing formatting artifacts and extra whitespace"""
        # Remove markdown-style formatting
        text = text.replace("*", "").replace("_", "").strip()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove bullet points and numbering
        text = text.lstrip("•-*1234567890. ")
        
        return text
