# Legal Research Agent

An AI-powered legal research assistant that helps streamline legal research and analysis through an intuitive web interface.

## Overview

This project provides an intelligent system for legal research, document analysis, and case law exploration. The web interface makes it easy to upload legal documents and get research insights.

## Features

- 📄 Legal document analysis
- 🔍 Case law research
- 📝 Document summarization
- ⚖️ Legal precedent tracking
- 🌐 User-friendly web interface
- 🔑 Keyword generation from documents
- 📊 Source relevance scoring

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd legal_research_agent
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit the `.env` file and add your API keys.

## Running the Application

Start the web interface:
```bash
python app.py
```

The application will start a local web server. Open your browser and navigate to:
```
http://localhost:7860
```

## Usage

1. Click "Upload Legal Document" to upload a PDF document
2. Enter your research question or angle in the text box
3. Click "Start Research" to begin analysis
4. View the results including:
   - Extracted base arguments
   - Research analysis
   - Generated keywords
   - Relevant sources

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add license information here]

## Contact

[Add contact information here]
